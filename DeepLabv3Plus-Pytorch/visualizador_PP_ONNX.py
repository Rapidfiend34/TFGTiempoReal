import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import random
import argparse
import numpy as np
import torch.nn.functional as F

from semantic_validation_postprocessor import (
    SemanticValidationPostProcessor,
    apply_semantic_validation,
    draw_validated_boxes
)

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from datasets import VOCSegmentation, Cityscapes, mapillary
from datasets.mapillary import MapillaryVistas, MapillaryTransform
from torchvision import transforms as T

from ultralytics import YOLO
import torch
import torch.nn as nn
from collections import OrderedDict, deque
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import time
import gc
import onnxruntime as ort

# Importar solo las clases ONNX necesarias para DeepLab
from convert_to_onnx import ONNXInferenceEngine, convert_deeplab_model

np.float=float
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

# Configuración global de optimización
PROCESS_EVERY_N_FRAMES = 10  # Procesar 1 de cada N frames
UPDATE_VISUALIZATION_EVERY_N_FRAMES = 10  # Actualizar visualización cada N frames
CLASSIFICATION_TIMEOUT = 5.0  # Segundos antes de reclasificar
INPUT_RESOLUTION = (640, 480)  # Resolución de entrada reducida
CLASSIFICATION_RESOLUTION = 160  # Resolución para clasificación
DETECTION_RESOLUTION = 416  # Resolución para detección
CONFIDENCE_THRESHOLD = 0.2  # Umbral de confianza para detección

# Resolución para segmentación semántica
SEMANTIC_RESOLUTION = (192, 192)  # Resolución interna para DeepLab ONNX

# Diccionario global para almacenar clasificaciones
track_classifications = {}

# Mapeo de clases de interés de Mapillary
MAPILLARY_CLASSES = {
    'road': 2,          # construction--flat--road
    'sidewalk': 3,      # construction--flat--sidewalk
    'person': 4,        # human--person--individual
    'rider': 5,         # human--rider--other-rider
    'bicycle': 9,       # object--vehicle--bicycle
    'bus': 10,         # object--vehicle--bus
    'car': 11,         # object--vehicle--car
    'motorcycle': 12,   # object--vehicle--motorcycle
    'truck': 13        # object--vehicle--truck
}

# Colores para visualización
COLOR_MAP = {
    2: (128, 64, 128),   # Road: azul grisáceo
    3: (244, 35, 232),   # Sidewalk: rosa
    4: (220, 20, 60),    # Person: rojo
    5: (255, 0, 200),    # Rider: magenta
    9: (119, 11, 32),    # Bicycle: rojo oscuro
    10: (0, 60, 100),    # Bus: azul oscuro
    11: (0, 0, 142),     # Car: azul
    12: (0, 0, 230),     # Motorcycle: azul brillante
    13: (0, 0, 70),      # Truck: azul muy oscuro
    255: (0, 0, 0)       # Fondo: negro
}

class DeepLabONNXManager:
    """Gestor específico para DeepLab ONNX"""
    
    def __init__(self):
        self.onnx_engine = ONNXInferenceEngine()
        self.deeplab_loaded = False
        
    def load_deeplab_model(self, model_path):
        """Carga solo el modelo DeepLab ONNX"""
        print("Cargando modelo DeepLab ONNX...")
        
        try:
            if os.path.exists(model_path):
                self.onnx_engine.load_model(model_path, 'deeplab')
                self.deeplab_loaded = True
                print("Modelo DeepLab ONNX cargado exitosamente")
                return True
            else:
                print(f"Modelo DeepLab ONNX no encontrado: {model_path}")
                return False
                
        except Exception as e:
            print(f"Error cargando modelo DeepLab ONNX: {e}")
            return False
    
    def segment_image(self, image_tensor, target_size):
        """
        Segmentación semántica con DeepLab ONNX
        
        Args:
            image_tensor: Tensor de entrada
            target_size: Tamaño objetivo (width, height) para redimensionar el resultado
        
        Returns:
            Máscara semántica redimensionada al tamaño objetivo
        """
        if not self.deeplab_loaded:
            return None
            
        try:
            # Inferencia ONNX
            outputs = self.onnx_engine.predict_deeplab('deeplab', image_tensor)
            
            if outputs is not None:
                # Obtener predicción
                pred = np.argmax(outputs, axis=1)[0]  # Shape: (192, 192)
                
                # Redimensionar al tamaño objetivo usando interpolación nearest neighbor
                # para mantener las etiquetas de clase intactas
                pred_resized = cv2.resize(pred.astype(np.uint8), target_size, 
                                        interpolation=cv2.INTER_NEAREST)
                
                return pred_resized
            else:
                # Fallback: crear máscara vacía del tamaño correcto
                return np.zeros(target_size[::-1], dtype=np.uint8)  # target_size es (w,h), necesitamos (h,w)
                
        except Exception as e:
            print(f"Error en segmentación ONNX: {e}")
            # Fallback: crear máscara vacía del tamaño correcto
            return np.zeros(target_size[::-1], dtype=np.uint8)

def draw_boxes(image, results):
    """
    Dibuja las bounding boxes sobre la imagen con colores diferentes según la clase
    """
    img_with_boxes = image.copy()

    # Definir colores para cada clase
    class_colors = {
        0: (255, 0, 0),    # Persona - Rojo
        1: (0, 255, 0),    # Bicicleta - Verde
        2: (0, 0, 255),    # Coche - Azul
        3: (255, 255, 0)   # Moto - Amarillo
    }

    # Definir nombres de las clases
    class_names = {
        0: 'Person',
        1: 'Bicycle',
        2: 'Car',
        3: 'Motorcycle'
    }

    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Obtener color según la clase
        color = class_colors.get(cls, (0, 255, 0))

        # Dibujar rectángulo
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

        # Añadir texto con clase y confidencia
        class_name = class_names.get(cls, str(cls))
        label = f'{class_name}: {conf:.2f}'

        # Añadir fondo al texto para mejor visibilidad
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_with_boxes, (x1, y1-h-10), (x1+w, y1), color, -1)
        cv2.putText(img_with_boxes, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img_with_boxes

def clear_memory():
    """Libera memoria GPU si está disponible"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        gc.collect()

def load_deeplab_model_pytorch(opts, device):
    """Carga modelo DeepLab PyTorch (para conversión a ONNX si es necesario)"""
    opts.num_classes = 114
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes,
                                                output_stride=opts.output_stride)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        try:
            checkpoint = torch.load(opts.ckpt, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            print("Checkpoint DeepLab cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar checkpoint: {e}")
            raise

    model = model.to(device)
    return model, opts

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--input", type=str, default='0',
                        help="path to input video file or camera index (0 for webcam)")
    parser.add_argument("--input_type", type=str, default='camera',
                        choices=['video', 'camera'], help='Type of input source')
    parser.add_argument("--dataset", type=str, default='mapillary',
                        choices=['voc', 'cityscapes', 'mapillary'], help='Name of training set')
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--mapillary_version", type=str, default='v2.0',
                        choices=['v1.2', 'v2.0'], help='Mapillary version')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and 
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")

    # ONNX Options - Solo para DeepLab
    parser.add_argument("--use_deeplab_onnx", action='store_true', default=True,
                        help="Use ONNX for DeepLab inference")
    parser.add_argument("--onnx_deeplab", type=str, default='models_onnx/deeplab.onnx',
                        help="Path to ONNX DeepLab model")

    # Output Options
    parser.add_argument("--save_video", action='store_true', default=False,
                        help="save output video")
    parser.add_argument("--output_dir", type=str, default='results',
                        help="directory to save results")

    return parser

def create_color_mask(pred):
    """Crea una máscara de color basada en la predicción"""
    height, width = pred.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Usar vectorización de numpy para mejorar rendimiento
    for class_id, color in COLOR_MAP.items():
        mask = pred == class_id
        color_mask[mask] = color

    return color_mask

class ByteTrackArgs:
    track_thresh = 0.3
    match_thresh = 0.7
    track_buffer = 30
    min_box_area = 10
    mot20 = False

def process_frame_hybrid(frame, deeplab_onnx_manager, device, transform, yolo_model, classifier_model, semantic_processor):
    """Procesa un frame con YOLO PyTorch + DeepLab ONNX con escalado correcto"""
    start_time = time.time()
    current_time = start_time

    # Reducir tamaño y convertir a tensor una sola vez
    frame = cv2.resize(frame, INPUT_RESOLUTION)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Guardar dimensiones del frame para escalado correcto
    frame_height, frame_width = frame_rgb.shape[:2]
    target_size = (frame_width, frame_height)  # (width, height) para cv2.resize

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Preparar imagen para segmentación semántica
    img_pil = Image.fromarray(frame_rgb)
    img = transform(img_pil).unsqueeze(0)
    
    # Redimensionar a la resolución de segmentación para inferencia ONNX
    img = F.interpolate(img, size=SEMANTIC_RESOLUTION, mode='bilinear', align_corners=True)
    
    # Convertir a numpy para ONNX
    img_numpy = img.cpu().numpy()
    
    # Segmentación con ONNX - ahora con escalado correcto
    semantic_mask = deeplab_onnx_manager.segment_image(img_numpy, target_size)
    
    if semantic_mask is None:
        # Fallback: crear máscara vacía del tamaño correcto
        semantic_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Crear máscara de color para visualización
    color_mask = create_color_mask(semantic_mask)

    # Detección con YOLO PyTorch (mantener original)
    results = yolo_model.track(
        source=frame_rgb,
        conf=CONFIDENCE_THRESHOLD,
        classes=[0, 1, 2, 3],  # person, bicycle, car, motorcycle
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        imgsz=DETECTION_RESOLUTION,
        augment=False,
        agnostic_nms=True
    )

    # Aplicar validación semántica con la máscara correctamente escalada
    validated_detections = apply_semantic_validation(results, semantic_mask, semantic_processor)

    # Procesar clasificaciones para detecciones válidas (mantener PyTorch)
    for detection in validated_detections:
        track_id = detection['track_id']
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        # Solo clasificar si es necesario y la detección es válida
        if (track_id not in track_classifications or
            current_time - track_classifications[track_id]['time'] > CLASSIFICATION_TIMEOUT):

            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                roi_pil = Image.fromarray(roi)
                class_results = classifier_model.predict(
                    roi_pil,
                    imgsz=CLASSIFICATION_RESOLUTION,
                    verbose=False
                )
                pred_class = class_results[0].names[class_results[0].probs.top1]
                pred_prob = class_results[0].probs.top1conf.item()

                track_classifications[track_id] = {
                    'class': pred_class,
                    'conf': pred_prob,
                    'time': current_time
                }

    # Crear visualización con detecciones validadas
    combined_visualization = draw_validated_boxes(frame_rgb, validated_detections, track_classifications)

    # Calcular y mostrar FPS y estadísticas
    fps = 1.0 / (time.time() - start_time)
    total_detections = len(results[0].boxes) if results[0].boxes.id is not None else 0
    valid_detections = len(validated_detections)

    cv2.putText(combined_visualization, f'FPS: {fps:.1f} (Hybrid)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_visualization, f'Detections: {total_detections} -> {valid_detections}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(combined_visualization, f'Semantic: {semantic_mask.shape}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame_rgb, color_mask, combined_visualization, validated_detections

def main():
    opts = get_argparser().parse_args()
    frame_count = 0
    
    print("=== Iniciando modo híbrido: YOLO PyTorch + DeepLab ONNX (Escalado Corregido) ===")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    # Configurar dispositivo
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Inicializar gestor DeepLab ONNX
    deeplab_onnx_manager = DeepLabONNXManager()
    
    # Verificar si el modelo DeepLab ONNX existe
    if not os.path.exists(opts.onnx_deeplab):
        print(f"Modelo DeepLab ONNX no encontrado: {opts.onnx_deeplab}")
        if opts.ckpt:
            print("Convirtiendo DeepLab a ONNX...")
            try:
                model, _ = load_deeplab_model_pytorch(opts, device)
                convert_deeplab_model(model, opts.onnx_deeplab)
            except Exception as e:
                print(f"Error convirtiendo DeepLab: {e}")
                return
        else:
            print("No se puede convertir DeepLab sin checkpoint. Especifica --ckpt")
            return
    
    # Cargar modelo DeepLab ONNX
    if not deeplab_onnx_manager.load_deeplab_model(opts.onnx_deeplab):
        print("Error cargando modelo DeepLab ONNX. Saliendo...")
        return

    # Preasignar memoria GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        
    # Cargar modelos YOLO PyTorch (mantener original)
    try:
       with torch.cuda.amp.autocast():
            yolo_model = YOLO('yolov8n-seg.pt', verbose=False)  # Usar versión nano
            classifier_model = YOLO(r'C:\\Users\\aaron\\Documents\\Año4\\TFG\\Classifier\\runs\\classify\\Mejores_matrix\\weights\\best.pt', verbose=False)
            yolo_model.to(device).half()  # Usar precisión FP16
            classifier_model.to(device).half()
    except Exception as e:
        print(f"Error loading YOLO models: {e}")
        return

    # Configurar transformaciones
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    ])

    if opts.input_type == 'camera':
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RESOLUTION[1])
            print("Cámara iniciada correctamente")
        except Exception as e:
            print(f"Error al abrir la cámara: {e}")
            return
    else:
        try:
            cap = cv2.VideoCapture(opts.input)
            if not cap.isOpened():
                print("Error: No se pudo abrir el video")
                return
            print("Video cargado correctamente")
        except Exception as e:
            print(f"Error al abrir el video: {e}")
            return

    # Obtener dimensiones del frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configurar FPS
    fps = 30 if opts.input_type == 'camera' else int(cap.get(cv2.CAP_PROP_FPS))

    # Configurar writer para guardar video
    if opts.save_video:
        os.makedirs(opts.output_dir, exist_ok=True)
        output_path = os.path.join(opts.output_dir,
                                 f'output_{opts.input_type}_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width*3, frame_height))

    plt.ion()
    fig = plt.figure(figsize=(18, 6))
    semantic_processor = SemanticValidationPostProcessor(
        min_semantic_consistency=0.4  # Ajustar según necesidades
    )
    detection_stats = {
        'total_detections': 0,
        'valid_detections': 0,
        'filtered_detections': 0
    }
    
    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    continue

                # Procesar frame con modo híbrido
                frame_rgb, color_mask, combined_viz, validated_detections = process_frame_hybrid(
                    frame, deeplab_onnx_manager, device, transform, yolo_model, classifier_model, semantic_processor
                )

                # Actualizar estadísticas
                detection_stats['valid_detections'] += len(validated_detections)

                # Actualizar visualización
                if frame_count % UPDATE_VISUALIZATION_EVERY_N_FRAMES == 0:
                    plt.clf()
                    plt.subplot(1, 3, 1)
                    plt.imshow(frame_rgb)
                    plt.title(f"Original ({frame_rgb.shape[1]}x{frame_rgb.shape[0]})")
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(color_mask)
                    plt.title(f"Segmentación ({color_mask.shape[1]}x{color_mask.shape[0]})")
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(combined_viz)
                    plt.title(f"Detección Validada ({combined_viz.shape[1]}x{combined_viz.shape[0]})")
                    plt.axis('off')

                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.001)

                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'capture_hybrid_{timestamp}.jpg',
                              cv2.cvtColor(combined_viz, cv2.COLOR_RGB2BGR))
                elif key == ord('p'):
                    cv2.waitKey(0)

                # Limpiar memoria periódicamente
                if frame_count % 30 == 0:
                    clear_memory()

                # Mostrar estadísticas cada 100 frames
                if frame_count % 100 == 0:
                    valid_ratio = (detection_stats['valid_detections'] /
                                 max(1, detection_stats['total_detections'])) * 100
                    print(f"Frame {frame_count}: Semantic mask shape: {color_mask.shape}")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
    finally:
        # Mostrar estadísticas finales
        print("\\n=== Estadísticas Finales (Híbrido Corregido) ===")
        print(f"Total detecciones: {detection_stats['total_detections']}")
        print(f"Detecciones válidas: {detection_stats['valid_detections']}")
        if detection_stats['total_detections'] > 0:
            valid_percentage = (detection_stats['valid_detections'] /
                              detection_stats['total_detections']) * 100
            print(f"Porcentaje de validez: {valid_percentage:.2f}%")

        cap.release()
        if opts.save_video:
            out.release()
        plt.ioff()
        plt.close('all')
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Controles:")
    print("  'q' - Salir")
    print("  's' - Capturar frame")
    print("  'p' - Pausar/Reanudar")
    print("\\nModo: YOLO PyTorch + DeepLab ONNX (Escalado Corregido)")
    main()