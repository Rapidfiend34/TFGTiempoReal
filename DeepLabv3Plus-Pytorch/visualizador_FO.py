import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
#from torch.utils.data import dataset
#from tqdm import tqdm
import network
""" import utils
import random """
import argparse
import numpy as np
import torch.nn.functional as F


from semantic_validation_postprocessor import (
    SemanticValidationPostProcessor,
    apply_semantic_validation,
    draw_validated_boxes
)

#from torch.utils import data
""" from datasets import VOCSegmentation, Cityscapes, cityscapes
from datasets import VOCSegmentation, Cityscapes, mapillary
from datasets.mapillary import MapillaryVistas, MapillaryTransform """
from torchvision import transforms as T

from ultralytics import YOLO
import torch
#import torch.nn as nn
from collections import OrderedDict, deque
from PIL import Image

from glob import glob
import time
import gc
np.float=float
#from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

# Configuración global de optimización
PROCESS_EVERY_N_FRAMES = 10  # Procesar 1 de cada N frames
UPDATE_VISUALIZATION_EVERY_N_FRAMES = 10  # Actualizar visualización cada N frames
CLASSIFICATION_TIMEOUT = 5.0  # Segundos antes de reclasificar
INPUT_RESOLUTION = (640, 480)  # Resolución de entrada reducida
CLASSIFICATION_RESOLUTION = 224  # Resolución para clasificación
DETECTION_RESOLUTION = 416  # Resolución para detección
CONFIDENCE_THRESHOLD = 0.2  # Umbral de confianza para detección

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

def create_opencv_visualization(frame_rgb, color_mask, combined_viz):
    """Visualización con OpenCV en lugar de matplotlib"""
    # Redimensionar para visualización
    h, w = frame_rgb.shape[:2]
    display_h, display_w = 480, 640  # Tamaño fijo para visualización
    
    frame_display = cv2.resize(frame_rgb, (display_w, display_h))
    mask_display = cv2.resize(color_mask, (display_w, display_h))
    combined_display = cv2.resize(combined_viz, (display_w, display_h))
    
    # Concatenar horizontalmente
    combined_display_bgr = cv2.cvtColor(combined_display, cv2.COLOR_RGB2BGR)
    mask_display_bgr = cv2.cvtColor(mask_display, cv2.COLOR_RGB2BGR)
    frame_display_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
    
    visualization = np.hstack([frame_display_bgr, mask_display_bgr, combined_display_bgr])
    
    # Añadir títulos
    cv2.putText(visualization, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(visualization, 'Semantic', (display_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(visualization, 'Detection', (display_w*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return visualization
def get_video_rotation(video_path):
    """
    Detecta la rotación del video usando ffprobe o metadatos
    """
    try:
        import subprocess
        import json
        
        # Intentar usar ffprobe para obtener metadatos de rotación
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    # Buscar tag de rotación
                    tags = stream.get('tags', {})
                    rotation = tags.get('rotate', '0')
                    return int(rotation)
    except:
        pass
    
    return 0

def correct_frame_rotation(frame, rotation_angle):
    """
    Corrige la rotación del frame basado en el ángulo detectado
    """
    if rotation_angle == 0:
        return frame
    elif rotation_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        # Para ángulos arbitrarios
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
        return cv2.warpAffine(frame, rotation_matrix, (width, height))

def detect_video_orientation(cap):
    """
    Detecta si el video es vertical analizando las primeras frames
    """
    # Leer algunos frames para determinar orientación
    frame_count = 0
    vertical_count = 0
    horizontal_count = 0
    
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    for _ in range(min(10, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width = frame.shape[:2]
        if height > width:
            vertical_count += 1
        else:
            horizontal_count += 1
        frame_count += 1
    
    # Volver al inicio del video
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
    
    is_vertical = vertical_count > horizontal_count
    print(f"Video detectado como: {'VERTICAL' if is_vertical else 'HORIZONTAL'}")
    print(f"Frames analizados: {frame_count}, Vertical: {vertical_count}, Horizontal: {horizontal_count}")
    
    return is_vertical

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

def load_model(opts, device):
    # Crear modelo con 114 clases (número correcto para Mapillary)
    opts.num_classes = 114
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes,
                                                output_stride=opts.output_stride)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        try:
            checkpoint = torch.load(opts.ckpt, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            print("Checkpoint cargado exitosamente")
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

    # Video Rotation Options
    parser.add_argument("--auto_rotate", action='store_true', default=True,
                        help="Automatically detect and correct video rotation")
    parser.add_argument("--force_rotation", type=int, default=None,
                        choices=[0, 90, 180, 270], help="Force specific rotation angle")

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
def create_color_mask_optimized(pred):
    """Versión ultra-optimizada con vectorización completa"""
    height, width = pred.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Crear array de colores indexado
    color_array = np.array([COLOR_MAP.get(i, (0, 0, 0)) for i in range(256)], dtype=np.uint8)
    
    # Aplicar colores de una vez usando indexación avanzada
    color_mask = color_array[pred]
    
    return color_mask
class ByteTrackArgs:
    track_thresh = 0.3
    match_thresh = 0.7
    track_buffer = 30
    min_box_area = 10
    mot20 = False

def process_frame(frame, model, device, transform, yolo_model, classifier_model, semantic_processor, rotation_angle=0):
    """Procesa un frame del video con validación semántica y corrección de rotación"""
    start_time = time.time()
    current_time = start_time

    # Aplicar corrección de rotación si es necesario
    if rotation_angle != 0:
        frame = correct_frame_rotation(frame, rotation_angle)

    # Reducir tamaño y convertir a tensor una sola vez
    frame = cv2.resize(frame, INPUT_RESOLUTION)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    st_time=time.time()
    # Procesamiento batch para DeepLab
    with torch.cuda.amp.autocast():
        img_pil = Image.fromarray(frame_rgb)
        img = transform(img_pil).unsqueeze(0)
        img = F.interpolate(img, size=(192, 192), mode='bilinear', align_corners=True)
        img = img.to(device)
        outputs = model(img)

    pred = outputs.max(1)[1].cpu().numpy()[0]

    # Redimensionar la máscara semántica al tamaño del frame
    semantic_mask = cv2.resize(pred, (frame_rgb.shape[1], frame_rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    color_mask = create_color_mask_optimized(semantic_mask)
    nd_time=time.time()
    print("Tiempo semantica: ",nd_time-st_time)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    st_time=time.time()
    # Detección con YOLO
    results = yolo_model.track(
        source=frame_rgb,
        conf=CONFIDENCE_THRESHOLD,
        classes=[0],  # person, bicycle, car, motorcycle
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        imgsz=DETECTION_RESOLUTION,
        augment=False,
        agnostic_nms=False,
        device=0
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    nd_time=time.time()
    print("Tiempo instanciacion: ",nd_time-st_time)
    # Aplicar validación semántica
    validated_detections = apply_semantic_validation(results, semantic_mask, semantic_processor)
    
    st_time=time.time()
    # Procesar clasificaciones para detecciones válidas
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
                    verbose=False,
                    device=0
                )
                pred_class = class_results[0].names[class_results[0].probs.top1]
                pred_prob = class_results[0].probs.top1conf.item()

                track_classifications[track_id] = {
                    'class': pred_class,
                    'conf': pred_prob,
                    'time': current_time
                }
    nd_time=time.time()
    print("Tiempo clasificacion: ",nd_time-st_time)

    # Crear visualización con detecciones validadas
    combined_visualization = draw_validated_boxes(frame_rgb, validated_detections, track_classifications)

    # Calcular y mostrar FPS y estadísticas
    fps = 1.0 / (time.time() - start_time)
    total_detections = len(results[0].boxes) if results[0].boxes.id is not None else 0
    valid_detections = len(validated_detections)

    cv2.putText(combined_visualization, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_visualization, f'Detections: {total_detections} -> {valid_detections}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Mostrar información de rotación si se aplicó
    if rotation_angle != 0:
        cv2.putText(combined_visualization, f'Rotation: {rotation_angle}°', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return frame_rgb, color_mask, combined_visualization, validated_detections

def main():
    opts = get_argparser().parse_args()
    frame_count = 0
    rotation_angle = 0
    
    print("=== Sistema con Corrección Automática de Rotación ===")
    
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

    # Preasignar memoria GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
    
    # Cargar modelos
    try:
       with torch.cuda.amp.autocast():
            yolo_model = YOLO('yolov8n-seg.pt', verbose=False)  # Usar versión nano
            classifier_model = YOLO(r'C:\\Users\\aaron\\Documents\\Año4\\TFG\\Classifier\\runs\\classify\\Mejores_matrix\\weights\\best.pt', verbose=False)
            yolo_model.to(device).half()  # Usar precisión FP16
            classifier_model.to(device).half()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Cargar modelo DeepLab
    model, opts = load_model(opts, device)
    model.eval()

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
            
            # Detectar rotación del video
            if opts.force_rotation is not None:
                rotation_angle = opts.force_rotation
                print(f"Rotación forzada: {rotation_angle}°")
            elif opts.auto_rotate:
                # Intentar detectar rotación por metadatos
                video_rotation = get_video_rotation(opts.input)
                if video_rotation != 0:
                    rotation_angle = video_rotation
                    print(f"Rotación detectada por metadatos: {rotation_angle}°")
                else:
                    # Detectar orientación analizando frames
                    is_vertical = detect_video_orientation(cap)
                    if is_vertical:
                        rotation_angle = 0  # Mantener vertical como está
                        print("Video vertical detectado - manteniendo orientación original")
                    else:
                        rotation_angle = 0
                        print("Video horizontal detectado")
            
        except Exception as e:
            print(f"Error al abrir el video: {e}")
            return

    # Obtener dimensiones del frame (después de posible rotación)
    ret, test_frame = cap.read()
    if ret:
        if rotation_angle != 0:
            test_frame = correct_frame_rotation(test_frame, rotation_angle)
        frame_height, frame_width = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Volver al inicio
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Dimensiones del frame (después de rotación): {frame_width}x{frame_height}")

    # Configurar FPS
    fps = 30 if opts.input_type == 'camera' else int(cap.get(cv2.CAP_PROP_FPS))

    # Configurar writer para guardar video
    if opts.save_video:
        os.makedirs(opts.output_dir, exist_ok=True)
        output_path = os.path.join(opts.output_dir,
                                 f'output_{opts.input_type}_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width*3, frame_height))

    semantic_processor = SemanticValidationPostProcessor(
        min_semantic_consistency=0.4  # Ajustar según necesidades
    )
    detection_stats = {
        'total_detections': 0,
        'valid_detections': 0,
        'filtered_detections': 0
    }
    # Inicializar variables para el rendimiento
    fps_history = deque(maxlen=30)  # Para calcular FPS promedio
    frame_start_time = time.time()

    # Configurar ventana de visualización optimizada
    cv2.namedWindow('Video Processing Optimized', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Processing Optimized', 1440, 360)  # Tamaño optimizado
    try:
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    continue

                # Procesar frame con validación semántica y corrección de rotación
                frame_rgb, color_mask, combined_viz, validated_detections = process_frame(
                    frame, model, device, transform, yolo_model, classifier_model, 
                    semantic_processor, rotation_angle
                )

                # Actualizar estadísticas
                detection_stats['valid_detections'] += len(validated_detections)

                # Actualizar visualización
                if frame_count % UPDATE_VISUALIZATION_EVERY_N_FRAMES == 0:
                    # Crear visualización con OpenCV (mucho más rápido que matplotlib)
                    visualization = create_opencv_visualization(frame_rgb, color_mask, combined_viz)
                    
                    # Añadir información de rendimiento
                    frame_fps = 1.0 / (time.time() - frame_start_time) if 'frame_start_time' in locals() else 0
                    fps_history.append(frame_fps)
                    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                    
                    # Añadir información adicional en la visualización
                    cv2.putText(visualization, f'Frame: {frame_count}', (10, visualization.shape[0] - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(visualization, f'Rotation: {rotation_angle}°', (10, visualization.shape[0] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(visualization, f'Avg FPS: {avg_fps:.1f}', (10, visualization.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Mostrar la visualización
                    cv2.imshow('Video Processing Optimized', visualization)
                    
                    # Guardar video si está habilitado
                    if 'out' in locals() and opts.save_video:
                        out.write(visualization)

                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # Guardar la visualización completa en lugar de solo combined_viz
                    if 'visualization' in locals():
                        cv2.imwrite(f'capture_corrected_{timestamp}.jpg', visualization)
                        print(f"Visualización completa capturada: capture_corrected_{timestamp}.jpg")
                    else:
                        # Fallback si no hay visualización disponible
                        cv2.imwrite(f'capture_corrected_{timestamp}.jpg',
                                cv2.cvtColor(combined_viz, cv2.COLOR_RGB2BGR))
                        print(f"Frame capturado: capture_corrected_{timestamp}.jpg")
                elif key == ord('p'):
                    print("Video pausado. Presiona cualquier tecla para continuar...")
                    cv2.waitKey(0)
                elif key == ord('r'):
                    # Cambiar rotación manualmente
                    rotation_angle = (rotation_angle + 90) % 360
                    print(f"Rotación cambiada a: {rotation_angle}°")
                elif key == ord('f'):
                    # Nuevo: Alternar pantalla completa
                    cv2.setWindowProperty('Video Processing Optimized', cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN)
                elif key == ord('n'):
                    # Nuevo: Volver a ventana normal
                    cv2.setWindowProperty('Video Processing Optimized', cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)

                # Limpiar memoria periódicamente (menos frecuente para mejor rendimiento)
                if frame_count % 150 == 0:  # Cambiado de 30 a 150
                    clear_memory()

                # Mostrar estadísticas cada 100 frames
                if frame_count % 100 == 0:
                    valid_ratio = (detection_stats['valid_detections'] /
                                max(1, detection_stats['total_detections'])) * 100 if detection_stats['total_detections'] > 0 else 0
                    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                    print(f"Frame {frame_count}: Rotación {rotation_angle}°, FPS promedio: {avg_fps:.2f}, Detecciones válidas: {valid_ratio:.1f}%")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
    finally:
        # Mostrar estadísticas finales
        print("\\n=== Estadísticas Finales ===")
        print(f"Total detecciones: {detection_stats['total_detections']}")
        print(f"Detecciones válidas: {detection_stats['valid_detections']}")
        print(f"Rotación aplicada: {rotation_angle}°")
        if detection_stats['total_detections'] > 0:
            valid_percentage = (detection_stats['valid_detections'] /
                              detection_stats['total_detections']) * 100
            print(f"Porcentaje de validez: {valid_percentage:.2f}%")

        cap.release()
        if opts.save_video:
            out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Controles:")
    print("  'q' - Salir")
    print("  's' - Capturar frame")
    print("  'p' - Pausar/Reanudar")
    print("  'r' - Cambiar rotación manualmente")
    print("\\nSistema con corrección automática de rotación de video")
    main()