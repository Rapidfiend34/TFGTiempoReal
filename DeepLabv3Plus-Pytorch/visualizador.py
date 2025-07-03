import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import random
import argparse
import numpy as np
import cv2
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from datasets import VOCSegmentation, Cityscapes, mapillary
from datasets.mapillary import MapillaryVistas, MapillaryTransform
from torchvision import transforms as T
from metrics import StreamSegMetrics
from ultralytics import YOLO
import torch
import torch.nn as nn
from collections import OrderedDict, deque
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import time
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
from FastTracker import FastTracker

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

def load_model(opts, device):
    # Crear modelo con 114 clases (número correcto para Mapillary)
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
            print("Checkpoint cargado exitosamente")

        except Exception as e:
            print(f"Error al cargar checkpoint: {e}")
            raise

    model = model.to(device)
    return model, opts

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to input video file")
    parser.add_argument("--dataset", type=str, default='mapillary',
                        choices=['voc', 'cityscapes', 'mapillary'], help='Name of training set')
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--mapillary_version", type=str, default='v2.0',
                        choices=['v1.2', 'v2.0'], help='Mapillary version')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
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

    for class_id in MAPILLARY_CLASSES.values():
        mask = pred == class_id
        color = COLOR_MAP[class_id]
        color_mask[mask] = color

    return color_mask

def process_frame(frame, model, device, transform, yolo_model, tracker):
    """Procesa un frame del video"""
    # Convertir frame de BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    # Procesamiento para DeepLab
    img = transform(img_pil).unsqueeze(0)
    img = img.to(device)

    # Predicción DeepLab
    outputs = model(img)
    pred = outputs.max(1)[1].cpu().numpy()[0]

    # Crear máscara de color
    color_mask = create_color_mask(pred)

    # Predicción YOLOv8 con umbral de confianza más bajo para personas
    results = yolo_model(frame_rgb, conf=0.25, classes=[0])  # Solo personas con conf > 0.25

    # Crear una copia del frame para las detecciones y tracking
    combined_visualization = frame_rgb.copy()

    # Dibujar las detecciones de YOLO
    combined_visualization = draw_boxes(combined_visualization, results)

    # Actualizar tracker y dibujar tracks
    tracks = tracker.update(results, frame_rgb)

    # Debug: imprimir información sobre tracks
    if len(tracks) > 0:
        print(f"Tracks activos: {len(tracks)}")

    combined_visualization = tracker.draw_tracks(combined_visualization, tracks)

    return frame_rgb, color_mask, combined_visualization

def main():
    opts = get_argparser().parse_args()

    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    # Configurar dispositivo
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Cargar modelo YOLOv8
    try:
        yolo_model = YOLO('yolov8n-seg.pt')
        print("YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return

    # Inicializar tracker con parámetros optimizados
    tracker = FastTracker(
        det_thresh=0.3,      # Umbral de detección
        max_age=15,          # Reducido para evitar tracks fantasma
        min_hits=2,          # Reducido para iniciar tracks más rápidamente
        iou_threshold=0.25,  # Reducido para ser más permisivo en la asociación
        delta_t=3,           # Mantener en 3 frames para predicción
        track_history=30     # Mantener historial de 30 frames
    )

    # Cargar modelo DeepLab
    model, opts = load_model(opts, device)

    # Configurar transformaciones
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    ])

    # Abrir video
    cap = cv2.VideoCapture(opts.input)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video")
        return

    # Configurar writer para guardar video si es necesario
    if opts.save_video:
        os.makedirs(opts.output_dir, exist_ok=True)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_path = os.path.join(opts.output_dir, 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width*3, frame_height))

    plt.ion()  # Modo interactivo de matplotlib
    fig = plt.figure(figsize=(18, 6))

    with torch.no_grad():
        model = model.eval()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar frame
            frame_rgb, color_mask, combined_viz = process_frame(
                frame, model, device, transform, yolo_model, tracker
            )

            # Mostrar resultados
            plt.clf()

            plt.subplot(1, 3, 1)
            plt.imshow(frame_rgb)
            plt.title("Frame Original")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(color_mask)
            plt.title("Segmentación Semántica")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(combined_viz)
            plt.title("Detección y Tracking")
            plt.axis('off')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

            # Guardar frame si es necesario
            if opts.save_video:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
                combined_bgr = cv2.cvtColor(combined_viz, cv2.COLOR_RGB2BGR)

                combined_frame = np.hstack((frame_bgr, mask_bgr, combined_bgr))
                out.write(combined_frame)

            # Limpiar memoria
            clear_memory()

            # Verificar si se presiona 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Limpieza
    cap.release()
    if opts.save_video:
        out.release()
    plt.ioff()
    plt.close('all')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()