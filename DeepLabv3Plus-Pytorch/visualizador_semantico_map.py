from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
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
from collections import OrderedDict

from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

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

            # Obtener estado del modelo
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint

            # Limpiar nombres de estado
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v

            # Cargar estado del modelo
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
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='cityscapes',
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
    parser.add_argument("--save_results", action='store_true', default=False,
                        help="save visualization results")
    parser.add_argument("--output_dir", type=str, default='results',
                        help="directory to save results")

    return parser

def create_color_mask(pred):
    """Crea una máscara de color basada en la predicción"""
    height, width = pred.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Aplicar colores solo a las clases de interés
    for class_id in MAPILLARY_CLASSES.values():
        mask = pred == class_id
        color = COLOR_MAP[class_id]
        color_mask[mask] = color

    return color_mask
def blend_images(original, semantic, instance, alpha_semantic=0.5, alpha_instance=0.5):
    """
    Combina la imagen original con las máscaras semánticas y de instancias.
    """
    # Convertir todas las imágenes a numpy arrays y normalizar
    original = np.array(original, dtype=np.float32) / 255.0
    semantic = np.array(semantic, dtype=np.float32) / 255.0
    instance = np.array(instance, dtype=np.float32) / 255.0

    # Asegurar que todas las imágenes tengan las mismas dimensiones
    h, w = original.shape[:2]
    semantic = cv2.resize(semantic, (w, h))
    instance = cv2.resize(instance, (w, h))

    # Combinar las imágenes
    blend = original.copy()

    # Añadir máscara semántica donde hay segmentación
    mask_semantic = (semantic.sum(axis=2) > 0)
    blend[mask_semantic] = (1 - alpha_semantic) * original[mask_semantic] + alpha_semantic * semantic[mask_semantic]

    # Añadir máscara de instancias donde hay detecciones
    mask_instance = (instance.sum(axis=2) > 0)
    blend[mask_instance] = (1 - alpha_instance) * blend[mask_instance] + alpha_instance * instance[mask_instance]

    return (blend * 255).astype(np.uint8)
def draw_boxes(image, results):
    """
    Dibuja solo las bounding boxes sobre la imagen
    """
    img_with_boxes = image.copy()

    # Obtener las bounding boxes, clases y confidencias
    boxes = results[0].boxes
    for box in boxes:
        # Obtener coordenadas
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Obtener clase y confidencia
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        # Convertir coordenadas a enteros
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Dibujar rectángulo
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Añadir texto con clase y confidencia
        label = f'Class: {cls}, Conf: {conf:.2f}'
        cv2.putText(img_with_boxes, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_with_boxes
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
        yolo_model = YOLO(r'C:\Users\aaron\Documents\Año4\TFG\CityScapesWithPanopticSegm\runs\segment\train83\weights\best.pt')
        print("YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    if not image_files:
        raise ValueError("No valid image files found!")

    print(f"Found {len(image_files)} images to process")

    # Cargar modelo
    model, opts = load_model(opts, device)

    # Configurar transformaciones
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    ])

    # Crear directorio de resultados si es necesario
    if opts.save_results:
        os.makedirs(opts.output_dir, exist_ok=True)

    # Procesamiento de imágenes
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            try:
                ext = os.path.basename(img_path).split('.')[-1]
                img_name = os.path.basename(img_path)[:-len(ext)-1]

                # Imagen original
                img_original = Image.open(img_path).convert('RGB')
                img_array = np.array(img_original)

                # Procesamiento para DeepLab
                img = transform(img_original).unsqueeze(0)
                img = img.to(device)

                # Predicción DeepLab
                outputs = model(img)
                pred = outputs.max(1)[1].cpu().numpy()[0]

                # Crear máscara de color para las clases de interés
                color_mask = create_color_mask(pred)

                # Predicción YOLOv8
                results = yolo_model(img_array, conf=0.2, classes=[0, 1, 2, 3])
                #img_yolo = results[0].plot()

                #if len(img_yolo.shape) == 3 and img_yolo.shape[2] == 3:
                #    img_yolo = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)
                img_yolo=draw_boxes(img_array,results)
                
                # Mostrar resultados
                plt.figure(figsize=(18, 6))

                # Imagen original
                plt.subplot(1, 3, 1)
                plt.imshow(img_original)
                plt.title("Imagen original")
                plt.axis('off')
                # img_res=blend_images(img_array,color_mask,img_yolo,0.4,0.5)
                # plt.subplot(1, 3, 2)
                # plt.imshow(img_res)
                # plt.title("Segmentación Semántica (clases seleccionadas)")
                # plt.axis('off')
                
                # # Segmentación semántica
                plt.subplot(1, 3, 2)
                plt.imshow(color_mask)
                plt.title("Segmentación Semántica (clases seleccionadas)")
                plt.axis('off')

                # # Segmentación de instancias YOLOv8
                plt.subplot(1, 3, 3)
                plt.imshow(img_yolo)
                plt.title("Segmentación Instancias YOLOv8")
                plt.axis('off')

                plt.tight_layout()

                # Guardar o mostrar resultados
                if opts.save_results:
                    plt.savefig(os.path.join(opts.output_dir, f'{img_name}_combined.png'))
                else:
                    plt.show()

                plt.close('all')
                clear_memory()

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

if __name__ == '__main__':
    main()