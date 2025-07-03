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
from torchvision import transforms as T
from metrics import StreamSegMetrics
from ultralytics import YOLO
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to input video file")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of training set')

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

    return parser

def process_frame(frame, model, device, transform, decode_fn, yolo_model):
    # Convertir frame de BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    # Procesamiento para DeepLab
    img = transform(img_pil).unsqueeze(0)
    img = img.to(device)

    # Predicción DeepLab
    pred = model(img).max(1)[1].cpu().numpy()[0]

    # Filtrar clases de interés
    ids_interes = [0, 1, 11, 12]
    mask_filtrada = np.full_like(pred, 255)
    for clase in ids_interes:
        mask_filtrada[pred == clase] = clase

    colorized_preds = decode_fn(mask_filtrada).astype('uint8')

    # Predicción YOLOv8
    results = yolo_model(frame_rgb, conf=0.2, classes=[0, 1, 2, 3])
    img_yolo = results[0].plot()
    img_yolo = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)

    return frame_rgb, colorized_preds, img_yolo

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    # Cargar modelo YOLOv8
    yolo_model = YOLO(r'C:\Users\aaron\Documents\Año4\TFG\CityScapesWithPanopticSegm\runs\segment\train83\weights\best.pt')

    # Configuración de dispositivo
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Configurar modelo DeepLab
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Cargar checkpoint si existe
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

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

    frame_count = 0
    plt.ion()  # Modo interactivo de matplotlib
    fig = plt.figure(figsize=(18, 6))

    frame_delay = 0.00001  # Segundos entre frames (ajustable)

    with torch.no_grad():
        model = model.eval()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fin del video")
                break

            frame_count += 1
            print(f"Procesando frame {frame_count}")

            # Procesar frame
            frame_rgb, colorized_preds, img_yolo = process_frame(
                frame, model, device, transform, decode_fn, yolo_model
            )

            # Limpiar figura anterior
            plt.clf()

            # Mostrar resultados
            plt.subplot(1, 3, 1)
            plt.imshow(frame_rgb)
            plt.title("Frame Original")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(colorized_preds)
            plt.title("Segmentación Semántica")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(img_yolo)
            plt.title("Segmentación Instancias YOLOv8")
            plt.axis('off')

            plt.tight_layout()
            plt.draw()

            # Pausa breve para mostrar el frame
            plt.pause(frame_delay)

            # Opcional: Verificar si se presiona 'q' para salir
            if plt.waitforbuttonpress(timeout=0.001):
                key = plt.ginput(timeout=0.001)
                if key == 'q':
                    break

    cap.release()
    plt.ioff()
    plt.close('all')

if __name__ == '__main__':
    main()