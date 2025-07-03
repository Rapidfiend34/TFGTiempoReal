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
                        help="path to a single image or image directory")
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

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    # Cargar modelo YOLOv8
    yolo_model = YOLO(r'C:\Users\aaron\Documents\Año4\TFG\CityScapesWithPanopticSegm\runs\segment\train82\weights\best.pt')

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

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

    if opts.crop_val:
        transform = T.Compose([
            T.Resize(opts.crop_size),
            T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ])

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]

            # Imagen original
            img_original = Image.open(img_path).convert('RGB')

            # Convertir a numpy array para YOLOv8
            img_array = np.array(img_original)

            # Procesamiento para DeepLab
            img = transform(img_original).unsqueeze(0)
            img = img.to(device)

            # Predicción DeepLab
            pred = model(img).max(1)[1].cpu().numpy()[0]

            # Filtrar solo clases: road(0), sidewalk(1), pedestrian(11), rider(12)
            ids_interes = [0, 1, 11, 12]
            mask_filtrada = np.full_like(pred, 255)
            for clase in ids_interes:
                mask_filtrada[pred == clase] = clase

            colorized_preds = decode_fn(mask_filtrada).astype('uint8')

            # Predicción YOLOv8
            results = yolo_model(img_array, conf=0.15, classes=[ 0])  # Ajusta las clases según necesites

            # Obtener la imagen con las predicciones de YOLOv8
            img_yolo = results[0].plot()

            # Convertir de BGR a RGB si es necesario
            if len(img_yolo.shape) == 3 and img_yolo.shape[2] == 3:
                img_yolo = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)

            # Mostrar resultados
            plt.figure(figsize=(18, 6))

            # Imagen original
            plt.subplot(1, 3, 1)
            plt.imshow(img_original)
            plt.title("Imagen original")
            plt.axis('off')

            # Segmentación semántica
            # plt.subplot(1, 3, 2)
            # plt.imshow(colorized_preds)
            # plt.title("Segmentación Semántica filtrada")
            # plt.axis('off')

            # Segmentación de instancias YOLOv8
            plt.subplot(1, 3, 2)
            plt.imshow(img_yolo)
            plt.title("Segmentación Instancias YOLOv8")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            # Opcional: Guardar las imágenes
            # plt.savefig(f'results/{img_name}_combined.png')
            # plt.close()

if __name__ == '__main__':
    main()
