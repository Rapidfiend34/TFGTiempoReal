import time
import cv2
import torch
from ultralytics import YOLO

# Cargar modelo en GPU si está disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('runs/segment/train52/weights/best.pt').to(device)

# Cargar imagen una sola vez en memoria
img_path = "C:/Users/aaron/Documents/Año4/TFG/CityScapesWithPanopticSegm/dataset/images/test/berlin_000001_000019_gtFine_polygons.png"
img = cv2.imread(img_path)  # OpenCV carga imágenes más rápido
img = cv2.resize(img, (640, 512))  # Reducción de tamaño para mejorar FPS

# Número de frames para la prueba
num_frames = 100  
total_time = 0

for _ in range(num_frames):
    start_time = time.time()
    results = model.predict(img, device=device, half=True, verbose=False)  # verbose=False desactiva logs
    total_time += time.time() - start_time

# FPS promedio
fps = num_frames / total_time
print(f"FPS promedio: {fps:.2f}")
