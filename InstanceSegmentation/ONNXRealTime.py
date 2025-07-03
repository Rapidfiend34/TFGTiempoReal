import time
import cv2
import torch
import onnxruntime as ort
import numpy as np

# Configurar la sesión de ONNX Runtime para usar la GPU si está disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
onnx_model_path = 'runs/segment/train52/weights/best.onnx'

# Iniciar la sesión de ONNX Runtime en la GPU
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])

# Cargar la imagen una sola vez en memoria
img_path = "dataset/images/test/berlin_000001_000019_gtFine_polygons.png"

# Cargar y redimensionar la imagen solo una vez para asegurar que está lista para la inferencia
img = cv2.imread(img_path)
img = cv2.resize(img, (1024, 1024))  # Redimensionar si es necesario

# Preprocesar la imagen (normalización y reordenación)
img = np.transpose(img, (2, 0, 1))  # Convertir de HWC a CHW
img = img.astype(np.float32)  # Asegurarse de que sea de tipo float32
img /= 255.0  # Normalizar entre 0 y 1
img = np.expand_dims(img, axis=0)  # Añadir dimensión de batch (1, C, H, W)

# Número de frames para la prueba
num_frames = 100
total_time = 0

# Inferencia sobre varios frames
for _ in range(num_frames):
    start_time = time.time()
    
    # Realizar la predicción con ONNX Runtime
    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)
    
    total_time += time.time() - start_time

# FPS promedio
fps = num_frames / total_time
print(f"FPS promedio: {fps:.2f}")
