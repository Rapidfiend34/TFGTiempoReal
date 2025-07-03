
from ultralytics import YOLO
import torch
print("CUDA disponible:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
model = YOLO('yolov8n.pt')  # O el modelo que quieras exportar
model.export(format='engine', imgsz=320)  # 'engine' es el formato TensorRT