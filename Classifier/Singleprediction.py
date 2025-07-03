from ultralytics import YOLO
from PIL import Image

def clasificar_imagen(ruta_modelo, ruta_imagen):
    # Cargar el modelo YOLOv8
    model = YOLO(ruta_modelo)

    # Realizar la predicción
    results = model.predict(ruta_imagen, imgsz=224, device='cpu')  # Cambia a 'cuda' si tienes GPU

    # Obtener clase y probabilidad
    pred_class_index = results[0].probs.top1
    pred_class = results[0].names[pred_class_index]
    pred_prob = results[0].probs.top1conf.item()

    return pred_class, pred_prob

# Ejemplo de uso
ruta_modelo = r'C:\\Users\\aaron\\Documents\\Año4\\TFG\\Classifier\\runs\\classify\\train4\\weights\\best.pt'
ruta_imagen = r'c:\Users\aaron\Downloads\Captura de pantalla 2025-06-19 153511.png'

clase, prob = clasificar_imagen(ruta_modelo, ruta_imagen)
print(f"Clase predicha: {clase}, Probabilidad: {prob:.2f}")
