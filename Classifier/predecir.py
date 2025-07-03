from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont

# Cargar el modelo YOLOv8 entrenado
model = YOLO('./runs/classify/train4/weights/best.pt')  # Ruta al modelo entrenado

# Carpeta con las imágenes a clasificar
input_folder = 'c:\Users\aaron\Downloads\4cd16901-5241-4c97-b6c4-4caeb98c2bdf.png'  # Carpeta con las imágenes
output_folder = './PrediccionClasificadas'  # Carpeta para guardar las imágenes clasificadas

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Fuente para las anotaciones (opcional, requiere PIL)
try:
    font = ImageFont.truetype("arial.ttf", 20)  # Cambia el tamaño según sea necesario
except IOError:
    font = ImageFont.load_default()

# Clasificar cada imagen en la carpeta
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)

    # Asegurarse de que sea un archivo de imagen
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Realizar la predicción
    results = model.predict(image_path, imgsz=224, device='cpu')  # Cambia a 'cuda' si usas GPU

    # Obtener la clase predicha y su probabilidad
    pred_class_index = results[0].probs.top1  # Índice de la clase con mayor probabilidad
    pred_class = results[0].names[pred_class_index]  # Nombre de la clase
    pred_prob = results[0].probs.top1conf.item()  # Probabilidad de la clase

    # Cargar la imagen original
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Añadir la anotación en la imagen
    text = f"{pred_class} ({pred_prob:.2f})"
    
    # Método compatible con versiones recientes de Pillow
    try:
        # Para Pillow >= 9.0.0
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        try:
            # Para Pillow >= 8.0.0
            text_width, text_height = font.getsize(text)
        except AttributeError:
            # Para versiones anteriores o alternativa
            text_width, text_height = 100, 20  # Valores predeterminados
    
    text_position = (10, 10)  # Posición del texto en la imagen
    draw.rectangle(
        [text_position, (text_position[0] + text_width, text_position[1] + text_height)], 
        fill="black"
    )
    draw.text(text_position, text, fill="white", font=font)

    # Guardar la imagen con la anotación
    output_path = os.path.join(output_folder, image_name)
    image.save(output_path)

    print(f"Procesada: {image_name} -> Clase: {pred_class}, Probabilidad: {pred_prob:.2f}")

print(f"Imágenes clasificadas guardadas en: {output_folder}")