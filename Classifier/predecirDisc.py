from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont

# Cargar el modelo YOLOv8 entrenado
model = YOLO('./runs/classify/train5/weights/best.pt')  # Ruta al modelo entrenado

# Carpeta con las imágenes a clasificar
input_folder = './output/X/train/with_disability'  # Carpeta con las imágenes
output_folder = './PrediccionClasificadas'  # Carpeta para guardar las imágenes clasificadas
images_folder = os.path.join(output_folder, 'images')
annotations_folder = os.path.join(output_folder, 'labels')  # Carpeta para guardar los archivos de anotación

# Crear las carpetas de salida si no existen
os.makedirs(output_folder, exist_ok=True)
os.makedirs(annotations_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

# Crear las carpetas para las predicciones específicas
blind_folder = os.path.join(output_folder, 'blind')
wheelchair_folder = os.path.join(output_folder, 'wheelchair')

os.makedirs(blind_folder, exist_ok=True)
os.makedirs(wheelchair_folder, exist_ok=True)

# Fuente para las anotaciones visuales
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

# Clasificar cada imagen en la carpeta
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)

    # Asegurarse de que sea un archivo de imagen
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Realizar la predicción
    results = model.predict(image_path, imgsz=224, device='cpu')

    # Obtener la clase predicha y su probabilidad
    pred_class_index = results[0].probs.top1
    pred_class = results[0].names[pred_class_index]
    pred_prob = results[0].probs.top1conf.item()

    # Guardar la anotación en un archivo TXT individual
    # Usar el mismo nombre de archivo que la imagen pero con extensión .txt
    txt_filename = os.path.splitext(image_name)[0] + '.txt'
    txt_path = os.path.join(annotations_folder, txt_filename)
    
    with open(txt_path, 'w') as txtfile:
        txtfile.write(f"Class: {pred_class}\n")
        txtfile.write(f"Confidence: {pred_prob:.4f}\n")
        # Puedes añadir más información si lo deseas
        txtfile.write(f"Image: {image_name}\n")
        txtfile.write(f"Date: {os.path.getmtime(image_path)}\n")

    # Crear la ruta de salida dependiendo de la predicción
    if pred_class == 'blind':
        class_folder = blind_folder
    elif pred_class == 'wheelchair':
        class_folder = wheelchair_folder
    else:
        # Si la clase no es 'blind' ni 'wheelchair', se guarda en una carpeta 'others'
        class_folder = os.path.join(output_folder, 'others')
        os.makedirs(class_folder, exist_ok=True)

    # Guardar la imagen en la carpeta correspondiente
    output_image_path = os.path.join(class_folder, image_name)
    image = Image.open(image_path).convert("RGB")

    # Añadir la anotación en la imagen
    text = f"{pred_class} ({pred_prob:.2f})"

    # Método compatible con versiones recientes de Pillow
    try:
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        try:
            text_width, text_height = font.getsize(text)
        except AttributeError:
            text_width, text_height = 100, 20

    text_position = (10, 10)
    """ draw.rectangle(
        [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
        fill="black"
    ) """
    #draw = ImageDraw.Draw(image)
    #draw.text(text_position, text, fill="white", font=font)

    # Guardar la imagen con la anotación visual
    image.save(output_image_path)

    print(f"Procesada: {image_name} -> Clase: {pred_class}, Probabilidad: {pred_prob:.2f}")
    print(f"Anotación guardada en: {txt_path}")
    print(f"Imagen guardada en: {output_image_path}")

print(f"Imágenes clasificadas guardadas en: {output_folder}")
print(f"Archivos de anotación guardados en: {annotations_folder}")
