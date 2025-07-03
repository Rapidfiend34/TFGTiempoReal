import os
import cv2


# Directorios
dataset_dir = "./NinosDataset/train"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
output_dir = "./output/Ninos/train"

# Crear directorio de salida para cada clase
# classes = ["non_vulnerable", "wheelchair", "blind", "elder", "child"]
classes = ["child", "elder", "non_vulnerable", "with_disability", "x"]
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# Procesar cada imagen y sus anotaciones
for img_file in os.listdir(images_dir):
    if not img_file.endswith(('.jpg', '.jpeg', '.png')):
        continue

    # Cargar imagen
    img_path = os.path.join(images_dir, img_file)
    img = cv2.imread(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f" Imagen no cargada: {img_path}")
        continue
    h, w, _ = img.shape

    # Cargar anotaciones (formato YOLO)
    label_file = os.path.splitext(img_file)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_file)

    if not os.path.exists(label_path):
        continue

    # Leer anotaciones
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Procesar cada bounding box
    for i, line in enumerate(lines):
        parts = line.strip().split()
        class_id = int(parts[0])  # Asumiendo que el primer valor es el ID decl

        # Convertir coordenadas YOLO (normalizadas) a píxeles
        x_center, y_center, width, height = map(float, parts[1:5])
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)

        # Asegurar que las coordenadas estén dentro de los límites
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            print(f" BBox inválida: {parts[1:5]}")
            continue
        # Recortar la imagen
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            print(f" Recorte vacío para {output_dir}")
            continue

        # Determinar la clase (esto  tu mapeo de class_id a categora)
        # Ejemplo simplificado:
        category = classes[class_id] if class_id < len(classes) else "unknown"

        # Guardar la imagen recortada
        output_path = os.path.join(output_dir, category,
                                   f"{os.path.splitext(img_file)[0]}_{i}.jpg")
        cv2.imwrite(output_path, cropped)

        print(f"Saved: {output_path}")