import json
import os
import cv2
import math
import numpy as np
from PIL import Image 
#from shapely.geometry import Polygon, box
import gc
import shutil
# Rutas
json_path = './mapillary/archivo_de_instancias.json'
output_dir = './dataset/labels'
def convert_polygons_to_yolo_bbox(json_data, image_width, image_height, label_mapping):
    """
    Convierte las anotaciones de polígonos en bounding boxes en formato YOLO.

    Args:
        json_data (dict): Datos del JSON con anotaciones.
        image_width (int): Ancho de la imagen en píxeles.
        image_height (int): Alto de la imagen en píxeles.
        label_mapping (dict): Diccionario que asigna 'label' a un ID numérico YOLO.

    Returns:
        List[str]: Anotaciones en formato YOLO (una línea por objeto).
    """
    yolo_annotations = []

    for obj in json_data['objects']:
        label = obj['label']
        polygon = obj['polygon']

        class_id = label_mapping.get(label, -1)
        if class_id == -1:
            print(f"Advertencia: Etiqueta '{label}' no encontrada en label_mapping.")
            continue

        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]

        x_min = max(min(xs), 0)
        x_max = min(max(xs), image_width)
        y_min = max(min(ys), 0)
        y_max = min(max(ys), image_height)

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Normalizar
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_annotations.append(yolo_line)

    return yolo_annotations

def load_yolo_bbox_labels(label_path, image_width, image_height):
    """
    Carga etiquetas en formato YOLO (bounding boxes).

    Args:
        label_path (str): Ruta al archivo de etiquetas.
        image_width (int): Ancho de la imagen.
        image_height (int): Alto de la imagen.

    Returns:
        List[Tuple[int, float, float, float, float]]: Lista de tuplas (class_id, center_x, center_y, width, height).
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            category_id = int(parts[0])
            center_x = float(parts[1]) * image_width
            center_y = float(parts[2]) * image_height
            bbox_width = float(parts[3]) * image_width
            bbox_height = float(parts[4]) * image_height

            labels.append((category_id, center_x, center_y, bbox_width, bbox_height))

    return labels
def draw_yolo_bboxes(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: no se pudo cargar la imagen en la ruta especificada.")
        exit(1)

    height, width, _ = img.shape  # dimensiones de la imagen (alto, ancho)

    # 2. Leer el archivo de anotaciones YOLO
    with open(label_path, 'r') as f:
        lines = f.read().strip().splitlines()  # Lee todas las líneas del archivo

    # 3. Para cada anotación, dibujar la caja delimitadora
    for line in lines:
        # Formato esperado: class_id center_x center_y width height (todos en [0,1])
        parts = line.split()
        if len(parts) != 5:
            continue  # ignorar líneas mal formateadas, si las hubiera

        class_id, x_center_norm, y_center_norm, w_norm, h_norm = parts
        x_center_norm = float(x_center_norm)
        y_center_norm = float(y_center_norm)
        w_norm = float(w_norm)
        h_norm = float(h_norm)

        # 4. Convertir coordenadas normalizadas a coordenadas absolutas en píxeles
        x_center = x_center_norm * width    # coordenada x del centro en píxeles
        y_center = y_center_norm * height   # coordenada y del centro en píxeles
        w_pixels = w_norm * width           # ancho de la caja en píxeles
        h_pixels = h_norm * height          # alto de la caja en píxeles

        # Calcular esquinas de la caja
        x1 = int(x_center - w_pixels / 2)   # esquina izquierda
        y1 = int(y_center - h_pixels / 2)   # esquina superior
        x2 = int(x_center + w_pixels / 2)   # esquina derecha
        y2 = int(y_center + h_pixels / 2)   # esquina inferior

        # Ajustar coordenadas para que estén dentro de los límites de la imagen
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        
        # 5. Dibujar el rectángulo de la bounding box sobre la imagen
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))    # color verde en formato BGR
        thickness = 2          # grosor de línea de 2 píxeles
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # (Opcional) Escribir el class_id junto a la caja
        label = str(class_id)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)

    # 6. Redimensionar la imagen a 980x980 para visualización (sin alterar la original)
    display_img = cv2.resize(img, (980, 980))

    # 7. Mostrar la imagen en una ventana
    # cv2.imshow("Imagen con Bounding Boxes", display_img)
    # cv2.waitKey(0)            # Esperar hasta que el usuario presione una tecla
    # cv2.destroyAllWindows()   # Cerrar la ventana al terminar
def save_yolo_bbox_labels(labels, output_path, tile_size):
    """
    Guarda etiquetas de bounding boxes en formato YOLO.

    Args:
        labels (list): Lista de (class_id, center_x, center_y, width, height) en píxeles.
        output_path (str): Ruta donde guardar el archivo de etiquetas.
        tile_size (int): Tamaño del tile en píxeles (para normalizar).
    """
    with open(output_path, 'w') as f:
        for label in labels:
            category_id, center_x, center_y, bbox_width, bbox_height = label
            norm_center_x = center_x / tile_size
            norm_center_y = center_y / tile_size
            norm_width = bbox_width / tile_size
            norm_height = bbox_height / tile_size
            f.write(f"{category_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

def split_image_and_bbox_labels(image_path, label_path, output_image_folder, output_label_folder, tile_size=640):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return

    height, width, _ = image.shape
    labels = load_yolo_bbox_labels(label_path, width, height)
    num_tiles_x = math.ceil(width / tile_size)
    num_tiles_y = math.ceil(height / tile_size)

    basename = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x_start = j * tile_size
            y_start = i * tile_size
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)

            tile = image[y_start:y_end, x_start:x_end]

            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                tile = cv2.copyMakeBorder(
                    tile,
                    0, tile_size - tile.shape[0],
                    0, tile_size - tile.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

            tile_labels = []

            for label in labels:
                category_id, center_x, center_y, bbox_width, bbox_height = label
                if (x_start <= center_x-bbox_width) and (y_start <= center_y+bbox_height):
                    adj_center_x = clamp_to_edges(center_x,x_end) - x_start
                    adj_center_y = clamp_to_edges(center_y,y_end) - y_start
                    adj_width= bbox_width/2
                    tile_labels.append((category_id, adj_center_x, adj_center_y, bbox_width, bbox_height))

            tile_image_name = f"{basename}_{i}_{j}.jpg"
            tile_label_name = f"{basename}_{i}_{j}.txt"

            cv2.imwrite(os.path.join(output_image_folder, tile_image_name), tile)
            save_yolo_bbox_labels(tile_labels, os.path.join(output_label_folder, tile_label_name), tile_size)

    print(f"Fragmentación completada para {image_path}.")
def convert_polygons_to_yolo_segmentation(json_data, image_width, image_height, label_mapping):
    """
    Convierte las anotaciones de polígonos en formato YOLO Segmentación.

    Args:
        json_data (dict): Datos del JSON con anotaciones.
        image_width (int): Ancho de la imagen en píxeles.
        image_height (int): Alto de la imagen en píxeles.
        label_mapping (dict): Diccionario que asigna 'label' a un ID numérico YOLO.

    Returns:
        List[str]: Anotaciones en formato YOLO Segmentación (una línea por objeto).
    """
    yolo_annotations = []

    for obj in json_data['objects']:
        label = obj['label']
        polygon = obj['polygon']


        class_id = label_mapping.get(label, -1)
        if class_id == -1:
            print(f"Advertencia: Etiqueta '{label}' no encontrada en label_mapping.")
            continue


        normalized_points = []
        for point in polygon:
            x_norm = min(max(point[0] / image_width, 0), 1)
            y_norm = min(max(point[1] / image_height, 0), 1)
            normalized_points.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])


        yolo_line = f"{class_id} " + " ".join(normalized_points)
        yolo_annotations.append(yolo_line)

    return yolo_annotations



def split_image_into_tiles(image_path, output_folder, tile_size=640):
    os.makedirs(output_folder, exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return

    height, width, _ = image.shape


    num_tiles_x = math.ceil(width / tile_size)
    num_tiles_y = math.ceil(height / tile_size)

    basename = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x_start = j * tile_size
            y_start = i * tile_size
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)


            tile = image[y_start:y_end, x_start:x_end]


            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                padded_tile = cv2.copyMakeBorder(
                    tile,
                    0, tile_size - tile.shape[0],
                    0, tile_size - tile.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]  # Color negro
                )
                tile = padded_tile

            tile_filename = os.path.join(output_folder, f"{basename}_{i}_{j}.jpg")
            cv2.imwrite(tile_filename, tile)

    print(f"Imagen {image_path} fragmentada en {output_folder}")

def load_yolo_segmentation_labels(label_path, image_width, image_height):
    """
    Carga anotaciones en formato YOLO Segmentación desde un archivo de texto.

    Args:
        label_path (str): Ruta al archivo de etiquetas.
        image_width (int): Ancho de la imagen.
        image_height (int): Alto de la imagen.

    Returns:
        List[Tuple[int, List[Tuple[float, float]]]]: Lista de tuplas (class_id, lista de puntos (x, y) en píxeles).
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3 or (len(parts) - 1) % 2 != 0:
                # Debe haber un class_id seguido de pares (x, y)
                continue

            category_id = int(parts[0])
            points = []

            coords = parts[1:]
            for i in range(0, len(coords), 2):
                x = float(coords[i]) * image_width
                y = float(coords[i+1]) * image_height
                points.append((x, y))

            labels.append((category_id, points))

    return labels

def save_yolo_segmentation_labels(labels, output_path, tile_size):
    """
    Guarda etiquetas de segmentación en formato YOLO.

    Args:
        labels (list): Lista de (class_id, lista de puntos [(x, y), ...]).
        output_path (str): Ruta del archivo donde guardar las etiquetas.
        tile_size (int): Tamaño del tile en píxeles (para normalizar las coordenadas).
    """
    with open(output_path, 'w') as f:
        for label in labels:
            category_id, points = label
            line = [str(category_id)]
            for x, y in points:
                norm_x = x / tile_size
                norm_y = y / tile_size
                line.append(f"{norm_x:.6f}")
                line.append(f"{norm_y:.6f}")
            f.write(' '.join(line) + '\n')

def clamp_to_edges(x, x_end):
    return min(x, x_end)

import os
import cv2
import numpy as np
import math
#from shapely.geometry import Polygon, box

def split_image_and_polygon_labels_only_vis(image_path, label_path,tile_size=640):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return

    height, width, _ = image.shape

    # Cargar anotaciones
    labels = load_yolo_segmentation_labels(label_path, width, height)

    # --- Visualización inicial de la imagen completa anotada ---
    image_full = image.copy()

    for category_id, points in labels:
        points_np = np.array(points, dtype=np.int32)
        if points_np.shape[0] >= 3:
            cv2.polylines(image_full, [points_np], isClosed=True, color=(0, 255, 0), thickness=2)
    image_full = cv2.resize(image_full, (980, 980))
    """  cv2.imshow("Imagen completa anotada", image_full)
    print("Mostrando imagen completa anotada. Pulse cualquier tecla para continuar...")
    key = cv2.waitKey(0)
    if key == 27:  # Si pulsa ESC, salir
        cv2.destroyAllWindows()
        return
    cv2.destroyAllWindows() """

    # --- Fragmentación en tiles ---
    num_tiles_x = math.ceil(width / tile_size)
    num_tiles_y = math.ceil(height / tile_size)

    pad_x = num_tiles_x * tile_size - width
    pad_y = num_tiles_y * tile_size - height

    image_padded = cv2.copyMakeBorder(
        image,
        0, pad_y,
        0, pad_x,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    tiles = image_padded.reshape(
        num_tiles_y, tile_size,
        num_tiles_x, tile_size,
        3
    ).swapaxes(1, 2).reshape(-1, tile_size, tile_size, 3)

    x_starts = np.arange(0, num_tiles_x * tile_size, tile_size)
    y_starts = np.arange(0, num_tiles_y * tile_size, tile_size)
    tile_grid = np.array([(y, x) for y in y_starts for x in x_starts])

    for idx, tile in enumerate(tiles):
        y_start, x_start = tile_grid[idx]
        tile_box = box(x_start, y_start, x_start + tile_size, y_start + tile_size)

        tile_labels = []

        for category_id, points in labels:
            poly = Polygon(points)

            if not poly.is_valid:
                poly = poly.buffer(0)

            intersection = poly.intersection(tile_box)

            if not intersection.is_empty and isinstance(intersection, Polygon):
                intersection_coords = np.array(intersection.exterior.coords)

                # Ajustar a coordenadas locales del tile
                adjusted_coords = intersection_coords - np.array([x_start, y_start])

                if len(adjusted_coords) >= 3:
                    tile_labels.append((category_id, adjusted_coords))

        # Visualizar tile con anotaciones
        debug_tile = tile.copy()

        for category_id, points in tile_labels:
            if len(points) >= 3:
                points_int = points.astype(np.int32)
                cv2.polylines(debug_tile, [points_int], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow(f"Tile {idx}", debug_tile)
        key = cv2.waitKey(0)
        if key == 27:  # ESC para salir
            cv2.destroyAllWindows()
            return
        cv2.destroyAllWindows()

    print(f"Visualización completada para {image_path}.")



def split_image_and_polygon_labels(image_path, label_path, output_image_folder, output_label_folder, tile_size=640):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return

    height, width, _ = image.shape
    labels = load_yolo_segmentation_labels(label_path, width, height)
    num_tiles_x = math.ceil(width / tile_size)
    num_tiles_y = math.ceil(height / tile_size)

    basename = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x_start = j * tile_size
            y_start = i * tile_size
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)

            tile = image[y_start:y_end, x_start:x_end]

            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                tile = cv2.copyMakeBorder(
                    tile,
                    0, tile_size - tile.shape[0],
                    0, tile_size - tile.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

            tile_labels = []

            for label in labels:
                category_id, points = label

                # Ajustar puntos al tile
                adjusted_points = []
                for x, y in points:
                   
                    if x_start <= x and y_start <= y:
                        print(f"x: {x}, y: {y} x_start: {x_start}, y_start: {y_start}, x_end: {x_end}, y_end: {y_end}")
                        #exit()
                        adj_x = min(x,x_end) - x_start
                        adj_y = min(y,y_end) - y_start
                        adjusted_points.append((adj_x, adj_y))


                if len(adjusted_points) >= 3:
                    tile_labels.append((category_id, adjusted_points))

            # Guardar imagen
            tile_image_name = f"{basename}_{i}_{j}.jpg"
            tile_label_name = f"{basename}_{i}_{j}.txt"

            cv2.imwrite(os.path.join(output_image_folder, tile_image_name), tile)

            # Guardar etiquetas
            save_yolo_segmentation_labels(tile_labels, os.path.join(output_label_folder, tile_label_name), tile_size)

    print(f"Fragmentación completada para {image_path}.")

def sample_class_slices_complete(image_path, instance_label_array, pending_classes, tile_size=128):
    """
    Extrae recortes SOLO para las clases aún pendientes y marca visualmente la región de la clase.

    Args:
        image_path (str): Ruta de la imagen original.
        instance_label_array (np.ndarray): Array de etiquetas (IDs de clases).
        pending_classes (set): Clases de las que aún no tenemos muestra.
        tile_size (int): Tamaño del recorte.

    Returns:
        dict: {class_id: tile_con_superposición}
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    height, width = instance_label_array.shape
    sampled_tiles = {}

    unique_labels = np.unique(instance_label_array)

    for class_id in unique_labels:
        if class_id == 0:
            continue  # ignorar fondo

        if class_id not in pending_classes:
            continue  # ya tenemos esta clase

        # Máscara binaria de la clase
        mask = (instance_label_array == class_id).astype(np.uint8) * 255

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        idx = np.random.choice(len(xs))
        x_center, y_center = xs[idx], ys[idx]

        x_start = max(x_center - tile_size // 2, 0)
        y_start = max(y_center - tile_size // 2, 0)
        x_end = min(x_start + tile_size, width)
        y_end = min(y_start + tile_size, height)

        # Recorte de la imagen
        tile = image[y_start:y_end, x_start:x_end].copy()
        
        # Recorte correspondiente de la máscara
        tile_mask = mask[y_start:y_end, x_start:x_end]

        # Crear overlay rojo (BGR = (0, 0, 255))
        red_overlay = np.zeros_like(tile, dtype=np.uint8)
        red_overlay[:, :] = (0, 0, 255)  # rojo

        # Aplicar superposición solo donde la clase esté presente (máscara > 0)
        tile[tile_mask > 0] = cv2.addWeighted(tile[tile_mask > 0], 0.5, red_overlay[tile_mask > 0], 0.2, 0)

        sampled_tiles[class_id] = tile
    gc.collect()

    return sampled_tiles



def process_all_images_until_full_coverage(images_folder, instances_folder, id_to_classname, tile_size=128):
    """
    Recorre todas las imágenes hasta obtener una muestra por cada clase.

    Args:
        images_folder (str): Carpeta de imágenes (.jpg).
        instances_folder (str): Carpeta de instancias (.png).
        id_to_classname (dict): Diccionario {id_real: nombre_clase}.
        tile_size (int): Tamaño de los recortes.

    Returns:
        dict: {class_id: tile}
    """
    all_tiles = {}
    pending_classes = set(id_to_classname.values())  # Todos los IDs de clase esperados
    con=2
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    for img_file in image_files:
        image_path = os.path.join(images_folder, img_file)
        instance_name = os.path.splitext(img_file)[0] + '.png'
        instance_path = os.path.join(instances_folder, instance_name)
        print(instance_path)
        if not os.path.exists(instance_path):
            print(f"Instancia no encontrada para {img_file}, saltando.")
            continue

        instance_image = Image.open(instance_path)
        instance_array = np.array(instance_image, dtype=np.uint16)
        instance_label_array = (instance_array / 256).astype(np.uint8)
        tiles = sample_class_slices_complete(image_path, instance_label_array, pending_classes, tile_size)

        for class_id, tile in tiles.items():
            if class_id not in all_tiles:
                all_tiles[class_id] = tile
                pending_classes.discard(class_id)

        if len(pending_classes) == 0:
            print("Se ha conseguido una muestra para cada clase.")
            break
        #if con==0:
            #return all_tiles
        con-=1
    if pending_classes:
        print(f"No se encontraron muestras para las siguientes clases: {pending_classes}")

    return all_tiles
from concurrent.futures import ProcessPoolExecutor, as_completed

def _process_single_image(image_path, instance_path, pending_classes, tile_size):
    """
    Procesa una imagen individual para extraer tiles de clases pendientes.

    Args:
        image_path (str): Ruta a la imagen.
        instance_path (str): Ruta al archivo de instancia.
        pending_classes (set): Conjunto de clases aún no cubiertas.
        tile_size (int): Tamaño del tile.

    Returns:
        dict: {class_id: tile}
    """
    try:
        instance_image = Image.open(instance_path)
        instance_array = np.array(instance_image, dtype=np.uint16)
        instance_label_array = (instance_array >> 8).astype(np.uint8)

        tiles = sample_class_slices_complete(image_path, instance_label_array, pending_classes, tile_size)
        
        return tiles
    except Exception as e:
        print(f"[ERROR] Procesando {image_path}: {e}")
        return {}
    

# def process_all_images_until_full_coverage_concurrent(images_folder, instances_folder, id_to_classname, tile_size=128, max_workers=4):
#     """
#     Versión concurrente: recorre imágenes hasta obtener una muestra por clase.

#     Args:
#         images_folder (str): Carpeta de imágenes (.jpg).
#         instances_folder (str): Carpeta de instancias (.png).
#         id_to_classname (dict): Diccionario {id_real: nombre_clase}.
#         tile_size (int): Tamaño de los recortes.
#         max_workers (int): Número máximo de procesos concurrentes.

#     Returns:
#         dict: {class_id: tile}
#     """
#     inverted_mapping = {v: k for k, v in id_to_classname.items()}
#     all_tiles = {}
#     pending_classes = set(id_to_classname.values())
#     image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
#     # image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')][:20]
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         con=2
#         for img_file in image_files:
#             image_path = os.path.join(images_folder, img_file)
#             instance_name = os.path.splitext(img_file)[0] + '.png'
#             instance_path = os.path.join(instances_folder, instance_name)

#             if not os.path.exists(instance_path):
#                 print(f"Instancia no encontrada para {img_file}, saltando.")
#                 continue

#             futures.append(executor.submit(_process_single_image, image_path, instance_path, pending_classes, tile_size))

#         for future in as_completed(futures):
#             result_tiles = future.result()
#             for class_id, tile in result_tiles.items():
#                 if class_id not in all_tiles:
#                     all_tiles[class_id] = class_id
#                     cv2.imwrite(f"Imagenes_Resultantes/Clase_{inverted_mapping[class_id]}_(ID_{class_id}).jpg", tile)
#                     pending_classes.discard(class_id)
            
#             print(f"Me queda {pending_classes}")
#             if not pending_classes:
#                 print("Se ha conseguido una muestra para cada clase.")
#                 break
    
#     if pending_classes:
#         print(f"No se encontraron muestras para las siguientes clases: {pending_classes}")


def process_all_images_until_full_coverage_concurrent(images_folder, instances_folder, id_to_classname, tile_size=128, max_workers=4):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import os
    import cv2

    inverted_mapping = {v: k for k, v in id_to_classname.items()}
    pending_classes = set(id_to_classname.values())
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    os.makedirs("Imagenes_Resultantes", exist_ok=True)

    lock = Lock()

    def process_one(img_file):
        nonlocal pending_classes
        if not pending_classes:
            return  # ya cubierto todo
        image_path = os.path.join(images_folder, img_file)
        instance_name = os.path.splitext(img_file)[0] + '.png'
        instance_path = os.path.join(instances_folder, instance_name)
        if not os.path.exists(instance_path):
            return

        result_tiles = _process_single_image(image_path, instance_path, pending_classes.copy(), tile_size)
        with lock:
            for class_id, tile in result_tiles.items():
                if class_id in pending_classes:
                    filename = f"Imagenes_Resultantes/Clase_{inverted_mapping[class_id]}_(ID_{class_id}).jpg"
                    cv2.imwrite(filename, tile)
                    pending_classes.discard(class_id)
            print(f"Me queda {pending_classes}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_one, image_files)
        print("Finalizado.")

    if pending_classes:
        print(f"No se encontraron muestras para las siguientes clases: {pending_classes}")
    else:
        print("Se ha conseguido una muestra para cada clase.")

def convert_yolo_segmentation_to_bounding_box(input_txt_path, output_txt_path):
    """
    Convierte anotaciones de polígonos en formato YOLO a anotaciones de bounding boxes en formato YOLO.
    
    Args:
        input_txt_path (str): Ruta al archivo de entrada de polígonos (YOLO segmentación).
        output_txt_path (str): Ruta donde guardar el archivo de salida (YOLO bounding boxes).
    """
    if not os.path.exists(input_txt_path):
        print(f"Error: No se encuentra el archivo {input_txt_path}")
        return

    lines_out = []

    with open(input_txt_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            # Debe tener un class_id seguido de pares (x,y) => mínimo 3 puntos
            print(f"Advertencia: línea ignorada por formato incorrecto -> {line.strip()}")
            continue

        class_id = parts[0]
        coords = list(map(float, parts[1:]))

        xs = coords[0::2]  # Extrae coordenadas x
        ys = coords[1::2]  # Extrae coordenadas y

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # Monta línea formato YOLO bounding box
        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        lines_out.append(yolo_line)

    # Guardar resultados
    with open(output_txt_path, 'w') as f:
        for line in lines_out:
            f.write(line + '\n')

    print(f"Conversión completada: {output_txt_path}")

""" image_path = '/home/lluis/Escritorio/AARON/Mapillary_model/mapillary/dataset/training/images/dojFUiCN6nSetZGUwnPthw.jpg'
with open('/home/lluis/Escritorio/AARON/Mapillary_model/mapillary/dataset/training/v2.0/polygons/dojFUiCN6nSetZGUwnPthw.json', 'r') as f:
    data = json.load(f)
image = cv2.imread(image_path)
if image is None:
    print(f"Error: No se pudo cargar la imagen {image_path}")
    

image_height, image_width, _ = image.shape



label_mapping = {
    "nature--sky": 0,
    "construction--flat--road": 1,
    "construction--flat--sidewalk": 2,
    "construction--flat--driveway": 3,

}


yolo_labels = convert_polygons_to_yolo_segmentation(data, image_width, image_height, label_mapping)


output_folder='./output/labels/transformadas'
os.makedirs(output_folder, exist_ok=True)
with open(output_folder+'/dojFUiCN6nSetZGUwnPthw.txt', 'w') as f:
    for line in yolo_labels:
        f.write(line + '\n') """
def process_all_images(images_folder, json_folder, output_folder, label_mapping):
    os.makedirs(output_folder, exist_ok=True)

    # Procesar todas las imágenes en la carpeta
    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_folder, filename)
            basename = os.path.splitext(filename)[0]

            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                continue

            image_height, image_width, _ = image.shape

            # Cargar JSON
            json_path = os.path.join(json_folder, basename + '.json')
            if not os.path.exists(json_path):
                print(f"Advertencia: No se encontró el JSON para {basename}")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Convertir a etiquetas YOLO
            yolo_labels = convert_polygons_to_yolo_segmentation(data, image_width, image_height, label_mapping)

            # Guardar etiquetas
            output_label_path = os.path.join(output_folder, basename + '.txt')
            with open(output_label_path, 'w') as f:
                for line in yolo_labels:
                    f.write(line + '\n')

            print(f"Procesado: {basename}")

    print("Transformación completada para todas las imágenes.")




def generate_yolo_polygon_labels(instance_label_array, label_mapping, config_path, output_txt_path):
    height, width = instance_label_array.shape

    # Cargar configuración de clases de Mapillary
    with open(config_path, 'r') as f:
        config = json.load(f)
    full_labels = config['labels']

    # Crear diccionario: nombre → id real
    name_to_id = {label['name']: idx for idx, label in enumerate(full_labels)}
    # Crear mapeo definitivo: id real → nuevo id
    id_real_to_class_id = {}

    if label_mapping is not None:
        # Crear un mapeo para convertir los class_id a índices consecutivos (0-13)
        unique_class_ids = sorted(set(label_mapping.values()))
        yolo_id_mapping = {old_id: idx for idx, old_id in enumerate(unique_class_ids)}

        for name, new_class_id in label_mapping.items():
            real_id = name_to_id.get(name)
            if real_id is not None:
                # Usar el mapeo a índices consecutivos
                id_real_to_class_id[real_id] = yolo_id_mapping[new_class_id]
    else:
        # Mapeo 1:1 (ID real --> mismo ID)
        id_real_to_class_id = {idx: idx for idx in range(len(name_to_id))}

    # Procesar instancias
    with open(output_txt_path, 'w') as f:
        unique_labels = np.unique(instance_label_array)

        for label_id in unique_labels:
            if label_id == 0:
                continue  # Ignorar fondo

            # Si no está en el mapping, saltarlo
            if label_id not in id_real_to_class_id:
                continue

            class_id = id_real_to_class_id[label_id]

            mask = (instance_label_array == label_id).astype(np.uint8) * 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3:
                    continue  # Necesita al menos 3 puntos

                # Simplificar contorno
                epsilon = 0.01 * cv2.arcLength(contour, True)
                simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

                if len(simplified_contour) < 3:
                    continue  # Si tras simplificar no hay polígono, saltar

                normalized_points = []
                for point in simplified_contour.squeeze():
                    x, y = point
                    norm_x = x / width
                    norm_y = y / height
                    normalized_points.extend([norm_x, norm_y])

                # Guardar línea en formato YOLO
                line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_points)
                f.write(line + '\n')


    #return label_mapping
def sample_class_slices(image_path, instance_label_array, id_to_classname, tile_size=240):
    """
    Genera un recorte por cada clase presente en instance_label_array a partir de una ruta de imagen.

    Args:
        image_path (str): Ruta a la imagen original.
        instance_label_array (np.ndarray): Array de etiquetas (IDs de clases).
        id_to_classname (dict): Diccionario {id_real: nombre_clase}.
        tile_size (int): Tamaño del recorte cuadrado (por defecto 128x128 píxeles).

    Returns:
        dict: {class_id: imagen_recorte}.
    """
    # Cargar imagen desde la ruta
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

    height, width = instance_label_array.shape
    sampled_tiles = {}

    unique_labels = np.unique(instance_label_array)

    for class_id in unique_labels:
        if class_id == 0:
            continue  # Ignorar fondo si es necesario

        id_to_classname = {v: k for k, v in id_to_classname.items()}
        if class_id not in id_to_classname:
            continue  # Ignorar clases que no estén en el mapping
        
        # Crear máscara de la clase
        mask = (instance_label_array == class_id).astype(np.uint8) * 255

        # Localizar píxeles de la clase
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        # Elegir un píxel aleatorio dentro de la clase
        idx = np.random.choice(len(xs))
        x_center, y_center = xs[idx], ys[idx]

        # Definir recorte
        x_start = max(x_center - tile_size // 2, 0)
        y_start = max(y_center - tile_size // 2, 0)
        x_end = min(x_start + tile_size, width)
        y_end = min(y_start + tile_size, height)

        tile = image[y_start:y_end, x_start:x_end]

        sampled_tiles[class_id] = tile

    return sampled_tiles
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_all_instance_images(instance_folder, output_folder, config_path, label_mapping=None, max_images=20, num_workers=4):
    image_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    subsets = [
        ['train', 'training/v2.0/instances', 'training/images'],
        ['val', 'validation/v2.0/instances', 'validation/images']
    ]

    os.makedirs(output_folder, exist_ok=True)

    def process_file(subset_folder, subset_output_folder, subset_input_image_folder, subset_output_image_folder, filename):
        if not filename.lower().endswith(image_formats):
            return False  # no contarlo

        image_id, _ = os.path.splitext(filename)
        instance_path = os.path.join(subset_folder, filename)
        output_txt_path = os.path.join(subset_output_folder, f"{image_id}.txt")

        # Buscar imagen original asociada
        original_image_path = None
        for ext in image_formats:
            candidate = os.path.join(subset_input_image_folder, f"{image_id}{ext}")
            if os.path.exists(candidate):
                original_image_path = candidate
                break

        if original_image_path:
            img = cv2.imread(original_image_path)
            if img is not None:
                resized_img = cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_AREA)
                dest_path = os.path.join(subset_output_image_folder, os.path.basename(original_image_path))
                cv2.imwrite(dest_path, resized_img)
            else:
                print(f"[AVISO] No se pudo leer imagen: {original_image_path}")
                return False
        else:
            print(f"[AVISO] No se encontró imagen para: {image_id}")
            return False

        # Procesar máscara de instancia
        try:
            instance_array = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
            if instance_array.ndim == 3:
                instance_array = instance_array[:, :, 0]  # por si es RGB codificada
            instance_array = cv2.resize(instance_array, (1280, 1280), interpolation=cv2.INTER_NEAREST)
            instance_label_array = (instance_array // 256).astype(np.uint8)

            generate_yolo_polygon_labels(
                instance_label_array,
                label_mapping,
                config_path,
                output_txt_path
            )
            return True
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            return False

    for output_subset, input_subset, subset_input_image in subsets:
        subset_folder = os.path.join(instance_folder, input_subset)
        subset_output_folder = os.path.join(output_folder, output_subset, 'labels')
        subset_input_image_folder = os.path.join(instance_folder, subset_input_image)
        subset_output_image_folder = os.path.join(output_folder, output_subset, 'images')

        os.makedirs(subset_output_folder, exist_ok=True)
        os.makedirs(subset_output_image_folder, exist_ok=True)

        print(f"Procesando: {subset_folder}")

        filenames = [f for f in os.listdir(subset_folder) if f.lower().endswith(image_formats)]
        #filenames = filenames[:max_images]  # limitar número

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_file, subset_folder, subset_output_folder,
                                subset_input_image_folder, subset_output_image_folder, f)
                for f in filenames
            ]

            processed = 0
            for future in as_completed(futures):
                print("A")
                if future.result():
                    processed += 1
                if processed >= max_images:
                    break

    return label_mapping


images_folder='./mapillary/dataset/validation/images'
json_folder='./mapillary/dataset/validation/v2.0/polygons'
output_folder='./output/labels/transformadas'
""" label_mapping = {
    "nature--sky": 0,
    "construction--flat--road": 1,
    "construction--flat--sidewalk": 2,
    "construction--flat--driveway": 3,

} """
label_mapping ={
    """ "animal--bird": 0,
    "animal--ground-animal": 1,
    "construction--barrier--ambiguous": 2,
    "construction--barrier--concrete-block": 3,
    "construction--barrier--curb": 4,
    "construction--barrier--fence": 5,
    "construction--barrier--guard-rail": 6,
    "construction--barrier--other-barrier": 7,
    "construction--barrier--road-median": 8,
    "construction--barrier--road-side": 9,
    "construction--barrier--separator": 10,
    "construction--barrier--temporary": 11,
    "construction--barrier--wall": 12,
    "construction--flat--bike-lane": 13,
    "construction--flat--crosswalk-plain": 14,
    "construction--flat--curb-cut": 15,
    "construction--flat--driveway": 16,
    "construction--flat--parking": 17,
    "construction--flat--parking-aisle": 18,
    "construction--flat--pedestrian-area": 19,
    "construction--flat--rail-track": 20, """
    "construction--flat--road": 21,
   """  "construction--flat--road-shoulder": 22,
    "construction--flat--service-lane": 23, """
    "construction--flat--sidewalk": 24,
   """  "construction--flat--traffic-island": 25,
    "construction--structure--bridge": 26,
    "construction--structure--building": 27,
    "construction--structure--garage": 28,
    "construction--structure--tunnel": 29,"""
    "human--person--individual": 30,
    """
    "human--person--person-group": 31,
    "human--rider--bicyclist": 32,
    "human--rider--motorcyclist": 33, """
    "human--rider--other-rider": 34,
   """  "marking--continuous--dashed": 35,
    "marking--continuous--solid": 36,
    "marking--continuous--zigzag": 37,
    "marking--discrete--ambiguous": 38,
    "marking--discrete--arrow--left": 39,
    "marking--discrete--arrow--other": 40,
    "marking--discrete--arrow--right": 41,
    "marking--discrete--arrow--split-left-or-straight": 42,
    "marking--discrete--arrow--split-right-or-straight": 43,
    "marking--discrete--arrow--straight": 44,
    "marking--discrete--crosswalk-zebra": 45,
    "marking--discrete--give-way-row": 46,
    "marking--discrete--give-way-single": 47,
    "marking--discrete--hatched--chevron": 48,
    "marking--discrete--hatched--diagonal": 49,
    "marking--discrete--other-marking": 50,
    "marking--discrete--stop-line": 51,
    "marking--discrete--symbol--bicycle": 52,
    "marking--discrete--symbol--other": 53,
    "marking--discrete--text": 54,
    "marking-only--continuous--dashed": 55,
    "marking-only--discrete--crosswalk-zebra": 56,
    "marking-only--discrete--other-marking": 57,
    "marking-only--discrete--text": 58,
    "nature--mountain": 59,
    "nature--sand": 60, """
    "nature--sky": 61,
    """ "nature--snow": 62,
    "nature--terrain": 63,
    "nature--vegetation": 64,
    "nature--water": 65,
    "object--banner": 66,
    "object--bench": 67,
    "object--bike-rack": 68,
    "object--catch-basin": 69,
    "object--cctv-camera": 70,
    "object--fire-hydrant": 71,
    "object--junction-box": 72,
    "object--mailbox": 73,
    "object--manhole": 74,
    "object--parking-meter": 75,
    "object--phone-booth": 76,
    "object--pothole": 77,
    "object--sign--advertisement": 78,
    "object--sign--ambiguous": 79,
    "object--sign--back": 80,
    "object--sign--information": 81,
    "object--sign--other": 82,
    "object--sign--store": 83,
    "object--street-light": 84, """
    "object--support--pole": 85,
    """ "object--support--pole-group": 86,
    "object--support--traffic-sign-frame": 87,
    "object--support--utility-pole": 88,
    "object--traffic-cone": 89,
    "object--traffic-light--general-single": 90,
    "object--traffic-light--pedestrians": 91,
    "object--traffic-light--general-upright": 92,
    "object--traffic-light--general-horizontal": 93,
    "object--traffic-light--cyclists": 94,
    "object--traffic-light--other": 95,
    "object--traffic-sign--ambiguous": 96,
    "object--traffic-sign--back": 97,
    "object--traffic-sign--direction-back": 98,
    "object--traffic-sign--direction-front": 99,
    "object--traffic-sign--front": 100,
    "object--traffic-sign--information-parking": 101,
    "object--traffic-sign--temporary-back": 102,
    "object--traffic-sign--temporary-front": 103,
    "object--trash-can": 104, """
    "object--vehicle--bicycle": 105,
    #"object--vehicle--boat": 106,
    "object--vehicle--bus": 107,
    "object--vehicle--car": 108,
    #"object--vehicle--caravan": 109,
    "object--vehicle--motorcycle": 110,
    "object--vehicle--on-rails": 111,
    #"object--vehicle--other-vehicle": 112,
    "object--vehicle--trailer": 113,
    "object--vehicle--truck": 114,
    #"object--vehicle--vehicle-group": 115,
    #"object--vehicle--wheeled-slow": 116,
    #"object--water-valve": 117,
    #"void--car-mount": 118,
    #"void--dynamic": 119,
    #"void--ego-vehicle": 120,
    #"void--ground": 121,
    #"void--static": 122,
    "void--unlabeled": 123
}

label_mapping = {
    "construction--flat--road": 21,
    "construction--flat--sidewalk": 24,
    "human--person--individual": 30,
    "human--rider--other-rider": 34,
    "nature--sky": 61,
    "object--support--pole": 85,
    "object--vehicle--bicycle": 105,
    "object--vehicle--bus": 107,
    "object--vehicle--car": 108,
    "object--vehicle--motorcycle": 110,
    "object--vehicle--on-rails": 111,
    "object--vehicle--trailer": 113,  # OJO: esto parece un error de índice, probablemente debería ser distinto
    "object--vehicle--truck": 114,
    "void--unlabeled": 123
}


#instance_path = "./mapillary/dataset/training/v2.0/instances/dojFUiCN6nSetZGUwnPthw.png"
instance_folder=r"C:\Users\aaron\Documents\GitHub\DeepLabV3Plus-Pytorch\mapillary"
output_folder=r"C:\Users\aaron\Documents\Año4\TFG\CityScapesWithPanopticSegm\Mapillary_dataset_instance"
#instance_image = Image.open(instance_path)
#instance_array = np.array(instance_image, dtype=np.uint16)
#instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
config_path = r'C:\Users\aaron\Documents\GitHub\DeepLabV3Plus-Pytorch\mapillary/config_v2.0.json'

process_all_instance_images(instance_folder, output_folder, config_path, label_mapping)
exit()

#tiles = sample_class_slices('/home/lluis/Escritorio/AARON/Mapillary_model/mapillary/dataset/training/images/dojFUiCN6nSetZGUwnPthw.jpg', instance_label_array, label_mapping, tile_size=240)
#process_all_images_until_full_coverage_concurrent(images_folder, instance_folder, label_mapping, tile_size=1024)

# Visualizar los recortes
""" with open(config_path, 'r') as f:
        config = json.load(f)
full_labels = config['labels']
full_labels={label['name']: idx for idx, label in enumerate(full_labels)}
id_to_name = {v: k for k, v in full_labels.items()}
for class_id, tile in tiles.items():
    cv2.imwrite(f"Imagenes_Resultantes/Clase_{id_to_name[class_id]}_(ID_{class_id}).jpg", tile)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit() """
#draw_yolo_bboxes('/home/lluis/Escritorio/AARON/Mapillary_model/output/imagenes/recortadas/dojFUiCN6nSetZGUwnPthw_0_0.jpg', './output/labels/to_bbox/dojFUiCN6nSetZGUwnPthw_0_0.txt')

input_image_folder = "./dataset/val/images"
input_label_folder = "./dataset/val/labels"
output_image_folder = "./output/for_dataset_recortadas/imagenes"
output_label_folder = "./output/for_dataset_recortadas/labels"
for filename in os.listdir(input_image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_image_folder, filename)
        label_path = os.path.join(input_label_folder, os.path.splitext(filename)[0] + '.txt')
        print(label_path)
        
        split_image_and_polygon_labels(
            image_path=image_path,
            label_path=label_path,
            output_image_folder=output_image_folder,
            output_label_folder=output_label_folder,
            tile_size=640
        )
        #convert_yolo_segmentation_to_bounding_box('/home/lluis/Escritorio/AARON/Mapillary_model/output/labels/recortadas/dojFUiCN6nSetZGUwnPthw_0_0.txt','./output/labels/to_bbox/dojFUiCN6nSetZGUwnPthw_0_0.txt')