import os
import shutil

def move_and_rename_images(src_dir, dest_dir):
    """
    Copia y renombra imágenes desde el directorio fuente al directorio destino.
    """
    os.makedirs(dest_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(('.png', '.jpg')):
                src_path = os.path.join(root, file)

                # Separar nombre y extensión
                name, ext = os.path.splitext(file)

                # Reemplazar en el nombre (sin la extensión)
                new_name = name.replace("_leftImg8bit", "_gtFine_polygons") + ext

                # Ruta de destino con el nuevo nombre
                dest_path = os.path.join(dest_dir, new_name)

                # Copiar el archivo
                shutil.copy(src_path, dest_path)
                print(f"Imagen copiada y renombrada: {src_path} -> {dest_path}")

# Define las rutas
move_and_rename_images('data/cityscapes/leftImg8bit/train', 'dataset/images/train')
move_and_rename_images('data/cityscapes/leftImg8bit/val', 'dataset/images/val')
move_and_rename_images('data/cityscapes/leftImg8bit/test', 'dataset/images/test')