import os
import shutil
def move_labels(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.txt'):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy(src_path, dest_path)

# Define las rutas
move_labels('data/cityscapes/gtFine/train', 'dataset/labels/train')
move_labels('data/cityscapes/gtFine/val', 'dataset/labels/val')
move_labels('data/cityscapes/gtFine/test', 'dataset/labels/test')
