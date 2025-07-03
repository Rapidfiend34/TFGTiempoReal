import cv2
import os
from random import choice
import albumentations as A

# Transformaciones de data augmentation
""" augment = A.Compose([
    # Transformaciones originales
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.Blur(blur_limit=3, p=0.1),

    # Transformaciones adicionales sugeridas
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),  # Escalado, rotación y desplazamiento
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),  # Mejora de contraste local
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # Ruido gaussiano
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  # Deformaciones elásticas
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),  # Distorsión geométrica
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),  # Simulación de oclusiones
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),  # Variaciones de color
    A.PiecewiseAffine(scale=(0.01, 0.05), p=0.2),  # Deformaciones locales
]) """

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),  # Reemplazo de ShiftScaleRotate
    A.GaussNoise(mean=0, var_limit=(10, 50), p=0.2),  # Corrección de GaussNoise
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # Corrección de ElasticTransform
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),  # Alternativa rápida a ElasticTransform
    A.CoarseDropout(max_holes=8, max_size=16, p=0.2),  # Corrección de CoarseDropout
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
])

def augment_images(src_folder, target_folder, target_count):
    # Crear carpeta de destino si no existe
    os.makedirs(target_folder, exist_ok=True)

    # Lista de imágenes originales (no augmentadas)
    images = [img for img in os.listdir(src_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_aug_count = len([f for f in os.listdir(target_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    i = current_aug_count

    while current_aug_count < target_count:
        img_name = choice(images)
        img_path = os.path.join(src_folder, img_name)

        # Intentar leer la imagen
        image = cv2.imread(img_path)
        if image is None:
            print(f"No se pudo leer: {img_path}")
            continue

        # Aplicar las transformaciones
        augmented = augment(image=image)["image"]

        # Guardar la imagen aumentada
        new_name = os.path.join(target_folder, f"aug_{i}_{img_name}")
        cv2.imwrite(new_name, augmented)
        print(f"Imagen aumentada guardada en: {new_name}")
        i += 1
        current_aug_count += 1

# Llamada a la función con las rutas correspondientes
augment_images('./output/Ninos/train/elder', './Augmented/ninos', 1200)
