from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from datasets.mapillary import MapillaryVistas, MapillaryTransform
def main():
    transform = MapillaryTransform(base_size=512, crop_size=512)

    dataset = MapillaryVistas(
        root='./mapillary',
        split='training',
        version='v2.0',
        transform=transform
    )

    # Obtener una muestra
    image, mask = dataset[4]

    # Visualizar
    image = image.numpy().transpose(1, 2, 0)
    image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

    colored_mask = dataset.decode_target(mask)

    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Imagen Original')
    plt.subplot(132)
    plt.imshow(mask)
    plt.title('Máscara (IDs)')
    plt.subplot(133)
    plt.imshow(colored_mask)
    plt.title('Máscara (Colores)')
    plt.show()

if __name__ == '__main__':
    main()