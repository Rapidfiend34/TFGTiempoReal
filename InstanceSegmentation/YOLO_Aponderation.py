from gc import freeze
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# Cargar modelo
model = YOLO('runs/segment/train40/weights/best.pt')
""" results = model.train(
    data='cityscapes_wh.yaml',
    epochs=30,        # Más épocas para mejor convergencia
    imgsz=640,      # Mantener resolución alta
    batch=8,         # Mantener batch size
    workers=0,
    pretrained=True,
    freeze=10,
    patience=10,
    # Augmentaciones más agresivas para mejor generalización
    flipud=0.3,
    fliplr=0.5,
    mosaic=0.8,      # Aumentar para más variedad
    mixup=0.3,       # Aumentar para mejor robustez

    # Ajustes de color más agresivos
    hsv_h=0.015,     # Aumentar variación de tono
    hsv_s=0.5,       # Aumentar saturación
    hsv_v=0.4,       # Más variación de brillo

    # Transformaciones geométricas más agresivas
    degrees=30.0,    # Más rotación
    translate=0.15,  # Más traslación
    scale=0.5,      # Más escala
    shear=7.0,      # Aumentar shear
    perspective=0.0007, # Más perspectiva

    # Otras configuraciones
    copy_paste=0.3,  # Aumentar copy-paste
    augment=True,

    # Optimización ajustada
    warmup_epochs=3,
    warmup_momentum=0.85,
    lr0=0.001,     # Learning rate más equilibrado
    lrf=0.0001,    # Learning rate final ajustado
    weight_decay=0.0004,  # Más regularización

    # Nuevas configuraciones
    label_smoothing=0.1,  # Añadir label smoothing
    overlap_mask=True,    # Mejorar máscaras
    mask_ratio=4,        # Ajustar ratio de máscara
    dropout=0.2          # Añadir dropout
) """

results = model.train(
    data='cityscapes_wh.yaml',
    epochs=70,        # Más épocas para mejor convergencia
    imgsz=640,      # Mantener resolución alta
    batch=8,         # Mantener batch size
    workers=0,
    pretrained=True,
    freeze=10,
    patience=10,
    # Augmentaciones más agresivas para mejor generalización
    flipud=0.3,
    fliplr=0.5,
    mosaic=0.8,      # Aumentar para más variedad
    mixup=0.3,       # Aumentar para mejor robustez

    # Ajustes de color más agresivos
    hsv_h=0.015,     # Aumentar variación de tono
    hsv_s=0.5,       # Aumentar saturación
    hsv_v=0.4,       # Más variación de brillo

    # Transformaciones geométricas más agresivas
    degrees=30.0,    # Más rotación
    translate=0.15,  # Más traslación
    scale=0.5,      # Más escala
    shear=7.0,      # Aumentar shear
    perspective=0.0007, # Más perspectiva

    # Otras configuraciones
    copy_paste=0.3,  # Aumentar copy-paste
    augment=True,

    # Optimización ajustada
    warmup_epochs=3,
    warmup_momentum=0.85,
    lr0=0.001,     # Learning rate más equilibrado
    lrf=0.0001,    # Learning rate final ajustado
    weight_decay=0.0004,  # Más regularización

    # Nuevas configuraciones
    label_smoothing=0.1,  # Añadir label smoothing
    overlap_mask=True,    # Mejorar máscaras
    mask_ratio=4,        # Ajustar ratio de máscara
    dropout=0.2          # Añadir dropout
)