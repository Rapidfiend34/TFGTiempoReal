import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r'C:\Users\aaron\Documents\Año4\TFG\CityScapesWithPanopticSegm\runs\segment\train83\weights\best.pt'
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    results = model(frame, conf=0.2, verbose=False)
    result = results[0]
    img = frame.copy()

    # Dibuja solo las detecciones de clase id==2
    if hasattr(result, "boxes"):
        boxes = result.boxes
        for i, box in enumerate(boxes):
            class_id = int(box.cls)
            if class_id == 2:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'ID: {class_id} ({conf:.2f})'
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Si el modelo es -seg y quieres la máscara:
                if hasattr(result, "masks") and result.masks is not None:
                    mask = result.masks.data[i].cpu().numpy()
                    # Redimensiona la máscara al tamaño del frame
                    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (mask_resized > 0.5).astype(np.uint8)
                    color = np.array([0, 255, 0], dtype=np.uint8)
                    colored_mask = np.zeros_like(img, dtype=np.uint8)
                    for c in range(3):
                        colored_mask[:, :, c] = mask_bin * color[c]
                    img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

    cv2.imshow("Solo clase id==2", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()