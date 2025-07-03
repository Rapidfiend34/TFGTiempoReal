import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# Método 1: ByteTrack integrado en YOLOv8 (Más rápido)
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
from collections import defaultdict

def main():
    # Cargar modelos
    detect_model = YOLO('yolov8n.pt')
    classifier = YOLO(r'C:\Users\aaron\Documents\Año4\TFG\Classifier\runs\classify\Mejores_matrix\weights\best.pt')

    # Diccionario para clasificaciones
    track_classifications = defaultdict(lambda: {"class": None, "conf": 0.0})

    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        TRACKING_CLASSES={
            0:'Person'
        }
        # Detección y tracking en un solo paso
        results = detect_model.track(
            source=frame,
            conf=0.2,
            iou=0.3,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0]
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            # Procesar cada detección
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = map(int, box)

                # Extraer ROI para clasificación
                roi = frame[y1:y2, x1:x2]

                if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                    try:
                        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        class_results = classifier.predict(roi_pil, imgsz=224)
                        
                        pred_class = class_results[0].names[class_results[0].probs.top1]
                        pred_prob = class_results[0].probs.top1conf.item()

                        if pred_prob > 0.5:
                            track_classifications[track_id] = {
                                "class": pred_class,
                                "conf": pred_prob
                            }

                    except Exception as e:
                        continue

                # Dibujar bbox y etiqueta
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                classification = track_classifications[track_id]
                if classification["class"] is not None:
                    label = f"ID:{track_id} {classification['class']} ({classification['conf']:.2f})"
                else:
                    label = f"ID:{track_id}"

                cv2.putText(frame, label, (x1, y1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow('YOLOv8 Tracking + Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()