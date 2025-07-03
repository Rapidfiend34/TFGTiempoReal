import cv2
import numpy as np
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
np.float = float
# Inicializa YOLOv8
model = YOLO('yolov8n.pt')

# Define los argumentos para ByteTrack
class Args:
    track_thresh = 0.5
    match_thresh = 0.8
    track_buffer = 30
    min_box_area = 10
    mot20 = False

args = Args()
tracker = BYTETracker(args)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferencia con YOLOv8
    results = model(frame, verbose=False)
    dets = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf)
        if cls == 0 and conf > 0.3:  # Solo personas
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # ByteTrack espera [x1, y1, x2, y2, score]
            dets.append([x1, y1, x2, y2, conf])

    # Convierte a np.ndarray
    dets = np.array(dets)
    img_info = frame.shape[:2]  # (alto, ancho)
    img_size = frame.shape[:2]  # (alto, ancho)

    # Si no hay detecciones, crea un array vacío con la forma correcta
    if dets.size == 0:
        dets = np.empty((0, 5))

    # Llama a ByteTrack
    tracks = tracker.update(dets, img_info, img_size)

    # Visualiza los tracks
    for track in tracks:
        tlwh = track.tlwh
        x1, y1, w, h = map(int, tlwh)
        x2, y2 = x1 + w, y1 + h
        track_id = int(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('ByteTrack + YOLOv8', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()