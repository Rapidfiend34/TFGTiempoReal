import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import numpy as np
import torch
import cv2
from collections import deque
import time
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import argparse
np.float=float
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='',
                        help='Ruta al archivo de video. Si está vacío, se usa la cámara.')
    parser.add_argument('--save', action='store_true',
                        help='Guardar el video de salida')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Ruta para guardar el video de salida')
    parser.add_argument('--original_size', action='store_true',
                        help='Mantener tamaño original del video')
    return parser.parse_args()

# Clase de argumentos para ByteTrack
class Args:
    track_thresh = 0.3
    match_thresh = 0.8
    track_buffer = 20
    min_box_area = 10
    mot20 = False

def main():
    args = get_args()

    # Inicializar el modelo YOLO
    yolo_model = YOLO(r'yolov8n-seg.pt')

    # Inicializar ByteTrack
    bytetrack_args = Args()
    tracker = BYTETracker(bytetrack_args)

    # FPS y conteo de tracks
    fps_deque = deque(maxlen=30)
    prev_time = time.time()

    # Configurar captura de video o cámara
    if args.video:
        print(f"Usando video: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video: {args.video}")
            return

        # Obtener dimensiones originales del video
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Tamaño original del video: {original_width}x{original_height}")

        if not args.original_size:
            # Calcular factor de escala para mantener la relación de aspecto
            scale_factor = min(640/original_width, 480/original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            print(f"Tamaño redimensionado: {new_width}x{new_height}")
        else:
            new_width, new_height = original_width, original_height
            print("Manteniendo tamaño original del video")
    else:
        print("Usando cámara")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara 0")
            # Intentar con cámara 1
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("Error: No se pudo abrir ninguna cámara")
                return
        # Configurar resolución de la cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        new_width, new_height = 640, 480

    # Configurar writer para guardar video
    writer = None
    if args.save:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:  # Si es cámara web y no se detecta FPS
            fps = 30
        writer = cv2.VideoWriter(args.output,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (new_width, new_height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame")
                break

            # Redimensionar si es necesario
            if args.video and not args.original_size:
                frame = cv2.resize(frame, (new_width, new_height))

            # Cálculo de FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            fps_deque.append(fps)
            prev_time = curr_time
            avg_fps = sum(fps_deque) / len(fps_deque)

            # Detección con YOLO
            results = yolo_model(frame, verbose=False)

            # Prepara las detecciones para ByteTrack
            dets = []
            if len(results) > 0:
                for box in results[0].boxes:
                    conf = float(box.conf)
                    cls = int(box.cls[0])
                    if conf > 0.3 and cls == 0:  # Solo personas
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        dets.append([x1, y1, x2, y2, conf])
            dets = np.array(dets)
            if dets.size == 0:
                dets = np.empty((0, 5))

            img_info = frame.shape[:2]  # (alto, ancho)
            img_size = frame.shape[:2]  # (alto, ancho)

            # Tracking con ByteTrack
            tracks = tracker.update(dets, img_info, img_size)

            # Dibuja las cajas de YOLO en azul
            if len(results) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Dibuja los tracks en verde
            track_ids = set()
            for track in tracks:
                tlwh = track.tlwh
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h
                track_id = int(track.track_id)
                track_ids.add(track_id)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, str(track_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Mostrar número de detecciones y tracks activos
            num_detections = len(results[0].boxes) if len(results) > 0 else 0
            num_tracks = len(track_ids)
            cv2.putText(frame, f'Detections: {num_detections}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Tracks: {num_tracks}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Guardar frame si es necesario
            if writer is not None:
                writer.write(frame)

            # Mostrar resultado
            cv2.imshow('Tracking', frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Limpieza
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()