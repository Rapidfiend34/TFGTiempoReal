import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from pathlib import Path
import numpy as np
import torch
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
import cv2
from collections import deque
import time
from ultralytics import YOLO
from FastTracker import FastTracker
import argparse

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

def main():
    args = get_args()

    # Inicializar el modelo YOLO
    yolo_model = YOLO(r'yolov8n-seg.pt')

    # Inicializar tracker
    tracker = FastTracker(
    det_thresh=0.3,
    max_age=20,
    min_hits=2,
    iou_threshold=0.3,
    delta_t=3
)

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

            print("Frame original shape:", frame.shape)

            # Redimensionar si es necesario
            if args.video and not args.original_size:
                frame = cv2.resize(frame, (new_width, new_height))
                print("Frame redimensionado shape:", frame.shape)
            else:
                print("Frame sin redimensionar shape:", frame.shape)

            # Detección con YOLO
            print("Antes de YOLO, frame shape:", frame.shape)
            results = yolo_model(frame, verbose=False)

            # Imprime las cajas de YOLO
            if len(results) > 0:
                print("YOLO boxes:")
                for box in results[0].boxes:
                    # Si es tensor, conviértelo a numpy para ver los valores
                    if hasattr(box.xyxy, 'cpu'):
                        print(box.xyxy.cpu().numpy())
                    else:
                        print(box.xyxy)

            # Tracking
            print("Antes de tracker.update, frame shape:", frame.shape)
            tracks = tracker.update(results, frame)
            # Dibuja las cajas de YOLO en azul
            if len(results) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Dibuja los tracks en rojo
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                color = (0, 0, 255) if track_id == 2 else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, str(track_id), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            # Imprime los tracks
            print("Tracks:")
            print(tracks)

            # Visualización
            print("Antes de draw_tracks, frame shape:", frame.shape)
            frame_with_tracks = tracker.draw_tracks(frame, tracks)
            print("Después de draw_tracks, frame_with_tracks shape:", frame_with_tracks.shape)

            # Mostrar número de detecciones
            if len(results) > 0:
                num_detections = len(results[0].boxes)
                cv2.putText(frame_with_tracks, f'Detections: {num_detections}',
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Guardar frame si es necesario
            if writer is not None:
                print("Antes de guardar, frame_with_tracks shape:", frame_with_tracks.shape)
                writer.write(frame_with_tracks)

            # Mostrar resultado
            print("Antes de imshow, frame_with_tracks shape:", frame_with_tracks.shape)
            cv2.imshow('Tracking', frame_with_tracks)

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