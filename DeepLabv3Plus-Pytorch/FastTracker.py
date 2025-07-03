import numpy as np
import cv2
from collections import deque
import time
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort

class FastTracker:
    def __init__(self,
                 det_thresh=0.1,
                 max_age=200,
                 min_hits=3,
                 iou_threshold=0.1,
                 delta_t=3,
                 track_history=30):
        """
        Inicializa el tracker
        """
        self.tracker = OCSort(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t
        )
        self.track_history = track_history
        self.tracks_dict = {}
        self.fps = 0
        self.frame_count = 0
        self.time_start = time.time()
        self.original_size = None
        self.yolo_size = (640, 640)  # Tamaño al que YOLO redimensiona

    def _scale_boxes(self, boxes, original_size):
        """
        Escala las cajas del tamaño de YOLO al tamaño original
        """
        if boxes.size == 0:
            return boxes

        # Obtener factores de escala
        scale_x = original_size[1] / self.yolo_size[1]
        scale_y = original_size[0] / self.yolo_size[0]

        # Aplicar escala a las coordenadas
        boxes[:, [0, 2]] *= scale_x  # x1, x2
        boxes[:, [1, 3]] *= scale_y  # y1, y2

        return boxes

    def _process_detections(self, results, original_frame):
        """
        Procesa las detecciones de YOLO
        """
        if not results or len(results) == 0:
            return np.empty((0, 5))

        # Guardar tamaño original
        self.original_size = original_frame.shape[:2]

        dets = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Obtener la clase del objeto
                cls = int(box.cls[0])
                # Filtrar solo personas (clase 0)
                if cls == 0:  # Solo personas
                    # Obtener coordenadas y confianza
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)

                    # Asegurarse de que las coordenadas son válidas
                    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

                    # Verificar que el bbox tiene dimensiones positivas
                    if x2 > x1 and y2 > y1:
                        dets.append([x1, y1, x2, y2, conf])

        dets = np.array(dets, dtype=np.float32) if dets else np.empty((0, 5))

        # Escalar las cajas al tamaño original
        if len(dets) > 0:
            dets[:, :4] = self._scale_boxes(dets[:, :4], self.original_size)
        
        return dets

    def update(self, results, frame):
        """
        Actualiza el tracker con nuevas detecciones
        """
        self.frame_count += 1
        
        # Extraer detecciones de YOLO
        output_results = self._process_detections(results, frame)

        # Obtener dimensiones de la imagen
        height, width = frame.shape[:2]
        img_info = np.array([height, width])
        img_size = np.array([width, height])

        # Actualizar tracker
        tracks = self.tracker.update(output_results, img_info, img_size)

        # Actualizar historial de tracks
        self._update_tracks_history(tracks)

        # Calcular FPS
        if self.frame_count % 30 == 0:
            self.fps = self.frame_count / (time.time() - self.time_start)

        return tracks

    def _update_tracks_history(self, tracks):
        """
        Actualiza el historial de tracks
        """
        for track in tracks:
            track_id = int(track[4])
            if track_id not in self.tracks_dict:
                self.tracks_dict[track_id] = deque(maxlen=self.track_history)
            self.tracks_dict[track_id].append(track[:4])

    def draw_tracks(self, frame, tracks):
        """
        Dibuja los tracks en el frame
        """
        frame_with_tracks = frame.copy()

        # Dibujar FPS y número de tracks
        cv2.putText(frame_with_tracks, f'FPS: {self.fps:.1f}',
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_tracks, f'Tracks: {len(tracks)}',
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Colores para los tracks
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
                 (255, 255, 0), (0, 255, 255), (255, 0, 255)]

        for track in tracks:
            if len(track) < 5:
                continue

            # Extraer información del track
            bbox = track[:4].astype(int)
            track_id = int(track[4])

            # Seleccionar color basado en el ID
            color = colors[track_id % len(colors)]

            # Dibujar bounding box
            cv2.rectangle(frame_with_tracks,
                         (bbox[0], bbox[1]),
                         (bbox[2], bbox[3]),
                         color, 2)

            # Dibujar ID
            label = f'ID: {track_id}'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_with_tracks,
                         (bbox[0], bbox[1] - t_size[1] - 10),
                         (bbox[0] + t_size[0], bbox[1]),
                         color, -1)
            cv2.putText(frame_with_tracks,
                       label,
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (255, 255, 255),
                       2)

            # Dibujar trayectoria
            if track_id in self.tracks_dict:
                history = list(self.tracks_dict[track_id])
                for i in range(1, len(history)):
                    if i > 1:
                        pt1 = (int((history[i-1][0] + history[i-1][2])/2),
                              int((history[i-1][1] + history[i-1][3])/2))
                        pt2 = (int((history[i][0] + history[i][2])/2),
                              int((history[i][1] + history[i][3])/2))
                        cv2.line(frame_with_tracks, pt1, pt2, color, 2)

        return frame_with_tracks