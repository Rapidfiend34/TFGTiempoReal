import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import network
import argparse
import numpy as np
import torch.nn.functional as F
import threading
import queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time
import gc

from semantic_validation_postprocessor import (
    SemanticValidationPostProcessor,
    apply_semantic_validation,
    draw_validated_boxes
)

from torchvision import transforms as T
from ultralytics import YOLO
import torch
from collections import OrderedDict, deque
from PIL import Image
from glob import glob

np.float = float

# Configuración global de optimización
PROCESS_EVERY_N_FRAMES = 2  # Procesar 1 de cada N frames (reducido para pipeline)
UPDATE_VISUALIZATION_EVERY_N_FRAMES = 2  # Actualizar visualización cada N frames
CLASSIFICATION_TIMEOUT = 5.0  # Segundos antes de reclasificar
INPUT_RESOLUTION = (480, 640)  # Resolución de entrada reducida
CLASSIFICATION_RESOLUTION = 224  # Resolución para clasificación
DETECTION_RESOLUTION = 416  # Resolución para detección
CONFIDENCE_THRESHOLD = 0.2  # Umbral de confianza para detección

# Diccionario global para almacenar clasificaciones
track_classifications = {}

# Mapeo de clases de interés de Mapillary
MAPILLARY_CLASSES = {
    'road': 2,          # construction--flat--road
    'sidewalk': 3,      # construction--flat--sidewalk
    'person': 4,        # human--person--individual
    'rider': 5,         # human--rider--other-rider
    'bicycle': 9,       # object--vehicle--bicycle
    'bus': 10,         # object--vehicle--bus
    'car': 11,         # object--vehicle--car
    'motorcycle': 12,   # object--vehicle--motorcycle
    'truck': 13        # object--vehicle--truck
}

# Colores para visualización
COLOR_MAP = {
    2: (128, 64, 128),   # Road: azul grisáceo
    3: (244, 35, 232),   # Sidewalk: rosa
    4: (220, 20, 60),    # Person: rojo
    5: (255, 0, 200),    # Rider: magenta
    9: (119, 11, 32),    # Bicycle: rojo oscuro
    10: (0, 60, 100),    # Bus: azul oscuro
    11: (0, 0, 142),     # Car: azul
    12: (0, 0, 230),     # Motorcycle: azul brillante
    13: (0, 0, 70),      # Truck: azul muy oscuro
    255: (0, 0, 0)       # Fondo: negro
}

@dataclass
class FrameData:
    """Estructura para datos del frame en el pipeline"""
    frame_id: int
    timestamp: float
    original_frame: np.ndarray
    processed_frame: Optional[np.ndarray] = None
    semantic_mask: Optional[np.ndarray] = None
    color_mask: Optional[np.ndarray] = None
    detections: Optional[list] = None
    validated_detections: Optional[list] = None
    classifications: Optional[Dict] = None
    visualization: Optional[np.ndarray] = None
    processing_times: Dict[str, float] = None
    rotation_angle: int = 0

    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = {}

class AsyncPipeline:
    def __init__(self, model, yolo_model, classifier_model, device, transform, 
                 semantic_processor, max_workers=3, buffer_size=8):
        
        # Modelos
        self.model = model
        self.yolo_model = yolo_model
        self.classifier_model = classifier_model
        self.device = device
        self.transform = transform
        self.semantic_processor = semantic_processor
        
        # Configuración del pipeline
        self.max_workers = max_workers
        self.buffer_size = buffer_size
        
        # Colas para cada etapa del pipeline
        self.input_queue = queue.Queue(maxsize=buffer_size)
        self.semantic_queue = queue.Queue(maxsize=buffer_size)
        self.detection_queue = queue.Queue(maxsize=buffer_size)
        self.classification_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue(maxsize=buffer_size)
        
        # Thread pools especializados
        self.semantic_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Semantic")
        self.detection_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Detection")
        self.classification_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Classification")
        self.visualization_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Visualization")
        
        # Control y estadísticas
        self.running = True
        self.paused = False
        self.stats = {
            'frames_processed': 0,
            'frames_dropped': 0,
            'avg_latency': 0,
            'stage_times': {
                'semantic': deque(maxlen=30),
                'detection': deque(maxlen=30),
                'classification': deque(maxlen=30),
                'visualization': deque(maxlen=30)
            },
            'queue_sizes': {
                'input': 0,
                'semantic': 0,
                'detection': 0,
                'classification': 0,
                'output': 0
            }
        }
        
        # Iniciar workers
        self._start_workers()
        print("🚀 Pipeline asíncrono iniciado con {} workers".format(max_workers))
    
    def _start_workers(self):
        """Iniciar todos los workers del pipeline"""
        
        # Worker para segmentación semántica
        threading.Thread(target=self._semantic_worker, daemon=True, name="SemanticWorker").start()
        
        # Worker para detección YOLO
        threading.Thread(target=self._detection_worker, daemon=True, name="DetectionWorker").start()
        
        # Worker para clasificación
        threading.Thread(target=self._classification_worker, daemon=True, name="ClassificationWorker").start()
        
        # Worker para visualización
        threading.Thread(target=self._visualization_worker, daemon=True, name="VisualizationWorker").start()
        
        print("✅ Todos los workers del pipeline iniciados")
    
    def _semantic_worker(self):
        """Worker dedicado para segmentación semántica"""
        print("🧠 Semantic worker iniciado")
        
        while self.running:
            try:
                frame_data = self.input_queue.get(timeout=1.0)
                if frame_data is None:
                    break
                
                # Verificar pausa
                while self.paused and self.running:
                    time.sleep(0.1)
                
                if not self.running:
                    break
                
                start_time = time.time()
                
                # Aplicar rotación si es necesario
                frame = frame_data.original_frame
                if frame_data.rotation_angle != 0:
                    frame = correct_frame_rotation(frame, frame_data.rotation_angle)
                # Procesar frame
                frame = cv2.resize(frame, INPUT_RESOLUTION)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Segmentación semántica
                with torch.no_grad():
                    img_pil = Image.fromarray(frame_rgb)
                    img = self.transform(img_pil).unsqueeze(0)
                    img = F.interpolate(img, size=(320, 320), mode='bilinear', align_corners=True)
                    img = img.to(self.device)
                    
                    with torch.cuda.amp.autocast():
                        outputs = self.model(img)
                    
                    pred = outputs.max(1)[1].cpu().numpy()[0]
                
                # Redimensionar máscara
                semantic_mask = cv2.resize(pred, (frame_rgb.shape[1], frame_rgb.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                color_mask = create_color_mask_optimized(semantic_mask)
                
                # Actualizar frame_data
                frame_data.processed_frame = frame_rgb
                frame_data.semantic_mask = semantic_mask
                frame_data.color_mask = color_mask
                frame_data.processing_times['semantic'] = time.time() - start_time
                
                # Enviar a siguiente etapa
                try:
                    self.semantic_queue.put_nowait(frame_data)
                except queue.Full:
                    self.stats['frames_dropped'] += 1
                    print("⚠️ Frame dropped en semantic queue")
                
                # Actualizar estadísticas
                self.stats['stage_times']['semantic'].append(frame_data.processing_times['semantic'])
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error en semantic worker: {e}")
        
        print("🛑 Semantic worker terminado")
    
    def _detection_worker(self):
        """Worker dedicado para detección YOLO"""
        print("🎯 Detection worker iniciado")
        
        while self.running:
            try:
                frame_data = self.semantic_queue.get(timeout=1.0)
                if frame_data is None:
                    break
                
                # Verificar pausa
                while self.paused and self.running:
                    time.sleep(0.1)
                
                if not self.running:
                    break
                
                start_time = time.time()
                
                # Detección con YOLO
                results = self.yolo_model.track(
                    source=frame_data.processed_frame,
                    conf=CONFIDENCE_THRESHOLD,
                    classes=[0,13],
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                    imgsz=DETECTION_RESOLUTION,
                    device=0
                )
                
                # Validación semántica
                validated_detections = apply_semantic_validation(
                    results, frame_data.semantic_mask, self.semantic_processor
                )
                
                # Actualizar frame_data
                frame_data.detections = results
                frame_data.validated_detections = validated_detections
                frame_data.processing_times['detection'] = time.time() - start_time
                
                # Enviar a siguiente etapa
                try:
                    self.detection_queue.put_nowait(frame_data)
                except queue.Full:
                    self.stats['frames_dropped'] += 1
                    print("⚠️ Frame dropped en detection queue")
                
                # Actualizar estadísticas
                self.stats['stage_times']['detection'].append(frame_data.processing_times['detection'])
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error en detection worker: {e}")
        
        print("🛑 Detection worker terminado")
    
    def _classification_worker(self):
        """Worker dedicado para clasificación"""
        print("🏷️ Classification worker iniciado")
        
        while self.running:
            try:
                frame_data = self.detection_queue.get(timeout=1.0)
                if frame_data is None:
                    break
                
                # Verificar pausa
                while self.paused and self.running:
                    time.sleep(0.1)
                
                if not self.running:
                    break
                
                start_time = time.time()
                current_time = time.time()
                
                # Clasificar solo las primeras 2 detecciones válidas para optimizar
                for detection in frame_data.validated_detections[:2]:
                    track_id = detection['track_id']
                    
                    if (track_id not in track_classifications or
                        current_time - track_classifications[track_id]['time'] > CLASSIFICATION_TIMEOUT):
                        
                        bbox = detection['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        x1=x1-15
                        y1=y1-15
                        x2=x2+15
                        y2=y2+15
                        roi = frame_data.processed_frame[y1:y2, x1:x2]
                        
                        if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                            roi_pil = Image.fromarray(roi)
                            class_results = self.classifier_model.predict(
                                roi_pil,
                                imgsz=CLASSIFICATION_RESOLUTION,
                                verbose=False,
                                device=0
                            )
                            
                            pred_class = class_results[0].names[class_results[0].probs.top1]
                            pred_prob = class_results[0].probs.top1conf.item()
                            
                            track_classifications[track_id] = {
                                'class': pred_class,
                                'conf': pred_prob,
                                'time': current_time
                            }
                
                frame_data.classifications = track_classifications.copy()
                frame_data.processing_times['classification'] = time.time() - start_time
                
                # Enviar a visualización
                try:
                    self.classification_queue.put_nowait(frame_data)
                except queue.Full:
                    self.stats['frames_dropped'] += 1
                    print("⚠️ Frame dropped en classification queue")
                
                # Actualizar estadísticas
                self.stats['stage_times']['classification'].append(frame_data.processing_times['classification'])
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error en classification worker: {e}")
        
        print("🛑 Classification worker terminado")
    
    def _visualization_worker(self):
        """Worker dedicado para crear visualizaciones"""
        print("🎨 Visualization worker iniciado")
        
        # Variables para cálculo correcto de FPS
        last_display_time = time.time()
        fps_history = deque(maxlen=10)  # Historial para suavizar FPS
        
        while self.running:
            try:
                frame_data = self.classification_queue.get(timeout=1.0)
                if frame_data is None:
                    break
                
                # Verificar pausa
                while self.paused and self.running:
                    time.sleep(0.1)
                
                if not self.running:
                    break
                
                start_time = time.time()
                
                # Crear visualización con detecciones validadas
                combined_visualization = draw_validated_boxes(
                    frame_data.processed_frame, 
                    frame_data.validated_detections, 
                    frame_data.classifications
                )
                
                # 🎯 CÁLCULO CORRECTO DE FPS
                current_time = time.time()
                display_fps = 1.0 / (current_time - last_display_time) if last_display_time else 0
                fps_history.append(display_fps)
                avg_display_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                last_display_time = current_time
                
                # Información de rendimiento corregida
                total_detections = len(frame_data.detections[0].boxes) if frame_data.detections[0].boxes.id is not None else 0
                valid_detections = len(frame_data.validated_detections)
                pipeline_latency = (current_time - frame_data.timestamp) * 1000  # Latencia en ms
                
                # Mostrar FPS real y latencia del pipeline
                cv2.putText(combined_visualization, f'Display FPS: {avg_display_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined_visualization, f'Pipeline Latency: {pipeline_latency:.0f}ms', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(combined_visualization, f'Detections: {total_detections} -> {valid_detections}',
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if frame_data.rotation_angle != 0:
                    cv2.putText(combined_visualization, f'Rotation: {frame_data.rotation_angle}°', (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
                # Crear visualización completa
                visualization = create_opencv_visualization(
                    frame_data.processed_frame, 
                    frame_data.color_mask, 
                    combined_visualization
                )
                
                frame_data.visualization = visualization
                frame_data.processing_times['visualization'] = time.time() - start_time
                
                # Enviar a cola de salida
                try:
                    self.output_queue.put_nowait(frame_data)
                    self.stats['frames_processed'] += 1
                except queue.Full:
                    self.stats['frames_dropped'] += 1
                    print("⚠️ Frame dropped en output queue")
                
                # Actualizar estadísticas
                self.stats['stage_times']['visualization'].append(frame_data.processing_times['visualization'])
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error en visualization worker: {e}")
        
        print("🛑 Visualization worker terminado")
    
    def add_frame(self, frame, frame_id, rotation_angle=0):
        """Añadir frame al pipeline"""
        frame_data = FrameData(
            frame_id=frame_id,
            timestamp=time.time(),
            original_frame=frame.copy(),
            rotation_angle=rotation_angle
        )
        
        try:
            self.input_queue.put_nowait(frame_data)
            return True
        except queue.Full:
            self.stats['frames_dropped'] += 1
            return False
    
    def get_result(self, timeout=0.1):
        """Obtener resultado procesado"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self):
        """Obtener estadísticas del pipeline"""
        # Actualizar tamaños de colas
        self.stats['queue_sizes'] = {
            'input': self.input_queue.qsize(),
            'semantic': self.semantic_queue.qsize(),
            'detection': self.detection_queue.qsize(),
            'classification': self.classification_queue.qsize(),
            'output': self.output_queue.qsize()
        }
        
        # Calcular tiempos promedio
        avg_times = {}
        for stage, times in self.stats['stage_times'].items():
            avg_times[stage] = sum(times) / len(times) if times else 0
        
        return {
            'frames_processed': self.stats['frames_processed'],
            'frames_dropped': self.stats['frames_dropped'],
            'queue_sizes': self.stats['queue_sizes'],
            'avg_times': avg_times,
            'total_latency': sum(avg_times.values())
        }
    
    def pause(self):
        """Pausar pipeline"""
        self.paused = True
        print("⏸️ Pipeline pausado")
    
    def resume(self):
        """Reanudar pipeline"""
        self.paused = False
        print("▶️ Pipeline reanudado")
    
    def toggle_pause(self):
        """Alternar pausa"""
        if self.paused:
            self.resume()
        else:
            self.pause()
    
    def shutdown(self):
        """Cerrar pipeline de forma segura"""
        print("🔄 Cerrando pipeline asíncrono...")
        self.running = False
        
        # Enviar señales de parada a todas las colas
        for _ in range(self.max_workers):
            try:
                self.input_queue.put_nowait(None)
                self.semantic_queue.put_nowait(None)
                self.detection_queue.put_nowait(None)
                self.classification_queue.put_nowait(None)
            except queue.Full:
                pass
        
        # Cerrar thread pools
        self.semantic_executor.shutdown(wait=True)
        self.detection_executor.shutdown(wait=True)
        self.classification_executor.shutdown(wait=True)
        self.visualization_executor.shutdown(wait=True)
        
        print("✅ Pipeline cerrado correctamente")

def create_opencv_visualization(frame_rgb, color_mask, combined_viz):
    """Visualización con OpenCV en lugar de matplotlib"""
    # Redimensionar para visualización
    h, w = frame_rgb.shape[:2]
    display_h, display_w = 480, 640  # Tamaño fijo para visualización
    
    frame_display = cv2.resize(frame_rgb, (display_w, display_h))
    mask_display = cv2.resize(color_mask, (display_w, display_h))
    combined_display = cv2.resize(combined_viz, (display_w, display_h))
    
    # Concatenar horizontalmente
    combined_display_bgr = cv2.cvtColor(combined_display, cv2.COLOR_RGB2BGR)
    mask_display_bgr = cv2.cvtColor(mask_display, cv2.COLOR_RGB2BGR)
    frame_display_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
    
    visualization = np.hstack([frame_display_bgr, mask_display_bgr, combined_display_bgr])
    
    # Añadir títulos
    cv2.putText(visualization, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(visualization, 'Semantic', (display_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(visualization, 'Detection', (display_w*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return visualization

def get_video_rotation(video_path):
    """Detecta la rotación del video usando ffprobe o metadatos"""
    try:
        import subprocess
        import json
        
        # Intentar usar ffprobe para obtener metadatos de rotación
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    # Buscar tag de rotación
                    tags = stream.get('tags', {})
                    rotation = tags.get('rotate', '0')
                    return int(rotation)
    except:
        pass
    
    return 0

def correct_frame_rotation(frame, rotation_angle):
    """Corrige la rotación del frame basado en el ángulo detectado"""
    if rotation_angle == 0:
        return frame
    elif rotation_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        # Para ángulos arbitrarios
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
        return cv2.warpAffine(frame, rotation_matrix, (width, height))

def detect_video_orientation(cap):
    """Detecta si el video es vertical analizando las primeras frames"""
    # Leer algunos frames para determinar orientación
    frame_count = 0
    vertical_count = 0
    horizontal_count = 0
    
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    for _ in range(min(10, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width = frame.shape[:2]
        if height > width:
            vertical_count += 1
        else:
            horizontal_count += 1
        frame_count += 1
    
    # Volver al inicio del video
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
    
    is_vertical = vertical_count > horizontal_count
    print(f"Video detectado como: {'VERTICAL' if is_vertical else 'HORIZONTAL'}")
    print(f"Frames analizados: {frame_count}, Vertical: {vertical_count}, Horizontal: {horizontal_count}")
    
    return is_vertical

def clear_memory():
    """Libera memoria GPU si está disponible"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        gc.collect()

def load_model(opts, device):
    # Crear modelo con 114 clases (número correcto para Mapillary)
    opts.num_classes = 114
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes,
                                                output_stride=opts.output_stride)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        try:
            checkpoint = torch.load(opts.ckpt, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            print("✅ Checkpoint cargado exitosamente")
        except Exception as e:
            print(f"❌ Error al cargar checkpoint: {e}")
            raise

    model = model.to(device)
    return model, opts

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--input", type=str, default='0',
                        help="path to input video file or camera index (0 for webcam)")
    parser.add_argument("--input_type", type=str, default='camera',
                        choices=['video', 'camera'], help='Type of input source')
    parser.add_argument("--dataset", type=str, default='mapillary',
                        choices=['voc', 'cityscapes', 'mapillary'], help='Name of training set')
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--mapillary_version", type=str, default='v2.0',
                        choices=['v1.2', 'v2.0'], help='Mapillary version')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and 
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")

    # Video Rotation Options
    parser.add_argument("--auto_rotate", action='store_true', default=True,
                        help="Automatically detect and correct video rotation")
    parser.add_argument("--force_rotation", type=int, default=None,
                        choices=[0, 90, 180, 270], help="Force specific rotation angle")

    # Output Options
    parser.add_argument("--save_video", action='store_true', default=False,
                        help="save output video")
    parser.add_argument("--output_dir", type=str, default='results',
                        help="directory to save results")
    
    # Pipeline Options
    parser.add_argument("--use_async_pipeline", action='store_true', default=True,
                        help="Use asynchronous pipeline for better performance")
    parser.add_argument("--pipeline_workers", type=int, default=3,
                        help="Number of workers for async pipeline")
    parser.add_argument("--pipeline_buffer", type=int, default=8,
                        help="Buffer size for async pipeline")

    return parser

def create_color_mask_optimized(pred):
    """Versión ultra-optimizada con vectorización completa"""
    height, width = pred.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Crear array de colores indexado
    color_array = np.array([COLOR_MAP.get(i, (0, 0, 0)) for i in range(256)], dtype=np.uint8)
    
    # Aplicar colores de una vez usando indexación avanzada
    color_mask = color_array[pred]
    
    return color_mask

def main():
    opts = get_argparser().parse_args()
    frame_count = 0
    rotation_angle = 0
    
    print("🚀 === Sistema con Pipeline Asíncrono ===")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available, using CPU")

    # Configurar dispositivo
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Preasignar memoria GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
    
    # Cargar modelos
    try:
        print("📥 Cargando modelos...")
        yolo_model = YOLO('yolov8n-seg.pt', verbose=False)
        classifier_model = YOLO(r'C:\\Users\\aaron\\Documents\\Año4\\TFG\\Classifier\\runs\\classify\\Mejores_matrix\\weights\\best.pt', verbose=False)
        yolo_model.to(device).half()
        classifier_model.to(device).half()
        print("✅ Modelos YOLO cargados correctamente")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # Cargar modelo DeepLab
    print("📥 Cargando modelo DeepLab...")
    model, opts = load_model(opts, device)
    model.eval()
    print("✅ Modelo DeepLab cargado correctamente")

    # Configurar transformaciones
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    ])

    # Configurar captura de video
    if opts.input_type == 'camera':
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RESOLUTION[1])
            print("✅ Cámara iniciada correctamente")
        except Exception as e:
            print(f"❌ Error al abrir la cámara: {e}")
            return
    else:
        try:
            cap = cv2.VideoCapture(opts.input)
            if not cap.isOpened():
                print("❌ Error: No se pudo abrir el video")
                return
            print("✅ Video cargado correctamente")
            
            # Detectar rotación del video
            if opts.force_rotation is not None:
                rotation_angle = opts.force_rotation
                print(f"🔄 Rotación forzada: {rotation_angle}°")
            elif opts.auto_rotate:
                video_rotation = get_video_rotation(opts.input)
                if video_rotation != 0:
                    rotation_angle = video_rotation
                    print(f"🔄 Rotación detectada por metadatos: {rotation_angle}°")
                else:
                    is_vertical = detect_video_orientation(cap)
                    if is_vertical:
                        rotation_angle = 0
                        print("📱 Video vertical detectado - manteniendo orientación original")
                    else:
                        rotation_angle = 0
                        print("🖥️ Video horizontal detectado")
            
        except Exception as e:
            print(f"❌ Error al abrir el video: {e}")
            return

    # Obtener dimensiones del frame
    ret, test_frame = cap.read()
    if ret:
        if rotation_angle != 0:
            test_frame = correct_frame_rotation(test_frame, rotation_angle)
        frame_height, frame_width = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📐 Dimensiones del frame: {frame_width}x{frame_height}")

    # Configurar FPS
    fps = 30 if opts.input_type == 'camera' else int(cap.get(cv2.CAP_PROP_FPS))
    print(f"🎬 FPS configurado: {fps}")

    # Configurar writer para guardar video
    if opts.save_video:
        os.makedirs(opts.output_dir, exist_ok=True)
        output_path = os.path.join(opts.output_dir,
                                 f'output_async_{opts.input_type}_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width*3, frame_height))
        print(f"💾 Video se guardará en: {output_path}")

    # Inicializar procesador semántico
    semantic_processor = SemanticValidationPostProcessor(
        min_semantic_consistency=0.0
    )
    
    # 🚀 INICIALIZAR PIPELINE ASÍNCRONO
    if opts.use_async_pipeline:
        pipeline = AsyncPipeline(
            model=model,
            yolo_model=yolo_model,
            classifier_model=classifier_model,
            device=device,
            transform=transform,
            semantic_processor=semantic_processor,
            max_workers=opts.pipeline_workers,
            buffer_size=opts.pipeline_buffer
        )
        print(f"🚀 Pipeline asíncrono iniciado con {opts.pipeline_workers} workers")
    
    # Variables para estadísticas
    detection_stats = {
        'total_detections': 0,
        'valid_detections': 0,
        'filtered_detections': 0
    }
    
    fps_history = deque(maxlen=30)
    processing_times = deque(maxlen=30)
    
    # Configurar ventana de visualización
    cv2.namedWindow('Async Video Processing', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Async Video Processing', 1440, 360)
    
    print("🚀 Iniciando procesamiento con pipeline asíncrono...")
    
    try:
        frame_start_time = time.time()
        
        while cap.isOpened():
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("📹 Fin del video o error de lectura")
                break

            frame_count += 1
            
            # Procesar solo cada N frames
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                continue

            # 🚀 AÑADIR FRAME AL PIPELINE ASÍNCRONO
            if opts.use_async_pipeline:
                success = pipeline.add_frame(frame, frame_count, rotation_angle)
                if not success:
                    print("⚠️ Frame descartado - pipeline lleno")
                
                # 🎯 OBTENER RESULTADO PROCESADO
                result = pipeline.get_result(timeout=0.01)  # Non-blocking
                
                if result is not None:
                    # Mostrar visualización
                    if result.visualization is not None:
                        # Añadir estadísticas del pipeline
                        pipeline_stats = pipeline.get_stats()
                        
                        # Información adicional en la visualización
                        y_offset = result.visualization.shape[0] - 160
                        info_color = (255, 255, 255)
                        
                        cv2.putText(result.visualization, f'Frame: {result.frame_id}', (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
                        cv2.putText(result.visualization, f'Processed: {pipeline_stats["frames_processed"]}', (10, y_offset + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(result.visualization, f'Dropped: {pipeline_stats["frames_dropped"]}', (10, y_offset + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(result.visualization, f'Latency: {pipeline_stats["total_latency"]*1000:.1f}ms', (10, y_offset + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Mostrar tamaños de colas
                        queue_info = f"Q: {pipeline_stats['queue_sizes']['input']}-{pipeline_stats['queue_sizes']['semantic']}-{pipeline_stats['queue_sizes']['detection']}-{pipeline_stats['queue_sizes']['classification']}-{pipeline_stats['queue_sizes']['output']}"
                        cv2.putText(result.visualization, queue_info, (10, y_offset + 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                        
                        # Mostrar tiempos promedio por etapa
                        avg_times = pipeline_stats['avg_times']
                        times_info = f"Sem:{avg_times['semantic']*1000:.0f} Det:{avg_times['detection']*1000:.0f} Cls:{avg_times['classification']*1000:.0f} Viz:{avg_times['visualization']*1000:.0f}ms"
                        cv2.putText(result.visualization, times_info, (10, y_offset + 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        
                        cv2.imshow('Async Video Processing', result.visualization)
                        
                        # Guardar video si está habilitado
                        if 'out' in locals() and opts.save_video:
                            out.write(result.visualization)
                        
                        # Actualizar estadísticas
                        detection_stats['valid_detections'] += len(result.validated_detections)

            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Saliendo del sistema...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if 'result' in locals() and result is not None and result.visualization is not None:
                    filename = f'capture_async_{timestamp}.jpg'
                    cv2.imwrite(filename, result.visualization)
                    print(f"📸 Captura guardada: {filename}")
            elif key == ord('p'):
                if opts.use_async_pipeline:
                    pipeline.toggle_pause()
            elif key == ord('r'):
                rotation_angle = (rotation_angle + 90) % 360
                print(f"🔄 Rotación cambiada a: {rotation_angle}°")
            elif key == ord('f'):
                cv2.setWindowProperty('Async Video Processing', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord('n'):
                cv2.setWindowProperty('Async Video Processing', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

            # Limpiar memoria menos frecuentemente
            if frame_count % 300 == 0:
                clear_memory()

            # Mostrar estadísticas cada 100 frames
            if frame_count % 100 == 0:
                if opts.use_async_pipeline:
                    pipeline_stats = pipeline.get_stats()
                    print(f"📊 Frame {frame_count}: "
                          f"Processed: {pipeline_stats['frames_processed']}, "
                          f"Dropped: {pipeline_stats['frames_dropped']}, "
                          f"Latency: {pipeline_stats['total_latency']*1000:.1f}ms")

    except KeyboardInterrupt:
        print("🛑 Interrupción por teclado (Ctrl+C)")
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔄 Cerrando sistema...")
        
        # Cerrar pipeline asíncrono
        if opts.use_async_pipeline:
            pipeline.shutdown()
        
        # Mostrar estadísticas finales
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS FINALES DEL PIPELINE ASÍNCRONO")
        print("="*60)
        print(f"📹 Total frames leídos: {frame_count}")
        
        if opts.use_async_pipeline:
            final_stats = pipeline.get_stats()
            print(f"🎯 Frames procesados: {final_stats['frames_processed']}")
            print(f"❌ Frames descartados: {final_stats['frames_dropped']}")
            print(f"⚡ Latencia promedio: {final_stats['total_latency']*1000:.1f}ms")
            
            print(f"\n🔧 Tiempos promedio por etapa:")
            for stage, time_ms in final_stats['avg_times'].items():
                print(f"   - {stage.capitalize()}: {time_ms*1000:.1f}ms")
            
            efficiency = (final_stats['frames_processed'] / max(1, frame_count)) * 100
            print(f"\n📈 Eficiencia del pipeline: {efficiency:.1f}%")
        
        print(f"🎯 Total detecciones válidas: {detection_stats['valid_detections']}")
        print(f"🔄 Rotación final aplicada: {rotation_angle}°")
        print("="*60)

        # Liberar recursos
        cap.release()
        if 'out' in locals() and opts.save_video:
            out.release()
            print(f"💾 Video guardado correctamente")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("🎮 CONTROLES DEL SISTEMA ASÍNCRONO:")
    print("  'q' - Salir del sistema")
    print("  's' - Capturar frame actual")
    print("  'p' - Pausar/Reanudar pipeline")
    print("  'r' - Cambiar rotación manualmente (+90°)")
    print("  'f' - Alternar pantalla completa")
    print("  'n' - Volver a ventana normal")
    print("\n🚀 Sistema con Pipeline Asíncrono de Alto Rendimiento")
    print("="*60)
    main()