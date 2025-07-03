import os
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from torchvision import transforms as T
from PIL import Image
import argparse
import time
from glob import glob

# ========== MAPILLARY CLASSES & COLOR MAP ==========
MAPILLARY_CLASSES = {
    'road': 2,
    'sidewalk': 3,
    'person': 4,
    'rider': 5,
    'bicycle': 9,
    'bus': 10,
    'car': 11,
    'motorcycle': 12,
    'truck': 13
}

COLOR_MAP = {
    2: (128, 64, 128),
    3: (244, 35, 232),
    4: (220, 20, 60),
    5: (255, 0, 200),
    9: (119, 11, 32),
    10: (0, 60, 100),
    11: (0, 0, 142),
    12: (0, 0, 230),
    13: (0, 0, 70),
    255: (0, 0, 0)
}

def create_color_mask(pred):
    height, width = pred.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id in MAPILLARY_CLASSES.values():
        mask = pred == class_id
        color = COLOR_MAP[class_id]
        color_mask[mask] = color
    return color_mask

def blend_images(original, semantic, instance, alpha_semantic=0.5, alpha_instance=0.5):
    original = np.array(original, dtype=np.float32) / 255.0
    semantic = np.array(semantic, dtype=np.float32) / 255.0
    instance = np.array(instance, dtype=np.float32) / 255.0
    h, w = original.shape[:2]
    semantic = cv2.resize(semantic, (w, h))
    instance = cv2.resize(instance, (w, h))
    blend = original.copy()
    mask_semantic = (semantic.sum(axis=2) > 0)
    blend[mask_semantic] = (1 - alpha_semantic) * original[mask_semantic] + alpha_semantic * semantic[mask_semantic]
    mask_instance = (instance.sum(axis=2) > 0)
    blend[mask_instance] = (1 - alpha_instance) * blend[mask_instance] + alpha_instance * instance[mask_instance]
    return (blend * 255).astype(np.uint8)

class FastTracker:
    def __init__(self,
                 det_thresh=0.3,
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.3,
                 delta_t=3,
                 track_history=30):
        from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
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

    def update(self, results, frame):
        self.frame_count += 1
        output_results = self._process_detections(results)
        height, width = frame.shape[:2]
        img_info = np.array([height, width])
        img_size = np.array([width, height])
        tracks = self.tracker.update(output_results, img_info, img_size)
        self._update_tracks_history(tracks)
        if self.frame_count % 30 == 0:
            self.fps = self.frame_count / (time.time() - self.time_start)
        return tracks

    def _process_detections(self, results):
        if not results or len(results) == 0:
            return np.empty((0, 5))
        dets = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == 0:  # Solo personas, cambia si quieres más clases
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    dets.append([x1, y1, x2, y2, conf])
        return np.array(dets) if dets else np.empty((0, 5))

    def _update_tracks_history(self, tracks):
        for track in tracks:
            track_id = int(track[4]) if len(track) > 4 else 0
            if track_id not in self.tracks_dict:
                self.tracks_dict[track_id] = deque(maxlen=self.track_history)
            self.tracks_dict[track_id].append(track[:4])

    def draw_tracks(self, frame, tracks):
        frame_with_tracks = frame.copy()
        cv2.putText(frame_with_tracks, f'FPS: {self.fps:.1f}',
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for track in tracks:
            if len(track) < 5:
                continue
            bbox = track[:4].astype(int)
            track_id = int(track[4])
            cv2.rectangle(frame_with_tracks,
                         (bbox[0], bbox[1]),
                         (bbox[2], bbox[3]),
                         (0, 255, 0), 2)
            cv2.putText(frame_with_tracks,
                       f'ID: {track_id}',
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (0, 255, 0),
                       2)
            if track_id in self.tracks_dict:
                history = self.tracks_dict[track_id]
                for i in range(1, len(history)):
                    if i > 1:
                        pt1 = (int((history[i-1][0] + history[i-1][2])/2),
                              int((history[i-1][1] + history[i-1][3])/2))
                        pt2 = (int((history[i][0] + history[i][2])/2),
                              int((history[i][1] + history[i][3])/2))
                        cv2.line(frame_with_tracks, pt1, pt2, (0, 255, 0), 1)
        return frame_with_tracks

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False,
                        help="path to a single image, image directory, video file, or leave empty for webcam")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes', 'mapillary'], help='Name of training set')
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--mapillary_version", type=str, default='v2.0',
                        choices=['v1.2', 'v2.0'], help='Mapillary version')
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--save_results", action='store_true', default=False,
                        help="save visualization results")
    parser.add_argument("--output_dir", type=str, default='results',
                        help="directory to save results")
    return parser

def load_model(opts, device):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import network
    from collections import OrderedDict
    opts.num_classes = 114 if opts.dataset == 'mapillary' else 19  # Ajusta según dataset
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes,
                                                  output_stride=opts.output_stride)
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Checkpoint cargado exitosamente")
    model = model.to(device)
    return model, opts

def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar modelo de segmentación
    model, opts = load_model(opts, device)
    model.eval()

    # Cargar modelo YOLOv8
    yolo_model = YOLO(r'C:\Users\aaron\Documents\Año4\TFG\CityScapesWithPanopticSegm\runs\segment\train83\weights\best.pt')  # Cambia por tu path si es necesario

    # Inicializar tracker
    tracker = FastTracker()

    # Transformación para segmentación
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Configurar entrada: webcam, vídeo o imágenes
    if opts.input is None:
        cap = cv2.VideoCapture(0)
        input_type = 'webcam'
    elif os.path.isfile(opts.input) and opts.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(opts.input)
        input_type = 'video'
    elif os.path.isdir(opts.input):
        image_files = []
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
        input_type = 'images'
    elif os.path.isfile(opts.input):
        image_files = [opts.input]
        input_type = 'images'
    else:
        print("No se reconoce el tipo de entrada.")
        return

    if opts.save_results:
        os.makedirs(opts.output_dir, exist_ok=True)

    if input_type in ['webcam', 'video']:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame.")
                break

            # Segmentación semántica
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.max(1)[1].cpu().numpy()[0]
            color_mask = create_color_mask(pred)

            # Detección de instancias (YOLOv8)
            results = yolo_model(frame,conf=0.05, verbose=False)

            # Tracking
            tracks = tracker.update(results, frame)

            # Visualización
            blend = blend_images(frame, color_mask, np.zeros_like(frame), alpha_semantic=0.4, alpha_instance=0.0)
            frame_with_tracks = tracker.draw_tracks(blend, tracks)

            cv2.imshow('Tracking + Semántica', frame_with_tracks)
            if opts.save_results:
                frame_name = f"frame_{tracker.frame_count:06d}.png"
                cv2.imwrite(os.path.join(opts.output_dir, frame_name), frame_with_tracks)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif input_type == 'images':
        for img_path in image_files:
            img_original = Image.open(img_path).convert('RGB')
            img_array = np.array(img_original)
            img_tensor = transform(img_original).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.max(1)[1].cpu().numpy()[0]
            color_mask = create_color_mask(pred)
            results = yolo_model(img_array, verbose=False)
            tracks = tracker.update(results, img_array)
            blend = blend_images(img_array, color_mask, np.zeros_like(img_array), alpha_semantic=0.4, alpha_instance=0.0)
            frame_with_tracks = tracker.draw_tracks(blend, tracks)
            cv2.imshow('Tracking + Semántica', frame_with_tracks)
            if opts.save_results:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                cv2.imwrite(os.path.join(opts.output_dir, f"{img_name}_tracked.png"), frame_with_tracks)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()