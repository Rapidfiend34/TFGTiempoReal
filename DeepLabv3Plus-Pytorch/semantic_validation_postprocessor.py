import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN

class SemanticValidationPostProcessor:
    """
    Clase para validar predicciones de instanciación usando segmentación semántica
    """
    
    def __init__(self, min_semantic_consistency=0.5):
        self.min_semantic_consistency = min_semantic_consistency
        
        # Mapeo de clases YOLO a clases semánticas de Mapillary
        self.class_mapping = {
            0: [4],      # Person -> human--person--individual
            1: [9],      # Bicycle -> object--vehicle--bicycle  
            2: [11],     # Car -> object--vehicle--car
            3: [12]      # Motorcycle -> object--vehicle--motorcycle
        }
    
    def validate_detection_with_semantics(self, bbox, yolo_class, semantic_mask):
        """
        Valida una detección comparando con la máscara semántica
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            yolo_class: clase detectada por YOLO
            semantic_mask: máscara de segmentación semántica
            
        Returns:
            dict: resultado de validación con score y detalles
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extraer región de la bbox en la máscara semántica
        roi_semantic = semantic_mask[y1:y2, x1:x2]
        
        if roi_semantic.size == 0:
            return {"valid": False, "score": 0.0, "reason": "empty_roi"}
        
        # Obtener clases semánticas esperadas para esta clase YOLO
        expected_semantic_classes = self.class_mapping.get(yolo_class, [])
        
        if not expected_semantic_classes:
            return {"valid": True, "score": 0.5, "reason": "no_mapping"}
        
        # Calcular consistencia semántica
        total_pixels = roi_semantic.size
        consistent_pixels = 0
        
        for semantic_class in expected_semantic_classes:
            consistent_pixels += np.sum(roi_semantic == semantic_class)
        
        consistency_ratio = consistent_pixels / total_pixels
        
        # Validación adicional: verificar que no hay demasiados píxeles de clases incompatibles
        incompatible_classes = [2, 3]  # road, sidewalk
        incompatible_pixels = 0
        for incompatible_class in incompatible_classes:
            incompatible_pixels += np.sum(roi_semantic == incompatible_class)
        
        incompatible_ratio = incompatible_pixels / total_pixels
        
        # Calcular score final
        final_score = consistency_ratio - (incompatible_ratio * 0.5)
        final_score = max(0.0, min(1.0, final_score))
        
        is_valid = (consistency_ratio >= self.min_semantic_consistency and 
                   incompatible_ratio < 0.7)
        
        return {
            "valid": is_valid,
            "score": final_score,
            "consistency_ratio": consistency_ratio,
            "incompatible_ratio": incompatible_ratio,
            "reason": "validated" if is_valid else "low_consistency"
        }
    
    def refine_bbox_with_semantics(self, bbox, yolo_class, semantic_mask):
        """
        Refina la bounding box usando información semántica
        """
        x1, y1, x2, y2 = map(int, bbox)
        expected_classes = self.class_mapping.get(yolo_class, [])
        
        if not expected_classes:
            return bbox
        
        # Crear máscara binaria de píxeles relevantes
        relevant_mask = np.zeros_like(semantic_mask, dtype=bool)
        for semantic_class in expected_classes:
            relevant_mask |= (semantic_mask == semantic_class)
        
        # Encontrar región conectada más grande en la bbox
        roi_mask = relevant_mask[y1:y2, x1:x2]
        
        if not np.any(roi_mask):
            return bbox
        
        # Encontrar componentes conectados
        labeled, num_features = ndimage.label(roi_mask)
        
        if num_features == 0:
            return bbox
        
        # Encontrar el componente más grande
        component_sizes = ndimage.sum(roi_mask, labeled, range(1, num_features + 1))
        largest_component = np.argmax(component_sizes) + 1
        
        # Obtener bbox del componente más grande
        component_mask = (labeled == largest_component)
        coords = np.where(component_mask)
        
        if len(coords[0]) == 0:
            return bbox
        
        # Calcular nueva bbox (relativa a la ROI original)
        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        
        # Convertir a coordenadas absolutas
        new_x1 = x1 + min_x
        new_y1 = y1 + min_y
        new_x2 = x1 + max_x
        new_y2 = y1 + max_y
        
        # Añadir pequeño margen
        margin = 5
        new_x1 = max(0, new_x1 - margin)
        new_y1 = max(0, new_y1 - margin)
        new_x2 = min(semantic_mask.shape[1], new_x2 + margin)
        new_y2 = min(semantic_mask.shape[0], new_y2 + margin)
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def filter_overlapping_detections(self, detections_with_validation):
        """
        Filtra detecciones superpuestas priorizando las mejor validadas
        """
        if len(detections_with_validation) <= 1:
            return detections_with_validation
        
        # Ordenar por score de validación
        sorted_detections = sorted(detections_with_validation, 
                                 key=lambda x: x['validation']['score'], reverse=True)
        
        filtered = []
        
        for detection in sorted_detections:
            bbox = detection['bbox']
            should_keep = True
            
            for kept_detection in filtered:
                kept_bbox = kept_detection['bbox']
                
                # Calcular IoU
                iou = self.calculate_iou(bbox, kept_bbox)
                
                # Si hay mucha superposición, mantener solo la mejor validada
                if iou > 0.5:
                    should_keep = False
                    break
            
            if should_keep:
                filtered.append(detection)
        
        return filtered
    
    def calculate_iou(self, bbox1, bbox2):
        """Calcula Intersection over Union entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calcular intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular unión
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

def apply_semantic_validation(results, semantic_mask, processor):
    """
    Aplica validación semántica a los resultados de YOLO
    
    Args:
        results: resultados de YOLO
        semantic_mask: máscara de segmentación semántica
        processor: instancia de SemanticValidationPostProcessor
        
    Returns:
        list: detecciones validadas y filtradas
    """
    if results[0].boxes.id is None:
        return []
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    
    validated_detections = []
    
    for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
        # Validar con semántica
        validation_result = processor.validate_detection_with_semantics(
            box, cls, semantic_mask
        )
        
        # Refinar bbox si la validación es exitosa
        if validation_result['valid']:
            refined_bbox = processor.refine_bbox_with_semantics(
                box, cls, semantic_mask
            )
        else:
            refined_bbox = box
        
        detection = {
            'bbox': refined_bbox,
            'original_bbox': box,
            'track_id': track_id,
            'class': cls,
            'confidence': conf,
            'validation': validation_result
        }
        
        validated_detections.append(detection)
    
    # Filtrar detecciones superpuestas
    filtered_detections = processor.filter_overlapping_detections(validated_detections)
    
    # Filtrar solo las válidas
    valid_detections = [d for d in filtered_detections if d['validation']['valid']]
    
    return valid_detections

# Función para visualizar resultados validados
def draw_validated_boxes(image, validated_detections, track_classifications):
    """
    Dibuja las bounding boxes validadas con información adicional
    """
    img_with_boxes = image.copy()
    
    class_colors = {
        0: (255, 0, 0),    # Persona - Rojo
        1: (0, 255, 0),    # Bicicleta - Verde
        2: (0, 0, 255),    # Coche - Azul
        3: (255, 255, 0)   # Moto - Amarillo
    }
    
    class_names = {
        0: 'Person',
        1: 'Bicycle', 
        2: 'Car',
        3: 'Motorcycle'
    }
    
    for detection in validated_detections:
        bbox = detection['bbox']
        track_id = detection['track_id']
        cls = detection['class']
        conf = detection['confidence']
        validation = detection['validation']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color base según clase
        base_color = class_colors.get(cls, (0, 255, 0))
        
        # Modificar color según score de validación
        validation_score = validation['score']
        if validation_score > 0.8:
            color = base_color  # Verde brillante para alta validación
        elif validation_score > 0.5:
            color = tuple(int(c * 0.8) for c in base_color)  # Color atenuado
        else:
            color = (128, 128, 128)  # Gris para baja validación
        
        # Dibujar rectángulo principal
        thickness = 3 if validation_score > 0.7 else 2
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # Dibujar bbox original si fue refinada
        original_bbox = detection['original_bbox']
        if not np.array_equal(bbox, original_bbox):
            ox1, oy1, ox2, oy2 = map(int, original_bbox)
            cv2.rectangle(img_with_boxes, (ox1, oy1), (ox2, oy2), (128, 128, 128), 1)
        
        # Preparar etiqueta
        class_name = class_names.get(cls, str(cls))
        label_parts = [f'ID:{track_id}', f'{conf:.2f}']
        
        # Añadir información de clasificación si existe
        if track_id in track_classifications:
            classification = track_classifications[track_id]
            label_parts.append(f"{classification['class']} ({classification['conf']:.2f})")
        
        # Añadir score de validación
        label_parts.append(f'V:{validation_score:.2f}')
        
        label = ' '.join(label_parts)
        
        # Dibujar etiqueta con fondo
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_with_boxes, (x1, y1-h-10), (x1+w, y1), color, -1)
        cv2.putText(img_with_boxes, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Añadir indicador de validación en esquina
        validation_color = (0, 255, 0) if validation['valid'] else (0, 0, 255)
        cv2.circle(img_with_boxes, (x2-10, y1+10), 5, validation_color, -1)
    
    return img_with_boxes