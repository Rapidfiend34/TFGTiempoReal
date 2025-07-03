import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO
import network
from collections import OrderedDict
import os
from PIL import Image
import torchvision.transforms as T
import cv2

class ModelConverter:
    """Clase para convertir modelos PyTorch a ONNX - VERSIÓN CORREGIDA"""
    
    def __init__(self, device='cpu'):  # Por defecto CPU
        self.device = device
        
    def convert_yolo_to_onnx(self, model_path, output_path, img_size=640):
        """
        Convierte modelo YOLO a ONNX
        """
        print(f"Convirtiendo YOLO: {model_path} -> {output_path}")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Cargar modelo YOLO
        model = YOLO(model_path)
        
        # Exportar a ONNX
        model.export(
            format='onnx',
            imgsz=img_size,
            optimize=True,
            half=False,  # Usar FP32 para mejor compatibilidad
            dynamic=True,  # Permitir tamaños dinámicos
            simplify=True,
            opset=11
        )
        
        # El archivo se guarda automáticamente con extensión .onnx
        generated_path = model_path.replace('.pt', '.onnx')
        if os.path.exists(generated_path) and generated_path != output_path:
            os.rename(generated_path, output_path)
            
        print(f"YOLO convertido exitosamente: {output_path}")
        return output_path
    
    def convert_deeplab_to_onnx(self, model, output_path, input_size=(1, 3, 192, 192)):
        """
        Convierte modelo DeepLab a ONNX - VERSIÓN CORREGIDA
        """
        print(f"Convirtiendo DeepLab -> {output_path}")
        
        # Asegurar que el modelo esté en CPU para conversión ONNX
        model = model.cpu()
        model.eval()
        
        # Crear tensor de entrada dummy en CPU
        dummy_input = torch.randn(input_size)  # Sin .to(device)
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Exportar a ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                },
                verbose=False
            )
            
            print(f"DeepLab convertido exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error en conversión ONNX: {e}")
            raise

class ONNXInferenceEngine:
    """Motor de inferencia optimizado para modelos ONNX"""
    
    def __init__(self):
        self.providers = self._get_available_providers()
        self.sessions = {}
        
    def _get_available_providers(self):
        """Obtiene los proveedores disponibles en orden de preferencia"""
        available = ort.get_available_providers()
        preferred_order = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        providers = []
        for provider in preferred_order:
            if provider in available:
                providers.append(provider)
                
        print(f"Proveedores ONNX disponibles: {providers}")
        return providers
    
    def load_model(self, model_path, model_name):
        """Carga un modelo ONNX"""
        print(f"Cargando modelo ONNX: {model_name} desde {model_path}")
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.sessions[model_name] = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=self.providers
        )
        
        # Mostrar información del modelo
        input_info = self.sessions[model_name].get_inputs()[0]
        output_info = self.sessions[model_name].get_outputs()[0]
        
        print(f"  Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        print(f"  Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
        
    def predict_yolo(self, model_name, image, conf_threshold=0.25):
        """Inferencia YOLO con ONNX"""
        session = self.sessions[model_name]
        
        # Preprocesar imagen
        if isinstance(image, np.ndarray):
            # Convertir BGR a RGB si es necesario
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensionar y normalizar
            input_size = session.get_inputs()[0].shape[2:]  # [height, width]
            image_resized = cv2.resize(image, (input_size[1], input_size[0]))
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Cambiar formato de HWC a CHW
            image_transposed = np.transpose(image_normalized, (2, 0, 1))
            
            # Añadir dimensión batch
            input_tensor = np.expand_dims(image_transposed, axis=0)
        
        # Ejecutar inferencia
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        
        return outputs
    
    def predict_deeplab(self, model_name, image_tensor):
        """Inferencia DeepLab con ONNX"""
        session = self.sessions[model_name]
        
        # Convertir tensor PyTorch a numpy si es necesario
        if torch.is_tensor(image_tensor):
            input_array = image_tensor.cpu().numpy()
        else:
            input_array = image_tensor
            
        # Ejecutar inferencia
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_array})
        
        return outputs[0]  # Retornar solo el primer output
    
    def predict_classifier(self, model_name, image):
        """Inferencia clasificador con ONNX"""
        session = self.sessions[model_name]
        
        # Preprocesar imagen para clasificación
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Redimensionar a tamaño esperado
        input_size = session.get_inputs()[0].shape[2:]
        image_resized = cv2.resize(image, (input_size[1], input_size[0]))
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Normalización estándar ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # Cambiar formato y añadir batch
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(image_transposed, axis=0)
        
        # Ejecutar inferencia
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        
        return outputs[0]  # Probabilidades de clase

def convert_all_models():
    """Script principal para convertir todos los modelos"""
    
    converter = ModelConverter(device='cpu')  # Forzar CPU
    
    # Rutas de modelos (ajustar según tu configuración)
    models_to_convert = {
        'yolo_detection': {
            'input': 'yolov8n-seg.pt',
            'output': 'models_onnx/yolo_detection.onnx',
            'img_size': 640
        },
        'yolo_classifier': {
            'input': r'C:\\Users\\aaron\\Documents\\Año4\\TFG\\Classifier\\runs\\classify\\Mejores_matrix\\weights\\best.pt',
            'output': 'models_onnx/yolo_classifier.onnx',
            'img_size': 224
        }
    }
    
    # Crear directorio para modelos ONNX
    os.makedirs('models_onnx', exist_ok=True)
    
    # Convertir modelos YOLO
    for model_name, config in models_to_convert.items():
        if os.path.exists(config['input']):
            try:
                converter.convert_yolo_to_onnx(
                    config['input'],
                    config['output'],
                    config['img_size']
                )
            except Exception as e:
                print(f"Error convirtiendo {model_name}: {e}")
        else:
            print(f"Modelo no encontrado: {config['input']}")
    
    print("\\nConversión completada!")
    print("Para convertir DeepLab, ejecuta convert_deeplab_model() después de cargar el modelo.")

def convert_deeplab_model(model, output_path='models_onnx/deeplab.onnx'):
    """Convierte modelo DeepLab específico - VERSIÓN CORREGIDA"""
    converter = ModelConverter(device='cpu')  # Forzar CPU para conversión
    converter.convert_deeplab_to_onnx(model, output_path)

if __name__ == "__main__":
    convert_all_models()