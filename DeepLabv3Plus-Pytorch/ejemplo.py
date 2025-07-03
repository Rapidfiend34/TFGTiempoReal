import pyzed.sl as sl
import cv2
import numpy as np
import os
from pathlib import Path

def convert_svo2_to_mp4(svo_path, output_name, fps=30):
    # Verificar que el archivo SVO existe
    if not os.path.exists(svo_path):
        print(f"❌ Error: El archivo SVO no existe en: {svo_path}")
        return False

    # Crear nombre de archivo de salida con extensión .mp4
    if not output_name.endswith('.mp4'):
        output_name += '.mp4'
    
    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(output_name) if os.path.dirname(output_name) else '.'
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔄 Convirtiendo SVO2 a MP4...")
    print(f"📂 Archivo entrada: {svo_path}")
    print(f"📂 Archivo salida: {output_name}")

    try:
        # Inicializar cámara ZED
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        init_params.svo_real_time_mode = False

        # Abrir SVO file
        zed = sl.Camera()
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Error opening SVO file: {status}")

        # Obtener información del video
        total_frames = zed.get_svo_number_of_frames()
        image = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        # Leer primer frame para obtener dimensiones
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            first_frame = image.get_data()
            height, width = first_frame.shape[:2]
        else:
            raise Exception("No se pudo leer el primer frame")

        # Configurar writer de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

        if not out.isOpened():
            raise Exception("No se pudo crear el archivo de salida")

        print(f"📊 Total frames: {total_frames}")
        print(f"📐 Resolución: {width}x{height}")
        print(f"🎬 FPS: {fps}")

        # Volver al inicio del video
        zed.set_svo_position(0)

        # Convertir frame por frame
        frame_count = 0
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Obtener frame
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()

                # Convertir de RGBA a BGR si es necesario
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                # Guardar frame
                out.write(frame)
                
                # Actualizar progreso
                frame_count += 1
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ Progreso: {progress:.1f}% ({frame_count}/{total_frames})")

            else:
                break

        # Liberar recursos
        out.release()
        zed.close()

        print(f"✅ Conversión completada!")
        print(f"📼 Video guardado en: {output_name}")
        return True

    except Exception as e:
        print(f"❌ Error durante la conversión: {e}")
        if 'out' in locals():
            out.release()
        if 'zed' in locals():
            zed.close()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convertir SVO2 a MP4')
    parser.add_argument('--input', type=str, 
                       default=r"c:\Users\aaron\Downloads\estadibalear_alante_1\estadibalear_alante_1.svo2", 
                       help='Ruta al archivo SVO2')
    parser.add_argument('--output', type=str, 
                       default="Video_Transformed/Video_Transformed.mp4", 
                       help='Nombre del archivo MP4 de salida')
    parser.add_argument('--fps', type=int, default=30, 
                       help='FPS para el video de salida')
    
    args = parser.parse_args()
    
    convert_svo2_to_mp4(args.input, args.output, args.fps)