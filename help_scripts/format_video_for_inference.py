import os
import glob
import cv2


def process_video(input_path, output_path, size=(640, 640)):
    """
    Cambia la resolución de un video completo y lo guarda en la carpeta de salida.

    :param input_path: Ruta del video de entrada.
    :param output_path: Ruta donde se guardará el video procesado.
    :param size: Tamaño para redimensionar (ancho, alto).
    """
    cap = cv2.VideoCapture(input_path)

    # Obtener propiedades del video original
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec de salida
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = size

    # Crear el VideoWriter para guardar el video procesado
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el frame
        frame_resized = cv2.resize(frame, size)
        out.write(frame_resized)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Video procesado: {output_path} ({frame_count} frames)")


def process_videos_in_folder(folder_path, size=(640, 640)):
    """
    Procesa todos los videos en una carpeta, cambiando su resolución a 640x640.

    :param folder_path: Ruta a la carpeta que contiene los videos.
    :param size: Tupla (ancho, alto) para redimensionar los videos.
    """
    videos = glob.glob(os.path.join(folder_path, "*.mp4"))

    output_folder_videos = os.path.join(folder_path, 'processed_videos')
    os.makedirs(output_folder_videos, exist_ok=True)

    # Procesar cada video
    for vid in videos:
        output_video_path = os.path.join(output_folder_videos, os.path.basename(vid))
        process_video(vid, output_video_path, size=size)

    print(f"Total videos procesados: {len(videos)}")


# Ejemplo de uso
media_folder = '../../../Descargas/'  # Cambia esta ruta a la carpeta que contiene los videos
process_videos_in_folder(media_folder, size=(640, 640))
