import cv2
from ultralytics import YOLO
import os

# Variables globales para los tiempos
time_capturing_frame = 0
time_inference_outside_function = 0
time_memory_update = 0
time_write_frame = 0


# Definir funciones


def actualizar_memoria(track_id, clase_detectada, memory):
    """Actualizar la memoria con el estado de los objetos detectados."""
    global time_memory_update
    t1 = cv2.getTickCount()
    if track_id not in memory:
        memory[track_id] = {'defectuosa': clase_detectada.endswith('-d'), 'frames_visibles': 30}
    else:
        # Marca como defectuosa de forma permanente si tiene el sufijo "-d"
        memory[track_id]['defectuosa'] |= clase_detectada.endswith('-d')
        memory[track_id]['frames_visibles'] = 30  # Reinicia el contador de frames

        # Si el objeto ya es defectuoso pero no tiene la clase defectuosa, cambiarla
        if memory[track_id]['defectuosa'] and not clase_detectada.endswith('-d'):
            # Aquí cambiamos la clase detectada a la versión defectuosa
            clase_detectada = clase_detectada + '-d'

    # Actualización de la memoria para el ID correspondiente
    memory[track_id]['clase'] = clase_detectada  # Almacena la clase actualizada (defectuosa o no)
    t2 = cv2.getTickCount()
    time_memory_update += (t2 - t1) / cv2.getTickFrequency()


def procesar_video(video_path, model, output_video_path, clases, memory):
    """Procesar el video, realizar la detección y tracking, y guardar el resultado."""
    global time_capturing_frame, time_inference_outside_function, time_write_frame

    cap = cv2.VideoCapture(video_path)  # O usa 0 para la cámara en tiempo real
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configurar el escritor de video para guardar el video procesado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Usar el códec mp4v
    out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

    frame_count = 0  # Contador de frames procesados
    total_preprocess_time = 0
    total_inference_time = 0
    total_postprocess_time = 0

    # Definir colores específicos para cada clase
    colores = {
        'negra': (0, 0, 255),  # Rojo
        'blanca': (0, 255, 0),  # Verde
        'verde': (255, 0, 0),  # Azul
        'azul': (255, 255, 0),  # Amarillo
        'negra-d': (0, 165, 255),  # Naranja
        'blanca-d': (255, 165, 0),  # Azul claro
        'verde-d': (255, 105, 180),  # Rosa
        'azul-d': (255, 0, 255),  # Magenta
    }

    while cap.isOpened():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        t2 = cv2.getTickCount()
        time_capturing_frame += (t2 - t1) / cv2.getTickFrequency()

        if not ret:
            break

        # Realizar inferencia
        t1 = cv2.getTickCount()
        results = model.track(
            source=frame,  # (str, optional) source directory for images or videos
            device=0,  # (int, optional) GPU id (0-9) or -1 for CPU
            persist=True,
            tracker='bytetrack.yaml',  # (str, optional) filename of tracker YAML
        )
        t2 = cv2.getTickCount()
        time_inference_outside_function += (t2 - t1) / cv2.getTickFrequency()

        if results[0].boxes.id is None:
            t1 = cv2.getTickCount()
            frame_count += 1
            out.write(frame)
            t2 = cv2.getTickCount()
            time_write_frame += (t2 - t1) / cv2.getTickFrequency()
            continue

        t1 = cv2.getTickCount()

        # Obtener coordenadas de las detecciones y IDs
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy().astype(float)

        total_preprocess_time += results[0].speed['preprocess']
        total_inference_time += results[0].speed['inference']
        total_postprocess_time += results[0].speed['postprocess']

        for box, obj_id, cls, conf in zip(boxes, ids, classes, confs):
            xmin, ymin, xmax, ymax = box
            clase_detectada = clases[cls]

            # Actualizar la memoria de acuerdo a si la clase es defectuosa
            actualizar_memoria(obj_id, clase_detectada, memory)
            clase_detectada = memory[obj_id]['clase']

            # Determinar el estado de la canica (defectuosa o no)
            color = colores.get(clase_detectada, (255, 255, 255))

            # Dibujar rectángulo y texto
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            texto = f'ID:{obj_id} {clase_detectada} {conf:.2f}'
            cv2.putText(
                frame, texto, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        out.write(frame)
        frame_count += 1
        t2 = cv2.getTickCount()
        time_write_frame += (t2 - t1) / cv2.getTickFrequency()

    cap.release()
    out.release()
    return frame_count, total_preprocess_time, total_inference_time, total_postprocess_time


def main():
    global time_capturing_frame, time_inference_outside_function, time_write_frame

    # Parámetros del modelo y archivo de salida
    model_path = '../models/canicas/2024_11_28/2024_11_28_canicas_yolo11n_FP16.engine'
    video_path = '../datasets_labeled/videos/prueba_tiempo_tracking.mp4'
    output_dir = '../inference_predictions/custom_tracker'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f'video_con_tracking.mp4')

    # Cargar el modelo
    model = YOLO(model_path)
    clases = {
        0: 'negra',
        1: 'blanca',
        2: 'verde',
        3: 'azul',
        4: 'negra-d',
        5: 'blanca-d',
        6: 'verde-d',
        7: 'azul-d',
    }
    memory = {}

    total_time = cv2.getTickCount()
    frame_count, total_preprocess_time, total_inference_time, total_postprocess_time = (
        procesar_video(video_path, model, output_video_path, clases, memory)
    )
    total_time = (cv2.getTickCount() - total_time) / cv2.getTickFrequency()

    print(f'Frames procesados: {frame_count}')
    print(f'Tiempo total: {total_time:.3f} segundos')
    print(f'Tiempo captura: {time_capturing_frame:.3f} segundos')
    print(f'Tiempo inferencia: {time_inference_outside_function:.3f} segundos')
    print(
        f'Tiempo de preprocesamiento: {total_preprocess_time/1000:.3f} segundos, '
        f'inferencia: {total_inference_time/1000:.3f} segundos, '
        f'postprocesamiento: {total_postprocess_time/1000:.3f} segundos'
    )
    print(f'Tiempo escritura: {time_write_frame:.3f} segundos')


if __name__ == "__main__":
    main()
