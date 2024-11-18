import cv2
from ultralytics import YOLO
import os


# Definir funciones

def actualizar_memoria(track_id, clase_detectada, memory):
    """Actualizar la memoria con el estado de los objetos detectados."""
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


def procesar_video(video_path, model, output_video_path, clases, memory):
    """Procesar el video, realizar la detección y tracking, y guardar el resultado."""
    cap = cv2.VideoCapture(video_path)  # O usa 0 para la cámara en tiempo real
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configurar el escritor de video para guardar el video procesado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Usar el códec mp4v
    out = cv2.VideoWriter(output_video_path, fourcc, 20, (frame_width, frame_height))

    total_time = 0
    frame_count = 0  # Contador de frames procesados
    total_preprocess_time = 0
    total_inference_time = 0
    total_postprocess_time = 0

    # Definir colores específicos para cada clase
    colores = {
        'negra': (0, 0, 255),    # Rojo
        'blanca': (0, 255, 0),   # Verde
        'verde': (255, 0, 0),    # Azul
        'azul': (255, 255, 0),   # Amarillo
        'negra-d': (0, 165, 255),# Naranja
        'blanca-d': (255, 165, 0),# Azul claro
        'verde-d': (255, 105, 180), # Rosa
        'azul-d': (255, 0, 255), # Magenta
    }

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocesar la imagen
        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.track(
            source=frame,           # (str, optional) source directory for images or videos
            device=0,               # (int, optional) GPU id (0-9) or -1 for CPU
            persist=True,
            tracker='bytetrack.yaml',  # (str, optional) filename of tracker YAML
        )

        if results[0].boxes.id is None:
            frame_count += 1  # Contabilizar este frame
            # Escribir el frame aunque no haya detección
            out.write(frame)
            continue

        # Obtener coordenadas de las detecciones y IDs
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy().astype(float)
                
        total_preprocess_time += results[0].speed['preprocess']
        total_inference_time += results[0].speed['inference']
        total_postprocess_time += results[0].speed['postprocess']

        # Dentro del bucle for que itera sobre las detecciones:
        for box, obj_id, cls, conf in zip(boxes, ids, classes, confs):
            xmin, ymin, xmax, ymax = box
            clase_detectada = clases[cls]

            # Actualizar la memoria de acuerdo a si la clase es defectuosa
            actualizar_memoria(obj_id, clase_detectada, memory)
            
            clase_detectada = memory[obj_id]['clase']  # Actualizar la clase detectada
            cls = list(clases.values()).index(clase_detectada)  # Obtener el índice de la clase actualizada

            # Determinar el estado de la canica (defectuosa o no)
            color = colores.get(clase_detectada, (255, 255, 255))  # Color predeterminado blanco si no se encuentra

            # Dibujar rectángulo alrededor del objeto detectado
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Crear el texto a dibujar
            texto = f'ID:{obj_id} {clase_detectada} {conf:.2f}'

            # Calcular el tamaño del texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(texto, font, font_scale, thickness)

            # Coordenadas del rectángulo de fondo para el texto
            rect_x1 = xmin
            rect_y1 = ymin - 10 - text_height - baseline  # Parte superior del rectángulo
            rect_x2 = xmin + text_width
            rect_y2 = ymin - 10  # Parte inferior del rectángulo

            # Dibujar el rectángulo del color correspondiente
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)

            # Dibujar el texto en blanco sobre el rectángulo
            cv2.putText(frame, texto, (xmin, ymin - 10), font, font_scale, (255, 255, 255), thickness)


        # Escribir el frame procesado al video de salida
        out.write(frame)
        
        for track_id in memory.copy():
            memory[track_id]['frames_visibles'] -= 1
            if memory[track_id]['frames_visibles'] <= 0:
                del memory[track_id]

        # Calcular el tiempo de procesamiento
        frame_count += 1  # Contabilizar este frame

    cap.release()
    out.release()  # Liberar el escritor de video
    

    return frame_count, total_preprocess_time, total_inference_time, total_postprocess_time


def main():
    # Parámetros del modelo y archivo de salida
    model_path = '../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n.engine'
    video_path = '../datasets_labeled/videos/video_general_defectos.mp4'
    output_dir = '../inference_predictions/custom_tracker'

    # Asegurarse de que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)

    # Añadir timestamp al nombre del archivo de salida
    output_video_path = os.path.join(output_dir, f'video_con_tracking.mp4')

    # Cargar el modelo
    model = YOLO(model_path)

    # Definir las clases de los objetos
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

    # Inicializar memoria
    memory = {}

    # Procesar el video y obtener el tiempo medio por frame
    total_time = cv2.getTickCount()
    frame_count, total_preprocess_time, total_inference_time, total_postprocess_time = procesar_video(video_path, model, output_video_path, clases, memory)

    total_time = (cv2.getTickCount() - total_time) / cv2.getTickFrequency()  # Tiempo total en segundos
    average_time_per_frame = total_time / frame_count  # Tiempo medio por frame en segundos
    # Mostrar el tiempo medio por frame
    print(f'Número de frames procesados: {frame_count}')
    print(f'Tiempo total de procesamiento: {total_time:.3f} segundos')
    print(f'Tiempo medio por frame: {average_time_per_frame * 1000 :.3f} ms')
    
    print(f'Tiempo total de preprocess, inference y postprocess: {(total_preprocess_time + total_inference_time + total_postprocess_time) / 1000:.3f} segundos')
    print(f'Tiempo total de preprocess: {total_preprocess_time / 1000:.3f} segundos')
    print(f'Tiempo total de inference: {total_inference_time / 1000:.3f} segundos')
    print(f'Tiempo total de postprocess: {total_postprocess_time / 1000:.3f} segundos')
    print(f'Tiempo medio de preprocess: {total_preprocess_time / 1000:.3f} segundos')
    print(f'Tiempo medio de inference: {total_inference_time / 1000:.3f} segundos')
    print(f'Tiempo medio de postprocess: {total_postprocess_time / frame_count:.3f} ms')
    
    print(f'Porcentaje de tiempo en inferencia respecto al total: {(total_preprocess_time + total_inference_time + total_postprocess_time) / (total_time * 1000) * 100:.2f}%')
    print(f'Tiempo en CPU: {total_time - (total_preprocess_time + total_inference_time + total_postprocess_time) / 1000:.3f} segundos')
    
    tiempo_30fps = frame_count / 30
    
    print(f'Tiempo máximo de procesamiento maximizar la tasa de frames de la cámara (20FPS): {tiempo_30fps:.3f} segundos')
    print(f'Velocidad de procesamiento: {1 / average_time_per_frame:.2f} FPS')


if __name__ == "__main__":
    main()
