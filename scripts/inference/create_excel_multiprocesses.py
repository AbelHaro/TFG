import os
import csv

def create_csv_file(file_name="default.csv"):
    """
    Crea o sobrescribe un archivo CSV con las cabeceras por defecto para tiempos.
    
    :param file_path: Ruta completa del archivo CSV.
    """
    # Cabeceras por defecto para tiempos
    headers = [
        "Frame", "Captura", "Procesamiento", "Tracking", "Escritura",
        "Cantidad Objetos", "Tiempo Total (ms)", "Tiempo por Objeto Tracking (ms)",
        "Preprocess", "Inference", "Postprocess"
    ]
    
    file_path = "/TFG/excels/" + file_name

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Crear o sobrescribir el archivo CSV con las cabeceras
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"[CREATE EXCEL] Archivo {file_path} creado o sobrescrito con cabeceras de tiempos.")
    
    return file_path


def add_row_to_csv(file_path, frame_index, times):
    """
    Añade una fila al archivo CSV con los datos de tiempos.
    
    :param file_path: Ruta completa del archivo CSV.
    :param frame_index: Índice del frame actual.
    :param times: Diccionario con los tiempos del frame.
    """
    capture_time = times.get("capture", 0)
    process_time = times.get("processing", 0)
    preprocess_time = times.get("detect_function", {}).get("preprocess", 0)
    inference_time = times.get("detect_function", {}).get("inference", 0)
    postprocess_time = times.get("detect_function", {}).get("postprocess", 0)
    track_time = times.get("tracking", 0)
    write_time = times.get("writing", 0)
    object_count = times.get("objects_count", 0)

    total_frame_time = capture_time + process_time + track_time + write_time
    time_per_object = (track_time / object_count * 1000) if object_count > 0 else 0

    row = [
        frame_index,
        format_number(capture_time),
        format_number(process_time),
        format_number(track_time),
        format_number(write_time),
        object_count,
        format_number(total_frame_time * 1000),
        format_number(time_per_object),
        format_number(preprocess_time),
        format_number(inference_time),
        format_number(postprocess_time)
    ]

    # Escribe la fila en el archivo CSV
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    #print(f"[CREATE EXCEL] Fila añadida en {file_path} para el frame {frame_index}")


def add_fps_to_csv(file_path, frame_index, fps_value):
    """
    Añade una fila al archivo CSV con el frame y su FPS.
    Si se llama por primera vez, reemplaza los encabezados existentes por los de FPS.
    
    :param file_path: Ruta completa del archivo CSV.
    :param frame_index: Índice del frame actual.
    :param fps_value: Valor de FPS calculado.
    """
    # Verificar si el archivo tiene los encabezados de tiempos
    rewrite_headers = False
    if os.path.exists(file_path):
        with open(file_path, mode='r', newline='') as f:
            first_row = next(csv.reader(f), [])
            if first_row != ["Frame", "FPS"]:  # No son los encabezados de FPS
                rewrite_headers = True

    # Si es la primera vez, reescribe los encabezados
    if rewrite_headers:
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "FPS"])
        #print(f"[CREATE EXCEL] Encabezados de tiempos reemplazados por los de FPS en {file_path}.")

    # Añadir fila con el valor de FPS
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frame_index, format_number(fps_value)])
    #print(f"[CREATE EXCEL] FPS {fps_value} añadido para el frame {frame_index} en {file_path}")


def format_number(number):
    """
    Formatea un número al estilo 'es_ES' (con coma como separador decimal).
    """
    return f"{number:.6f}".replace('.', ',')

