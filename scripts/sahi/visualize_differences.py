from ultralytics import YOLO
import cv2
import numpy as np
from inference.lib.sahi import split_image_with_overlap, process_detection_results, apply_nms, apply_nms_custom
import os
import shutil
from datetime import datetime
from typing import Dict, Tuple, List

# Mapeo de clases a nombres
CLASS_MAPPING: Dict[int, str] = {
    0: "negra",
    1: "blanca",
    2: "verde",
    3: "azul",
    4: "negra-d",
    5: "blanca-d",
    6: "verde-d",
    7: "azul-d",
}

# Mapeo de nombres a colores BGR
COLOR_MAPPING: Dict[str, Tuple[int, int, int]] = {
    "negra": (0, 0, 255),      # Rojo en formato BGR
    "blanca": (0, 255, 0),     # Verde en formato BGR
    "verde": (255, 0, 0),      # Azul en formato BGR
    "azul": (255, 255, 0),     # Cyan en formato BGR
    "negra-d": (0, 165, 255),  # Naranja en formato BGR
    "blanca-d": (255, 165, 0), # Azul claro en formato BGR
    "verde-d": (255, 105, 180),# Rosa en formato BGR
    "azul-d": (255, 0, 255),   # Magenta en formato BGR
}

def draw_detection(image: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int, 
                  cls_id: int, confidence: float, detection_id: int) -> None:
    """
    Dibuja una detección en la imagen con el color correspondiente a su clase.
    
    Args:
        image: Imagen sobre la que dibujar
        xmin, ymin, xmax, ymax: Coordenadas de la caja
        cls_id: ID de la clase detectada
        confidence: Nivel de confianza de la detección
        detection_id: Número de la detección
    """
    # Obtener nombre y color de la clase
    class_name = CLASS_MAPPING.get(cls_id, f"unknown-{cls_id}")
    color = COLOR_MAPPING.get(class_name, (0, 0, 255))  # Rojo por defecto
    
    # Dibujar rectángulo
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    
    # Añadir etiqueta
    label = f"#{detection_id} {class_name} ({confidence:.2f})"
    
    # Fondo para texto
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image,
                 (xmin, ymin - text_height - 5),
                 (xmin + text_width, ymin),
                 (0, 0, 0),
                 -1)
    
    # Texto
    cv2.putText(
        image,
        label,
        (xmin, ymin - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

def process_image_batch(image_paths, model, output_dir, batch_size=4):
    """
    Procesa un lote de imágenes usando YOLO y SAHI.
    
    Args:
        image_paths (list): Lista de rutas de imágenes a procesar
        model: Modelo YOLO cargado
        output_dir (str): Directorio de salida
        batch_size (int): Tamaño del batch para predicción
    """
    total_images = len(image_paths)
    print(f"\nProcesando {total_images} imágenes con SAHI...")
    
    for idx, image_path in enumerate(image_paths, 1):
        try:
            # Cargar imagen
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                continue
                
            print(f"\nProcesando imagen {idx}/{total_images}: {os.path.basename(image_path)}")
            
            # Dividir imagen
            sliced_images, horizontal_splits, vertical_splits = split_image_with_overlap(
                original_image, 640, 640, 100
            )
            
            # Predicción
            try:
                results = model.predict(sliced_images, conf=0.5, half=True, augment=True, batch=batch_size)
            except Exception as e:
                print(f"Error en la predicción: {str(e)}")
                continue
            
            # Liberar memoria de slices
            sliced_images = None
            
            # Procesar resultados
            transformed_results = process_detection_results(
                results, horizontal_splits, vertical_splits, 640, 640, 100, original_image.shape[1], original_image.shape[0]    
            )
            
            # Aplicar NMS
            final_results = apply_nms_custom(transformed_results, iou_threshold=0.3, conf_threshold=0.3)
            print(f"Detecciones encontradas: {len(final_results)}")
            
            # Preparar imagen final
            image_with_final = original_image.copy()
            original_image = None
            
            # Dibujar resultados
            for i, (cls, conf, xmin, ymin, xmax, ymax) in enumerate(final_results):
                # Verificar coordenadas
                xmin = max(0, int(xmin))
                ymin = max(0, int(ymin))
                xmax = min(image_with_final.shape[1], int(xmax))
                ymax = min(image_with_final.shape[0], int(ymax))
                
                draw_detection(image_with_final, xmin, ymin, xmax, ymax, int(cls), float(conf), i)
            
            # Guardar resultado
            output_name = f"{os.path.basename(image_path)}_final.jpg"
            final_path = os.path.join(output_dir, output_name)
            cv2.imwrite(final_path, image_with_final)
            
            # Liberar memoria
            image_with_final = None
            
            print(f"Resultado guardado en: {final_path}")
            
        except Exception as e:
            print(f"Error procesando {image_path}: {str(e)}")
            continue

def process_image(image_paths, model, output_dir):
    """
    Procesa un lote de imágenes usando YOLO.
    
    Args:
        image_paths (list): Lista de rutas de imágenes a procesar
        model: Modelo YOLO cargado
        output_dir (str): Directorio de salida
    """
    total_images = len(image_paths)
    print(f"\nProcesando {total_images} imágenes sin SAHI...")
    
    for idx, image_path in enumerate(image_paths, 1):
        try:
            # Cargar imagen
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                continue
                
            print(f"\nProcesando imagen {idx}/{total_images}: {os.path.basename(image_path)}")
            
            # Predicción
            try:
                results = model.predict(original_image, conf=0.5, half=True, augment=True)
            except Exception as e:
                print(f"Error en la predicción: {str(e)}")
                continue
            
            # Extraer resultados
            boxes = results[0].boxes
            print(f"Detecciones encontradas: {len(boxes)}")
            
            # Preparar imagen final
            image_with_final = original_image.copy()
            original_image = None
            
            # Dibujar resultados
            for i in range(len(boxes)):
                # Obtener coordenadas (x, y, w, h) y convertir a (xmin, ymin, xmax, ymax)
                x, y, w, h = boxes.xywh[i].tolist()
                xmin = int(x - w/2)
                ymin = int(y - h/2)
                xmax = int(x + w/2)
                ymax = int(y + h/2)
                
                # Obtener clase y confianza
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                # Verificar coordenadas
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(image_with_final.shape[1], xmax)
                ymax = min(image_with_final.shape[0], ymax)
                
                draw_detection(image_with_final, xmin, ymin, xmax, ymax, cls, conf, i)
            
            # Guardar resultado
            output_name = f"{os.path.basename(image_path)}_final.jpg"
            final_path = os.path.join(output_dir, output_name)
            cv2.imwrite(final_path, image_with_final)
            
            # Liberar memoria
            image_with_final = None
            
            print(f"Resultado guardado en: {final_path}")
            
        except Exception as e:
            print(f"Error procesando {image_path}: {str(e)}")
            continue

def main():
    # Configuración de directorios
    base_dir = 'runs'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Crear directorio para esta ejecución
    run_number = 1
    while os.path.exists(os.path.join(base_dir, f'run{run_number}')):
        run_number += 1
    
    run_dir = os.path.join(base_dir, f'run{run_number}')
    os.makedirs(run_dir)
    print(f"Creado directorio para run {run_number}: {run_dir}")
    
    # Cargar modelo
    model_path_batch = '../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n_FP16_GPU_batch4.engine'
    try:
        model_batch = YOLO(model_path_batch, task='detect')
        print("Modelo con batch cargado correctamente")
    except Exception as e:
        print(f"Error cargando el modelo: {str(e)}")
        return
    
    # Configurar directorio de imágenes
    images_path_1080 = '../datasets_labeled/images/1080'
    try:
        image_files_1080 = [os.path.join(images_path_1080, f) for f in os.listdir(images_path_1080)]
    except Exception as e:
        print(f"Error accediendo al directorio de imágenes: {str(e)}")
        return
    
    if not image_files_1080:
        print(f"No se encontraron imágenes en {images_path_1080}")
        return
    
    dir_1080 = os.path.join(run_dir, '1080')
    os.makedirs(dir_1080)
    
    # Procesar imágenes con batch y SAHI
    process_image_batch(image_files_1080, model_batch, dir_1080)
    
    print("\nProcesamiento con SAHI completado.")
    print(f"Los resultados se encuentran en: {dir_1080}")
    
    # Cargar modelo sin batch
    model_path = '../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n_FP16_GPU.engine'
    try:
        model = YOLO(model_path, task='detect')
        print("Modelo sin batch cargado correctamente")
    except Exception as e:
        print(f"Error cargando el modelo sin batch: {str(e)}")
        return
    
    dir_640 = os.path.join(run_dir, '640')
    os.makedirs(dir_640)
    
    images_path_640 = '../datasets_labeled/images/640'
    
    try:
        image_files_640 = [os.path.join(images_path_640, f) for f in os.listdir(images_path_640)]
    except Exception as e:
        print(f"Error accediendo al directorio de imágenes: {str(e)}")
        return
    
    # Procesar imágenes sin batch y sin SAHI
    process_image(image_files_640, model, dir_640)
    
    print("\nProcesamiento sin SAHI completado.")
    print(f"Los resultados se encuentran en: {dir_640}")

if __name__ == "__main__":
    main()