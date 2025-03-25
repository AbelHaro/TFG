from ultralytics import YOLO
import cv2
import numpy as np
from inference.lib.sahi import split_image_with_overlap, process_detection_results, apply_nms, apply_nms_custom
import os
import shutil
from datetime import datetime
from typing import Dict, Tuple, List
import glob

def process_normal_results(results) -> List[Tuple[float, float, float, float, int, float]]:
    """
    Procesa los resultados de la inferencia normal para obtener formato xyxy, cls, conf
    
    Args:
        results: Resultados de YOLO
        
    Returns:
        List[Tuple]: Lista de detecciones en formato (xmin, ymin, xmax, ymax, cls, conf)
    """
    boxes = results[0].boxes
    processed_results = []
    
    for i in range(len(boxes)):
        # Convertir de xywh a xyxy
        x, y, w, h = boxes.xywh[i].tolist()
        xmin = float(x - w/2)
        ymin = float(y - h/2)
        xmax = float(x + w/2)
        ymax = float(y + h/2)
        
        # Obtener clase y confianza
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        
        processed_results.append((xmin, ymin, xmax, ymax, cls, conf))
    
    return processed_results

def process_sahi_results(results, horizontal_splits, vertical_splits, original_width, original_height) -> List[Tuple[float, float, float, float, int, float]]:
    """
    Procesa los resultados de la inferencia con SAHI para obtener formato xyxy, cls, conf
    
    Args:
        results: Resultados de YOLO con SAHI
        horizontal_splits: Divisiones horizontales de la imagen
        vertical_splits: Divisiones verticales de la imagen
        
    Returns:
        List[Tuple]: Lista de detecciones en formato (xmin, ymin, xmax, ymax, cls, conf)
    """
    # Procesar detecciones de los slices a coordenadas globales
    transformed_results = process_detection_results(
        results, horizontal_splits, vertical_splits, 640, 640, 100, original_width, original_height
    )
    
    print(f"Detecciones de SAHI antes del NMS: {len(transformed_results)}")
    
    # Aplicar NMS
    final_results = apply_nms_custom(transformed_results, iou_threshold=0.3, conf_threshold=0.3)
    
    # Los resultados de SAHI ya vienen en formato (cls, conf, xmin, ymin, xmax, ymax)
    # Reordenar a (xmin, ymin, xmax, ymax, cls, conf)
    processed_results = [(float(xmin), float(ymin), float(xmax), float(ymax), int(cls), float(conf)) 
                        for cls, conf, xmin, ymin, xmax, ymax in final_results]
    
    return processed_results

def process_ground_truth(label_path: str, image_shape: Tuple[int, int]) -> List[Tuple[float, float, float, float, int, float]]:
    """
    Procesa el archivo de ground truth que contiene las coordenadas en formato
    class x_center y_center width height (normalizadas)
    
    Args:
        label_path: Ruta al archivo .txt de labels
        image_shape: Tupla (height, width) de la imagen
        
    Returns:
        List[Tuple]: Lista de ground truth en formato (xmin, ymin, xmax, ymax, cls, conf=1.0)
    """
    boxes = []
    height, width = image_shape
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    # Formato: class x_center y_center width height (normalizados)
                    values = line.strip().split()
                    if len(values) == 5:
                        cls = int(values[0])
                        x_center, y_center, w, h = map(float, values[1:])
                        
                        # Convertir coordenadas normalizadas a píxeles
                        x_center *= width
                        y_center *= height
                        w *= width
                        h *= height
                        
                        # Convertir a coordenadas absolutas (xmin, ymin, xmax, ymax)
                        xmin = x_center - w/2
                        ymin = y_center - h/2
                        xmax = x_center + w/2
                        ymax = y_center + h/2
                        
                        # Asegurar que las coordenadas están dentro de la imagen
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(width, xmax)
                        ymax = min(height, ymax)
                        
                        # Usar confidence = 1.0 para ground truth
                        boxes.append((xmin, ymin, xmax, ymax, cls, 1.0))
                    else:
                        print(f"Error: línea con formato incorrecto en {label_path}: {line.strip()}")
                except ValueError as e:
                    print(f"Error procesando línea en {label_path}: {line.strip()}")
                    continue
    except Exception as e:
        print(f"Error leyendo archivo {label_path}: {e}")
        return []
    
    return boxes

def get_image_sets(base_path: str) -> List[Tuple[str, str, str]]:
    """
    Obtiene los conjuntos de imágenes y sus labels correspondientes.
    
    Args:
        base_path: Ruta base de las imágenes
        
    Returns:
        List[Tuple]: Lista de tuplas (imagen_640, imagen_1080, label)
    """
    # Definir rutas de subdirectorios
    path_640 = os.path.join(base_path, '640')
    path_1080 = os.path.join(base_path, '1080')
    path_labels = os.path.join(base_path, 'labels')
    
    # Verificar que existan los directorios
    if not all(os.path.exists(p) for p in [path_640, path_1080, path_labels]):
        raise RuntimeError("No se encontraron todos los directorios necesarios")
    
    image_sets = []
    
    # Obtener prefijos únicos de las imágenes 640x640
    files_640 = os.listdir(path_640)
    prefixes = set()
    
    for f in files_640:
        if f.endswith('.jpg'):
            prefix = f.split('-')[0]
            if prefix.isdigit():  # Asegurarse de que es un número
                prefixes.add(prefix.zfill(3))  # Rellenar con ceros para ordenar correctamente
    
    # Para cada prefijo, buscar sus archivos correspondientes
    for prefix in sorted(prefixes):  # sorted() ordenará los prefijos numéricamente
        img_640 = f"{prefix}-640x640.jpg"
        img_1080 = f"{prefix}-1080x1080.jpg"
        label = f"{prefix}-label.txt"
        
        # Verificar que existan los tres archivos
        path_640_file = os.path.join(path_640, img_640)
        path_1080_file = os.path.join(path_1080, img_1080)
        path_label_file = os.path.join(path_labels, label)
        
        if all(os.path.exists(p) for p in [path_640_file, path_1080_file, path_label_file]):
            image_sets.append((path_640_file, path_1080_file, path_label_file))
    
    return image_sets

def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Calcula la intersección sobre unión (IoU) entre dos cajas delimitadoras.
    
    Args:
        box1: Primera caja en formato (xmin, ymin, xmax, ymax)
        box2: Segunda caja en formato (xmin, ymin, xmax, ymax)
        
    Returns:
        float: Valor IoU entre 0 y 1
    """
    # Calcular coordenadas de intersección
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    
    # Verificar si hay intersección
    if xmax <= xmin or ymax <= ymin:
        return 0.0
        
    # Calcular áreas
    intersection = (xmax - xmin) * (ymax - ymin)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_metrics(detections: List[Tuple], ground_truth: List[Tuple], iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
    Calcula precisión y recall para un umbral IoU específico.
    
    Args:
        detections: Lista de detecciones en formato (xmin, ymin, xmax, ymax, cls, conf)
        ground_truth: Lista de ground truth en formato (xmin, ymin, xmax, ymax, cls, conf)
        iou_threshold: Umbral de IoU para considerar una detección correcta
        
    Returns:
        Tuple[float, float]: (precision, recall)
    """
    if not ground_truth:  # Si no hay ground truth
        return 0.0, 0.0
        
    if not detections:  # Si no hay detecciones
        return 0.0, 0.0
    
    # Matriz de IoU entre todas las detecciones y ground truths
    iou_matrix = np.zeros((len(detections), len(ground_truth)))
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truth):
            if det[4] == gt[4]:  # Si son de la misma clase
                iou_matrix[i,j] = calculate_iou(det[:4], gt[:4])
            else:
                iou_matrix[i,j] = 0.0
    
    # Considerar como true positive cualquier detección que tenga IoU >= threshold
    # con al menos un ground truth de la misma clase
    true_positives = np.sum(np.any(iou_matrix >= iou_threshold, axis=1))
    
    # Para el recall, contar cuántos ground truths tienen al menos una detección válida
    detected_gt = np.sum(np.any(iou_matrix >= iou_threshold, axis=0))
    
    # Calcular métricas
    precision = true_positives / len(detections)
    recall = detected_gt / len(ground_truth)
    
    return precision, recall

def evaluate_detections(results: Dict[str, Dict], mode: str = 'normal') -> Dict[str, float]:
    """
    Evalúa las detecciones calculando IoU50, IoU50-95, precisión y recall.
    
    Args:
        results: Diccionario con los resultados de las detecciones
        mode: 'normal' o 'sahi' para seleccionar el tipo de detecciones y ground truth
        
    Returns:
        Dict[str, float]: Diccionario con las métricas calculadas
    """
    # Inicializar acumuladores
    total_precision = 0.0
    total_recall = 0.0
    total_iou50 = 0.0
    total_iou5095 = 0.0
    num_images = len(results)
    
    for set_id, data in results.items():
        detections = data[mode]
        ground_truth = data[f'ground_truth_{mode}']
        
        # Calcular precisión y recall con IoU 0.5
        precision, recall = calculate_metrics(detections, ground_truth, 0.5)
        total_precision += precision
        total_recall += recall
        
        # Calcular IoU para cada detección con su mejor ground truth match
        iou_scores = []
        for det in detections:
            best_iou = 0.0
            for gt in ground_truth:
                if det[4] == gt[4]:  # Misma clase
                    iou = calculate_iou(det[:4], gt[:4])
                    best_iou = max(best_iou, iou)
            if best_iou > 0:  # Solo considerar detecciones con algún match
                iou_scores.append(best_iou)
        
        # IoU50
        iou50 = sum(1 for iou in iou_scores if iou >= 0.5) / len(detections) if detections else 0
        total_iou50 += iou50
        
        # IoU50-95
        thresholds = np.arange(0.5, 1.0, 0.05)
        iou5095_scores = []
        for threshold in thresholds:
            score = sum(1 for iou in iou_scores if iou >= threshold) / len(detections) if detections else 0
            iou5095_scores.append(score)
        total_iou5095 += np.mean(iou5095_scores) if iou5095_scores else 0
    
    # Calcular promedios
    return {
        'precision': total_precision / num_images,
        'recall': total_recall / num_images,
        'iou50': total_iou50 / num_images,
        'iou50-95': total_iou5095 / num_images
    }

def get_next_run_number(base_path: str) -> int:
    """
    Obtiene el siguiente número de run basado en las carpetas existentes.
    
    Args:
        base_path: Ruta base donde se encuentran las carpetas de runs
        
    Returns:
        int: Siguiente número de run
    """
    # Buscar todas las carpetas que coincidan con el patrón run_*
    existing_runs = glob.glob(os.path.join(base_path, 'run_*'))
    if not existing_runs:
        return 1
        
    # Extraer números de run y encontrar el máximo
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(os.path.basename(run).split('_')[1])
            run_numbers.append(num)
        except (ValueError, IndexError):
            continue
            
    return max(run_numbers) + 1 if run_numbers else 1

def main():
    # Configurar rutas
    images_path = '../datasets_labeled/images'
    results_base_path = 'runs/comparative_inference'
    
    # Obtener siguiente número de run y crear directorio
    run_number = get_next_run_number(results_base_path)
    results_path = os.path.join(results_base_path, f'run_{run_number}')
    os.makedirs(results_path, exist_ok=True)
    
    # Cargar modelos
    models_path = '../models/canicas/2025_02_24'
    batch4_path = os.path.join(models_path, '2025_02_24_canicas_yolo11n_FP16_GPU_batch4.engine')
    batch1_path = os.path.join(models_path, '2025_02_24_canicas_yolo11n_FP16_GPU.engine')
    
    try:
        model_batch4 = YOLO(batch4_path, task='detect')
        model_batch1 = YOLO(batch1_path, task='detect')
        print("Modelos cargados correctamente")
    except Exception as e:
        print('Error loading models:', e)
        return
    
    # Obtener conjuntos de imágenes
    try:
        image_sets = get_image_sets(images_path)
        print(f"Encontrados {len(image_sets)} conjuntos de imágenes")
    except Exception as e:
        print(f"Error obteniendo conjuntos de imágenes: {e}")
        return
    
    # Diccionario para almacenar resultados
    all_results = {}
    
    # Archivo para guardar resultados
    results_file = os.path.join(results_path, 'results.txt')
    
    # Procesar cada conjunto de imágenes
    for path_640, path_1080, path_label in image_sets:
        # Obtener ID del conjunto desde el nombre del archivo
        set_id = os.path.basename(path_640).split('-')[0]
        print(f"\nProcesando conjunto {set_id}")
        
        # Cargar imágenes
        image_640 = cv2.imread(path_640)
        image_1080 = cv2.imread(path_1080)
        
        if image_640 is None or image_1080 is None:
            print(f"Error cargando imágenes")
            continue
            
        print(f"Tamaño imagen 640: {image_640.shape}")
        print(f"Tamaño imagen 1080: {image_1080.shape}")
        
        # Inferencia normal (640x640)
        results_normal = model_batch1.predict(image_640, conf=0.5, half=True, augment=True)
        detections_normal = process_normal_results(results_normal)
        print(f"Detecciones normales encontradas: {len(detections_normal)}")
        
        # Inferencia con SAHI (1080x1080)
        sliced_images_1080, horizontal_splits, vertical_splits = split_image_with_overlap(
            image_1080, 640, 640, 100
        )
        results_sahi = model_batch4.predict(sliced_images_1080, conf=0.5, half=True, augment=True, batch=4)
        detections_sahi = process_sahi_results(results_sahi, horizontal_splits, vertical_splits, image_1080.shape[1], image_1080.shape[0])
        print(f"Detecciones SAHI encontradas: {len(detections_sahi)}")
        
        # Procesar ground truth - usar tamaño de imagen 1080 ya que las etiquetas están en ese formato
        ground_truth_boxes_640 = process_ground_truth(path_label, image_640.shape[:2])
        print(f"Ground truth boxes: {len(ground_truth_boxes_640)}")
        
        ground_truth_boxes_1080 = process_ground_truth(path_label, image_1080.shape[:2])
        print(f"Ground truth boxes: {len(ground_truth_boxes_1080)}")
        
        # Guardar detecciones en el diccionario
        all_results[set_id] = {
            'normal': detections_normal,
            'sahi': detections_sahi,
            'ground_truth_normal': ground_truth_boxes_640,
            'ground_truth_sahi': ground_truth_boxes_1080,
            'image_shape': image_1080.shape[:2]  # (height, width)
        }
    
    # Calcular métricas para ambos métodos
    print("\nCalculando métricas...")
    
    metrics_normal = evaluate_detections(all_results, 'normal')
    metrics_sahi = evaluate_detections(all_results, 'sahi')
    
    # Guardar resultados en archivo
    with open(results_file, 'w') as f:
        f.write(f"Resultados de la comparación - Run {run_number}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Métricas para detección normal:\n")
        f.write(f"Precision: {metrics_normal['precision']:.4f}\n")
        f.write(f"Recall: {metrics_normal['recall']:.4f}\n")
        f.write(f"IoU50: {metrics_normal['iou50']:.4f}\n")
        f.write(f"IoU50-95: {metrics_normal['iou50-95']:.4f}\n\n")
        
        f.write("Métricas para detección SAHI:\n")
        f.write(f"Precision: {metrics_sahi['precision']:.4f}\n")
        f.write(f"Recall: {metrics_sahi['recall']:.4f}\n")
        f.write(f"IoU50: {metrics_sahi['iou50']:.4f}\n")
        f.write(f"IoU50-95: {metrics_sahi['iou50-95']:.4f}\n")
    
    print("\nProcesamiento completado.")
    print(f"Resultados guardados en: {results_file}")

if __name__ == "__main__":
    main()
