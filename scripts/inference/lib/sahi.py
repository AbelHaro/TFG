import cv2
import numpy as np
import math as Math
import torch
from ultralytics import YOLO  # type: ignore
import os
import shutil

import numpy as np
import math

def split_image_with_overlap(original_image, new_width, new_height, overlap_pixels):
    """
    Splits an image into smaller sub-images with a specified overlap in pixels.
    All sub-images will have the same size (new_width x new_height), even those at the edges.

    Args:
        original_image (numpy.ndarray): The original image to split.
        new_width (int): Width of the new sub-images.
        new_height (int): Height of the new sub-images.
        overlap_pixels (int): Overlap between sub-images in pixels.

    Returns:
        list: A list of sub-images.
        int: The number of horizontal splits.
        int: The number of vertical splits.
    """
    # Get the dimensions of the original image
    original_height, original_width = original_image.shape[:2]

    # Check if the new dimensions are valid
    if new_width > original_width or new_height > original_height:
        raise ValueError("New dimensions cannot be larger than the original image dimensions.")

    # Ensure overlap_pixels is a valid value
    if overlap_pixels < 0:
        raise ValueError("Overlap pixels must be a non-negative value.")

    # Calculate the step size for x and y (accounting for overlap)
    step_x = new_width - overlap_pixels
    step_y = new_height - overlap_pixels

    # Initialize a list to store the sub-images
    sub_images = []
    
    
    # Calculate the number of splits for horizontal and vertical axes
    horizontal_splits = math.ceil(original_width / step_x)
    vertical_splits = math.ceil(original_height / step_y)
    

    # Use for loops to extract the sub-images
    for y in range(0, original_height, step_y):
        for x in range(0, original_width, step_x):
            # Calculate the start and end coordinates for the sub-image
            start_x = x
            start_y = y
            end_x = start_x + new_width
            end_y = start_y + new_height

            # If the sub-image goes beyond the original image boundaries, adjust the start coordinates
            if end_x > original_width:
                start_x = original_width - new_width
                end_x = original_width
            if end_y > original_height:
                start_y = original_height - new_height
                end_y = original_height

            # Extract the sub-image
            sub_image = original_image[start_y:end_y, start_x:end_x]

            # If the sub-image is smaller than the desired size, pad it with zeros (black)
            if sub_image.shape[0] < new_height or sub_image.shape[1] < new_width:
                padded_sub_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                padded_sub_image[:sub_image.shape[0], :sub_image.shape[1]] = sub_image
                sub_image = padded_sub_image

            sub_images.append(sub_image)

    return sub_images, horizontal_splits, vertical_splits

def apply_nms(transformed_results, iou_threshold=0.4, conf_threshold=0.5):
    """
    Aplica el algoritmo NMS para eliminar las cajas superpuestas.

    Args:
        transformed_results (list): Lista de resultados con clases, confianza y coordenadas de las cajas.
        iou_threshold (float): Umbral de IOU para la NMS. (por defecto 0.4).
        conf_threshold (float): Umbral de confianza. Las cajas con confianza menor que este umbral se descartan. (por defecto 0.5).

    Returns:
        list: Lista de resultados filtrados después de aplicar NMS.
    """
    boxes = []
    confidences = []
    class_ids = []

    # Recolectamos las cajas, las confianzas y las clases
    for cls, conf, xmin, ymin, xmax, ymax in transformed_results:
        if conf >= conf_threshold:  # Si la confianza es mayor que el umbral
            boxes.append([xmin, ymin, xmax, ymax])
            confidences.append(conf)
            class_ids.append(cls)

    # Aplica la función NMS de OpenCV
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=conf_threshold, nms_threshold=iou_threshold)
    

    # Filtrar las cajas de acuerdo con los índices de NMS
    filtered_results = []
    
    
    
    if len(indices) == 0:
        return filtered_results
    
    for i in indices.flatten():
        cls = class_ids[i]
        conf = confidences[i]
        xmin, ymin, xmax, ymax = boxes[i]
        filtered_results.append((cls, conf, xmin, ymin, xmax, ymax))

    return filtered_results


def is_box_inside(box1, box2, threshold=0.8):
    """
    Verifica si box1 está dentro de box2 en al menos el porcentaje dado de la caja más pequeña.

    Args:
        box1 (list): Coordenadas de la caja 1 [xmin, ymin, xmax, ymax].
        box2 (list): Coordenadas de la caja 2 [xmin, ymin, xmax, ymax].
        threshold (float): Umbral de solapamiento (porcentaje de la caja pequeña que debe estar dentro de la grande).

    Returns:
        bool: Si box1 está dentro de box2 en al menos el umbral especificado.
    """
    # Obtener las coordenadas de las cajas
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # Verificar si box1 está completamente dentro de box2
    if xmin1 >= xmin2 and ymin1 >= ymin2 and xmax1 <= xmax2 and ymax1 <= ymax2:
        # Calcular el área de ambas cajas
        box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

        # Verificar si box1 está dentro de box2 en al menos el umbral
        intersection_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        if intersection_area / box1_area >= threshold:
            return True
    return False

def apply_overlapping(detected_boxes, overlap_threshold=0.8):
    """
    Aplica la lógica de eliminación de cajas basadas en el solapamiento.
    Si dos objetos de la misma clase están dentro de otro en más del 80% de su tamaño, 
    se conserva solo la caja más grande.

    Args:
        detected_boxes (list): Lista de cajas detectadas [clase, confianza, xmin, ymin, xmax, ymax].
        overlap_threshold (float): Umbral de solapamiento para considerar que una caja está dentro de otra (por defecto 0.8).

    Returns:
        list: Lista de cajas filtradas.
    """
    filtered_boxes = []

    for i in range(len(detected_boxes)):
        cls1, conf1, xmin1, ymin1, xmax1, ymax1 = detected_boxes[i]
        box1 = [xmin1, ymin1, xmax1, ymax1]
        
        # Marcar si la caja debe ser eliminada (ser borrada) o no
        should_add = True
        
        for j in range(len(detected_boxes)):
            if i == j:
                continue  # No comparar la caja consigo misma
                
            cls2, conf2, xmin2, ymin2, xmax2, ymax2 = detected_boxes[j]
            box2 = [xmin2, ymin2, xmax2, ymax2]
            
            if cls1 == cls2:  # Solo comparar cajas de la misma clase
                if is_box_inside(box1, box2, overlap_threshold):
                    # Si box1 está dentro de box2 en al menos el umbral de overlap_threshold,
                    # eliminar la caja más pequeña (box1 en este caso)
                    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
                    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

                    if box2_area >= box1_area:  # Si la caja 2 es más grande, mantenerla
                        should_add = False
                        break

        if should_add:
            filtered_boxes.append((cls1, conf1, xmin1, ymin1, xmax1, ymax1))

    return filtered_boxes

def process_detection_results(results, horizontal_splits, vertical_splits, new_width, new_height, overlap_pixels):
    """
    Procesa los resultados de detección de las sub-imágenes y los transforma a coordenadas globales.
    
    Args:
        results: Resultados de detección YOLO para cada sub-imagen.
        sub_images: Lista de sub-imágenes.
        horizontal_splits: Número de divisiones horizontales.
        new_width: Ancho de cada sub-imagen.
        new_height: Alto de cada sub-imagen.
        overlap_pixels: Solapamiento entre sub-imágenes en píxeles.
        
    Returns:
        list: Lista de detecciones transformadas a coordenadas globales.
    """
    transformed_results = []
    iter = 0
    
    # Iterar sobre las sub-imágenes y los resultados de la detección

    for idx in range(len(results)):
        
        
        
        # Obtener las clases, confianzas y coordenadas de las cajas
        cls = results[idx].boxes.cls.cpu()
        conf = results[idx].boxes.conf.cpu()
        xyxy = results[idx].boxes.xyxy.cpu()

    

        for cls, conf, xyxy in zip(cls, conf, xyxy):

            # Las coordenadas locales en la sub-imagen
            xmin, ymin, xmax, ymax = map(int, xyxy)  # Convertir las coordenadas a enteros

            # Calcular el desplazamiento de la sub-imagen dentro de la imagen original
            row = idx // vertical_splits  # Fila de la sub-imagen
            col = idx % horizontal_splits  # Columna de la sub-imagen
            
            # Calcular el desplazamiento para la sub-imagen
            offset_x = col * (new_width - overlap_pixels * 2)
            offset_y = row * (new_height - overlap_pixels * 2)
            
            # Ajustar las coordenadas a la imagen original
            global_xmin = xmin + offset_x
            global_ymin = ymin + offset_y
            global_xmax = xmax + offset_x
            global_ymax = ymax + offset_y

            # Almacenar los resultados transformados
            transformed_results.append((cls.item(), conf.item(), global_xmin, global_ymin, global_xmax, global_ymax))

        # Guardar la imagen con los cuadros delimitadores ajustados
        #cv2.imwrite(f"./results/1080x1080_{iter}.jpg", image)
        iter += 1
        
    return transformed_results