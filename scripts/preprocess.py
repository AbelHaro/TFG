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
    
    print(f"Original image dimensions: {original_width} x {original_height}")
    print(f"New sub-image dimensions: {new_width} x {new_height}")
    print(f"Overlap: {overlap_pixels} pixels")
    print(f"Step size (x, y): {step_x}, {step_y}")
    
    # Calculate the number of splits for horizontal and vertical axes
    horizontal_splits = math.ceil(original_width / step_x)
    vertical_splits = math.ceil(original_height / step_y)
    
    print(f"Splitting the image into {horizontal_splits} horizontal and {vertical_splits} vertical sub-images.")

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

    print(f"Split the image into {len(sub_images)} sub-images.")

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


if __name__ == "__main__":
    # Cargar la imagen original
    original_image = cv2.imread('./1080x1080.jpg')

    if original_image is None:
        print("Error: No se pudo cargar la imagen.")
        exit()

    model_path = "../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n.pt"
    model = YOLO(model_path, task="detect")

    new_width = 640
    new_height = 640
    overlap_pixels = 100

    # Dividir la imagen original en sub-imágenes con solapamiento
    sub_images, horizontal_splits, vertical_splits = split_image_with_overlap(original_image, new_width, new_height, overlap_pixels)
    print(f"Total de sub-imágenes: {len(sub_images)}")

    # Preparar las imágenes para la predicción
    images_chw = np.array([np.transpose(img, (2, 0, 1)) for img in sub_images])
    images_tensor = torch.tensor(images_chw, dtype=torch.float32)
    images_tensor /= 255.0
    images_tensor = images_tensor.half()

    # Realizar las predicciones con el modelo YOLO
    results = model.predict(sub_images, conf=0.5, half=True, augment=True, batch=4)

    iter = 0
    if os.path.exists("./results"):
        for file in os.listdir("./results"):
            file_path = os.path.join("./results", file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs("./results", exist_ok=True)
    
    # Para almacenar los resultados transformados
    transformed_results = []

    # Iterar sobre las sub-imágenes y los resultados de la detección
    for idx, (result, image) in enumerate(zip(results, sub_images)):
        print(f"Número de detecciones en sub-imagen {idx}: ", len(result))

        # Obtener las coordenadas y ajustar a la imagen original
        for cls, conf, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):

            # Las coordenadas locales en la sub-imagen
            xmin, ymin, xmax, ymax = map(int, xyxy)  # Convertir las coordenadas a enteros

            # Calcular el desplazamiento de la sub-imagen dentro de la imagen original
            row = idx // horizontal_splits  # Fila de la sub-imagen
            col = idx % horizontal_splits  # Columna de la sub-imagen
            

            # Calcular el desplazamiento para la sub-imagen
            offset_x = col * (new_width - overlap_pixels * 2)
            offset_y = row * (new_height - overlap_pixels * 2)
            

            # Ajustar las coordenadas a la imagen original
            global_xmin = xmin + offset_x
            global_ymin = ymin + offset_y
            global_xmax = xmax + offset_x
            global_ymax = ymax + offset_y

            # Dibujar el cuadro delimitador en la sub-imagen
            #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            #cv2.putText(image, f"{cls} ({conf:.2f})", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Almacenar los resultados transformados
            transformed_results.append((cls.item(), conf.item(), global_xmin, global_ymin, global_xmax, global_ymax))

        # Guardar la imagen con los cuadros delimitadores ajustados
        #cv2.imwrite(f"./results/1080x1080_{iter}.jpg", image)
        iter += 1

    print("---------------------------------------------------------------")
    
    print("Resultados transformados: ", len(transformed_results))
    print("Resultados sin NMS: ", transformed_results)
    
    for cls, conf, xmin, ymin, xmax, ymax in transformed_results:
        print(f"Clase: {cls}, Confianza: {conf:.2f}, Coordenadas: ({xmin}, {ymin}, {xmax}, {ymax})")
    
    print("Aplicando NMS...")
    
        
    nms_results = apply_nms(transformed_results, iou_threshold=0.4, conf_threshold=0.5)
        
    print("Resultados después de NMS: ", len(nms_results))
    print(f"Resultados con NMS:")
    
    for cls, conf, xmin, ymin, xmax, ymax in nms_results:
        print(f"Clase: {cls}, Confianza: {conf:.2f}, Coordenadas: ({xmin}, {ymin}, {xmax}, {ymax})")
    
    final_results = apply_overlapping(nms_results, overlap_threshold=0.8)
    
    print("Resultados finales: ", len(final_results))
    print(f"Resultados finales:")
    
    for cls, conf, xmin, ymin, xmax, ymax in final_results:
        print(f"Clase: {cls}, Confianza: {conf:.2f}, Coordenadas: ({xmin}, {ymin}, {xmax}, {ymax})")
    
        
    for cls, conf, xmin, ymin, xmax, ymax in final_results:
        cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
        cv2.putText(original_image, f"{cls} ({conf:.2f})", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
    cv2.imwrite("./results/1080x1080_junta.jpg", original_image)
                          
        
        
