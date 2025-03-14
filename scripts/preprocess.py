import torch
import numpy as np
from argparse import Namespace
from ultralytics import YOLO  # type: ignore
import math
import cv2

from inference.lib.sahi import split_image_with_overlap, process_detection_results, apply_nms, apply_overlapping


def main():
    
    video_path = "../datasets_labeled/videos/test/test_640x640.mp4"
    model_path = "../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n.pt"
    
    model = YOLO(model_path, task="detect")
    
    new_width = 640
    new_height = 640
    overlap_pixels = 100
    
    # Cargar el video
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  
    
        sub_images, horizontal_splits, vertical_splits = split_image_with_overlap(frame, new_width, new_height, overlap_pixels)


        images_chw = np.array([np.transpose(img, (2, 0, 1)) for img in sub_images])
        images_tensor = torch.tensor(images_chw, dtype=torch.float32)
        images_tensor /= 255.0
        images_tensor = images_tensor.half()

        results = model.predict(sub_images, conf=0.5, half=True, augment=True, batch=4)
        
        transformed_results = process_detection_results(results, horizontal_splits, vertical_splits, new_width, new_height, overlap_pixels
        )

        # Se aplica NMS a los resultados
        nms_results = apply_nms(transformed_results, iou_threshold=0.4, conf_threshold=0.5)

        # Se aplica NMS con solapamiento a los resultados
        final_results = apply_overlapping(nms_results, overlap_threshold=0.8)

        # Inicializar listas vacías
        xywh = []
        confidences = []
        classes = []

        # Procesar cada detección en final_results
        for i, result in enumerate(final_results):

            cls, conf, xmin, ymin, xmax, ymax = result

            # Calcular las coordenadas xywh (x, y, ancho, alto)
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            xywh.append([x, y, width, height])

            # Agregar la confianza y la clase a las listas
            confidences.append(conf)
            classes.append(cls)

            # Convertir las listas a tensores de PyTorch con tamaño correcto
            xywh_tensor = torch.tensor(xywh, dtype=torch.float32) if xywh else torch.empty((0, 4), dtype=torch.float32)
            confidences_tensor = torch.tensor(confidences, dtype=torch.float32) if confidences else torch.empty(0, dtype=torch.float32)
            classes_tensor = torch.tensor(classes, dtype=torch.float32) if classes else torch.empty(0, dtype=torch.float32)

            # Crear el objeto Namespace con los resultados formateados
            result_formatted = Namespace(
            xywh=xywh_tensor,  # Tensor de 2D: [N, 4]
            conf=confidences_tensor,  # Tensor de 2D: [N, 1]
            cls=classes_tensor,  # Tensor de 2D: [N, 1]
            )
            
            
            rf_xywh = result_formatted.xywh
            rf_conf = result_formatted.conf
            rf_cls = result_formatted.cls
            

            # Mostrar los resultados formateados
            print("Resultados formateados:", result_formatted)
            
            cv2.rectangle(frame, (int(rf_xywh[0][0]), int(rf_xywh[0][1])), (int(rf_xywh[0][0] + rf_xywh[0][2]), int(rf_xywh[0][1] + rf_xywh[0][3])), (0, 255, 0), 2)
            cv2.putText(frame, str(rf_cls[0].item()), (int(rf_xywh[0][0]), int(rf_xywh[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
    
if __name__ == "__main__":
    main()