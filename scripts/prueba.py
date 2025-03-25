from ultralytics import YOLO
import cv2
from inference.lib.sahi import split_image_with_overlap, process_detection_results, apply_nms_custom
import os
import shutil
from typing import Dict, Tuple

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

# Crear directorio base 'slices' si no existe
base_dir = 'slices'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Encontrar el siguiente número de run disponible
run_number = 1
while os.path.exists(os.path.join(base_dir, f'run{run_number}')):
    run_number += 1

# Crear el directorio para esta ejecución
slices_dir = os.path.join(base_dir, f'run{run_number}')
os.makedirs(slices_dir)
print(f"Creado directorio para run {run_number}: {slices_dir}")

#model_path = '../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n_FP16_GPU_batch4.engine'
model_path = '../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n.pt'

model = YOLO(model_path, task='detect')

image_path = '../datasets_labeled/images/1080/049-1080x1080.jpg'

# Carga la imagen original
original_image = cv2.imread(image_path)

# Divide la imagen en slices con solapamiento
sliced_images, horizontal_splits, vertical_splits = split_image_with_overlap(
    original_image, 640, 640, 200
)

# Guarda los slices en el directorio del run actual
print(f"[Run {run_number}] Guardando slices...")
for i, img in enumerate(sliced_images):
    slice_path = os.path.join(slices_dir, f'slice_{i:02d}.jpg')
    cv2.imwrite(slice_path, img)

# Realiza la predicción en los slices
results = model.predict(sliced_images, conf=0.5, half=True, augment=True, batch=4)

predict_len = 0

# Pinta los resultados sobre los slices
for i, result in enumerate(results):
    cls = result.boxes.cls.cpu()
    conf = result.boxes.conf.cpu()
    xywh = result.boxes.xywh.cpu()
    
    # Crear una copia del slice para dibujar
    slice_with_dets = sliced_images[i].copy()
    
    for cls_val, conf_val, xywh_val in zip(cls, conf, xywh):
        predict_len += 1
        cls_int = int(cls_val.item())
        cls_name = CLASS_MAPPING.get(cls_int, f"clase_{cls_int}")
        color = COLOR_MAPPING.get(cls_name, (255, 255, 255))
        
        # Convertir de xywh a xyxy
        x, y, w, h = xywh_val.tolist()
        xmin = int(x - w/2)
        ymin = int(y - h/2)
        xmax = int(x + w/2)
        ymax = int(y + h/2)
        
        # Dibujar bbox con el color correspondiente
        cv2.rectangle(slice_with_dets, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Añadir etiqueta
        label = f"{cls_name} ({conf_val:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(slice_with_dets,
                     (xmin, ymin - text_height - 5),
                     (xmin + text_width, ymin),
                     (0, 0, 0),
                     -1)
        cv2.putText(slice_with_dets, label, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Guardar slice con detecciones
    cv2.imwrite(os.path.join(slices_dir, f'slice_{i:02d}_result.jpg'), slice_with_dets)
    print(f"[Run {run_number}] Procesado slice {i+1}/{len(sliced_images)}")

# Transforma los resultados al sistema de coordenadas de la imagen original
transformed_results = process_detection_results(
    results, horizontal_splits, vertical_splits, 640, 640, 200, original_image.shape[1], original_image.shape[0]
)

# 1. Guardar imagen con todas las detecciones sin filtrar
print(f"\n[Run {run_number}] Guardando imagen con todas las detecciones ({len(transformed_results)})...")
image_raw = original_image.copy()
for cls, conf, xmin, ymin, xmax, ymax in transformed_results:
    cls_name = CLASS_MAPPING.get(int(cls), f"clase_{int(cls)}")
    color = COLOR_MAPPING.get(cls_name, (255, 255, 255))
    
    cv2.rectangle(image_raw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
    
    # Añadir etiqueta
    label = f"{cls_name} ({conf:.2f})"
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image_raw,
                 (int(xmin), int(ymin) - text_height - 5),
                 (int(xmin) + text_width, int(ymin)),
                 (0, 0, 0),
                 -1)
    cv2.putText(image_raw, label, (int(xmin), int(ymin) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

raw_path = os.path.join(slices_dir, "detecciones_sin_filtrar.jpg")
cv2.imwrite(raw_path, image_raw)
image_raw = None

# 2. Aplicar y visualizar resultados de NMS
print(f"[Run {run_number}] Aplicando NMS (IoU: 0.3, Conf: 0.5)...")
final_results = apply_nms_custom(transformed_results, iou_threshold=0.3, conf_threshold=0.5)
print(f"Detecciones finales después de NMS: {len(final_results)}")

# Preparar imagen para resultados finales
image_with_final = original_image.copy()
original_image = None  # Liberar la imagen original

for i, (cls, conf, xmin, ymin, xmax, ymax) in enumerate(final_results):
    # Verifica que las coordenadas estén dentro de los límites de la imagen
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(image_with_final.shape[1], int(xmax))
    ymax = min(image_with_final.shape[0], int(ymax))
    
    cls_name = CLASS_MAPPING.get(int(cls), f"clase_{int(cls)}")
    color = COLOR_MAPPING.get(cls_name, (255, 255, 255))
    
    # Dibujar el rectángulo con el color correspondiente
    cv2.rectangle(image_with_final, (xmin, ymin), (xmax, ymax), color, 2)
    
    # Añadir texto con clase, ID y confianza
    label = f"#{i} {cls_name} ({conf:.2f})"
    
    # Dibujar fondo para el texto para mejor visibilidad
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image_with_final,
                 (xmin, ymin - text_height - 5),
                 (xmin + text_width, ymin),
                 (0, 0, 0),
                 -1)
    
    # Dibujar el texto con el color correspondiente
    cv2.putText(
        image_with_final,
        label,
        (xmin, ymin - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )
    
    # Imprimir información detallada en la consola
    print(f"Detection #{i}:")
    print(f"  - Class: {cls_name}")
    print(f"  - Confidence: {conf:.3f}")
    print(f"  - BBox: ({xmin},{ymin},{xmax},{ymax})")

# Guardar imagen final
print(f"\n[Run {run_number}] Guardando resultado final...")
final_path = os.path.join(slices_dir, "detecciones_final.jpg")
cv2.imwrite(final_path, image_with_final)
image_with_final = None  # Liberar memoria

print(f"\n[Run {run_number}] Resumen del proceso de detección:")
print(f"1. Procesamiento inicial:")
print(f"   - Detecciones en slices individuales: {predict_len}")
print(f"   - Archivos: slice_XX.jpg, slice_XX_result.jpg")

print(f"\n2. Proceso de filtrado:")
print(f"   a. Transformación a coordenadas globales: {len(transformed_results)} detecciones")
print(f"      → Ver: detecciones_sin_filtrar.jpg")
print(f"   b. Después de NMS: {len(final_results)} detecciones")
print(f"      → Ver: detecciones_final.jpg")

print(f"\nEstadísticas de reducción:")
if len(transformed_results) > 0:
    nms_filtered = ((len(transformed_results) - len(final_results)) / len(transformed_results) * 100)
    print(f"- NMS eliminó: {nms_filtered:.1f}% de las detecciones")

print(f"\nArchivos generados en: {slices_dir}")
print("1. Slices:")
print("   - Originales: slice_XX.jpg")
print("   - Con detecciones: slice_XX_result.jpg")
print("2. Resultados de filtrado:")
print("   - Todas las detecciones: detecciones_sin_filtrar.jpg")
print("   - Después de NMS: detecciones_final.jpg")