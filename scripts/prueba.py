from ultralytics import YOLO
import cv2
from inference.lib.sahi import split_image_with_overlap, process_detection_results, apply_nms, apply_overlapping
import os
import shutil

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

model_path = '../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n_FP16_GPU_batch4.engine'

model = YOLO(model_path, task='detect')

# Carga la imagen original
original_image = cv2.imread('1080x1080.jpg')

# Divide la imagen en slices con solapamiento
sliced_images, horizontal_splits, vertical_splits = split_image_with_overlap(
    original_image, 640, 640, 100
)

# Guarda los slices en el directorio del run actual
print(f"[Run {run_number}] Guardando slices...")
for i, img in enumerate(sliced_images):
    slice_path = os.path.join(slices_dir, f'slice_{i:02d}.jpg')
    cv2.imwrite(slice_path, img)
    # Forzar la liberación de la imagen
    img = None

# Realiza la predicción en los slices
results = model.predict(sliced_images, conf=0.5, half=True, augment=True, batch=4)

predict_len = 0

# Pinta los resultados sobre los slices
for i, result in enumerate(results):
    cls = result.boxes.cls.cpu()
    conf = result.boxes.conf.cpu()
    xywh = result.boxes.xywh.cpu()

    for cls_val, conf_val, xywh_val in zip(cls, conf, xywh):
        predict_len += 1
        
    # Solo guardamos los slices sin dibujar las detecciones
    cv2.imwrite(os.path.join(slices_dir, f'slice_{i:02d}.jpg'), sliced_images[i])
    print(f"[Run {run_number}] Procesado slice {i+1}/{len(sliced_images)}")
    # Liberar la imagen después de guardarla
    sliced_images[i] = None

# Transforma los resultados al sistema de coordenadas de la imagen original
transformed_results = process_detection_results(
    results, horizontal_splits, vertical_splits, 640, 640, 100
)

# 1. Guardar imagen con todas las detecciones sin filtrar (AZUL)
print(f"\n[Run {run_number}] Guardando imagen con todas las detecciones ({len(transformed_results)})...")
image_raw = original_image.copy()
for cls, conf, xmin, ymin, xmax, ymax in transformed_results:
    # Color azul puro (255, 0, 0) en formato BGR
    cv2.rectangle(image_raw, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
raw_path = os.path.join(slices_dir, "detecciones_sin_filtrar_azul.jpg")
cv2.imwrite(raw_path, image_raw)
image_raw = None

# 2. Aplicar y visualizar resultados de NMS (ROJO)
print(f"[Run {run_number}] Aplicando NMS (IoU: 0.3, Conf: 0.5)...")
final_results = apply_nms(transformed_results, iou_threshold=0.3, conf_threshold=0.3)
print(f"Detecciones finales después de NMS: {len(final_results)}")

# # Código comentado: Filtrado de solapamiento
# print(f"[Run {run_number}] Aplicando filtrado de solapamiento (threshold: 0.8)...")
# overlap_results = apply_overlapping(final_results, overlap_threshold=0.8)
# print(f"Detecciones después del filtrado de solapamiento: {len(overlap_results)}")
#
# image_overlap = original_image.copy()
# for cls, conf, xmin, ymin, xmax, ymax in overlap_results:
#     cv2.rectangle(image_overlap, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#     cv2.putText(image_overlap, f"{conf:.2f}", (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# overlap_path = os.path.join(slices_dir, "detecciones_post_overlap_verde.jpg")
# cv2.imwrite(overlap_path, image_overlap)
# image_overlap = None

# Preparar imagen para resultados finales
image_with_final = original_image.copy()
original_image = None  # Liberar la imagen original

# Cambiar todos los cv2.rectangle y cv2.putText posteriores para usar rojo puro (0, 0, 255)

for i, (cls, conf, xmin, ymin, xmax, ymax) in enumerate(final_results):
    # Verifica que las coordenadas estén dentro de los límites de la imagen
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image_with_final.shape[1], xmax)
    ymax = min(image_with_final.shape[0], ymax)
    
    # Color rojo puro para las detecciones finales
    RED_COLOR = (0, 0, 255)  # BGR format
    
    # Dibujar el rectángulo en rojo
    cv2.rectangle(image_with_final, (xmin, ymin), (xmax, ymax), RED_COLOR, 2)
    
    # Añadir texto con clase, ID y confianza
    label = f"#{i} {cls} ({conf:.2f})"
    
    # Dibujar fondo para el texto para mejor visibilidad
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image_with_final,
                 (xmin, ymin - text_height - 5),
                 (xmin + text_width, ymin),
                 (0, 0, 0),
                 -1)
    
    # Dibujar el texto en rojo
    cv2.putText(
        image_with_final,
        label,
        (xmin, ymin - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        RED_COLOR,
        2,
    )
    
    # Imprimir información detallada en la consola
    print(f"Detection #{i}:")
    print(f"  - Class: {cls}")
    print(f"  - Confidence: {conf:.3f}")
    print(f"  - BBox: ({xmin},{ymin},{xmax},{ymax})")

# Guardar imagen final
print(f"\n[Run {run_number}] Guardando resultado final...")
final_path = os.path.join(slices_dir, "detecciones_final_rojo.jpg")
cv2.imwrite(final_path, image_with_final)
image_with_final = None  # Liberar memoria

print(f"\n[Run {run_number}] Resumen del proceso de detección:")
print(f"1. Procesamiento inicial:")
print(f"   - Detecciones en slices individuales: {predict_len}")
print(f"   - Archivos: slice_XX.jpg, slice_XX_result.jpg")

print(f"\n2. Proceso de filtrado:")
print(f"   a. Transformación a coordenadas globales: {len(transformed_results)} detecciones")
print(f"      → Ver: detecciones_sin_filtrar_azul.jpg (AZUL)")
print(f"   b. Después de NMS: {len(final_results)} detecciones")
print(f"      → Ver: detecciones_final_rojo.jpg (ROJO)")

print(f"\nEstadísticas de reducción:")
if len(transformed_results) > 0:
    nms_filtered = ((len(transformed_results) - len(final_results)) / len(transformed_results) * 100)
    print(f"- NMS eliminó: {nms_filtered:.1f}% de las detecciones")

print(f"\nArchivos generados en: {slices_dir}")
print("1. Slices:")
print("   - Originales: slice_XX.jpg")
print("   - Con detecciones: slice_XX_result.jpg")
print("2. Resultados de filtrado (por color):")
print("   - Todas las detecciones: detecciones_sin_filtrar_azul.jpg (AZUL)")
print("   - Después de NMS: detecciones_final_rojo.jpg (ROJO)")