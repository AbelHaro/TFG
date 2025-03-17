from ultralytics import YOLO
import cv2
from inference.lib.sahi import split_image_with_overlap, process_detection_results, apply_nms, apply_overlapping
import os
import shutil

# Elimina todo el contenido del directorio 'slices' si existe
if os.path.exists('slices'):
    shutil.rmtree('slices')

# Crea el directorio 'slices'
os.makedirs('slices', exist_ok=True)

model_path = '../models/canicas/2025_02_24/2025_02_24_canicas_yolo11n_FP16_GPU_batch4.engine'

model = YOLO(model_path, task='detect')

# Carga la imagen original
original_image = cv2.imread('1080x1080.jpg')

# Divide la imagen en slices con solapamiento
sliced_images, horizontal_splits, vertical_splits = split_image_with_overlap(
    original_image, 640, 640, 100
)

# Guarda los slices en el directorio 'slices'
for i, img in enumerate(sliced_images):
    cv2.imwrite(f'slices/slice_{i}.jpg', img)

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
        x, y, w, h = map(int, xywh_val)
        xmin = x - w // 2
        ymin = y - h // 2
        xmax = x + w // 2
        ymax = y + h // 2

        cv2.rectangle(sliced_images[i], (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.putText(
            sliced_images[i],
            str(cls_val.item()),
            (xmin, ymin),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    cv2.imwrite(f'slices/slice_{i}_result.jpg', sliced_images[i])

# Transforma los resultados al sistema de coordenadas de la imagen original
transformed_results = process_detection_results(
    results, horizontal_splits, vertical_splits, 640, 640, 10
)

# Copia de la imagen original antes de dibujar los resultados transformados
image_with_transformed = original_image.copy()

for cls, conf, xmin, ymin, xmax, ymax in transformed_results:
    cv2.rectangle(image_with_transformed, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    cv2.putText(
        image_with_transformed,
        str(cls),
        (xmin, ymin),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
cv2.imwrite("slices/1080x1080_result_overlap.jpg", image_with_transformed)

# Copia de la imagen original antes de dibujar los resultados finales tras NMS y overlap
image_with_final = original_image.copy()

# Aplica NMS a los resultados transformados
nms_results = apply_nms(transformed_results, iou_threshold=0.4, conf_threshold=0.5)

# Aplica NMS con solapamiento a los resultados
final_results = apply_overlapping(nms_results, overlap_threshold=0.8)

for i, (cls, conf, xmin, ymin, xmax, ymax) in enumerate(final_results):
    # Verifica que las coordenadas estén dentro de los límites de la imagen
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image_with_final.shape[1], xmax)
    ymax = min(image_with_final.shape[0], ymax)
    
    cv2.rectangle(image_with_final, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    cv2.putText(
        image_with_final,
        f"{i}: {cls}",
        (xmin, ymin),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
    print(f"Detection {i}: class={cls}, conf={conf:.2f}, bbox=({xmin},{ymin},{xmax},{ymax})")

cv2.imwrite("slices/1080x1080_final_result_v2.jpg", image_with_final)

print(f"Predictions: {predict_len}")
print(f"Tras el NMS: {len(nms_results)}")
print(f"Tras el NMS con solapamiento: {len(final_results)}")