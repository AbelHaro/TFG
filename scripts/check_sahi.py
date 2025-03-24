from inference.lib.sahi import split_image_with_overlap, process_detection_results, apply_nms, apply_overlapping
import cv2

image_path = '../datasets_labeled/images/1080/049-1080x1080.jpg'

# Carga la imagen original
original_image = cv2.imread(image_path)

sub_images, horizontal_splits, vertical_splits = split_image_with_overlap(original_image, 640, 640, 200)

print(f"Se han generado {len(sub_images)} sub-imágenes, con {horizontal_splits} divisiones horizontales y {vertical_splits} divisiones verticales.")

for idx, sub_image in enumerate(sub_images):
    cv2.imwrite(f'sub-image_{idx}.jpg', sub_image)