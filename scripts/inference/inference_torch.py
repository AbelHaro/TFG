import cv2
from ultralytics import YOLO

# Cargar el modelo
model = YOLO('../models/canicas/2024_11_15/2024_11_15_canicas_yolo11n.pt')

# Ruta de la imagen
image_path = '../datasets_labeled/2024_11_15_canicas_dataset/test/images/390.png'

clases = {
    0: 'negra',
    1: 'blanca',
    2: 'verde',
    3: 'azul',
    4: 'negra-d',
    5: 'blanca-d',
    6: 'verde-d',
    7: 'azul-d',
}

# Cargar la imagen
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en {image_path}")

# Realizar inferencia
results = model.track(
    source=image_path,           # (str, optional) source directory for images or videos
    device=0,                    # (int, optional) GPU id (0-9) or -1 for CPU
    persist=True,
    tracker='bytetrack.yaml',    # (str, optional) filename of tracker YAML
)

# Obtener coordenadas de las detecciones y IDs
boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
ids = results[0].boxes.id.cpu().numpy().astype(int)
classes = results[0].boxes.cls.cpu().numpy().astype(int)

print(results[0].boxes)

# Dibujar resultados en la imagen
for box, obj_id, cls in zip(boxes, ids, classes):
    xmin, ymin, xmax, ymax = box
    print(f'Detección {obj_id}: {xmin}, {ymin}, {xmax}, {ymax}')

    # Dibujar rectángulo alrededor del objeto detectado
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Agregar texto con el ID
    cv2.putText(image, f'{obj_id} - {clases[cls]}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Mostrar la imagen con las detecciones
cv2.imshow('Detecciones', image)

# Esperar a que el usuario presione una tecla y cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()
