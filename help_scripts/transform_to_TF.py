import os
import tensorflow as tf

# Función para cargar las anotaciones en formato YOLOv8
def load_yolo_annotations(data_dir):
    images = []
    labels = []
    
    for image_file in os.listdir(os.path.join(data_dir, 'images')):
        if image_file.endswith('.jpg'):
            img_path = os.path.join(data_dir, 'images', image_file)
            annotations_path = os.path.join(data_dir, 'labels', image_file.replace('.jpg', '.txt'))
            
            # Cargar la imagen
            images.append(img_path)
            
            # Cargar las etiquetas
            if os.path.exists(annotations_path):
                with open(annotations_path, 'r') as file:
                    boxes = []
                    for line in file.readlines():
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        boxes.append([class_id, x_center, y_center, width, height])  # [class_id, x, y, w, h]
                labels.append(boxes)
            else:
                labels.append([])  # Sin etiquetas

    return images, labels

# Función para preprocesar las imágenes y las etiquetas
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [300, 300])  # Cambia el tamaño según tus necesidades
    image = tf.cast(image, tf.float32) / 255.0  # Normalización

    # Convertir las etiquetas de YOLO a formato de píxeles
    height = image.shape[0]
    width = image.shape[1]
    boxes = []
    
    for box in label:
        class_id, x_center, y_center, box_width, box_height = box
        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)
        boxes.append([class_id, x_min, y_min, x_max, y_max])

    return image, boxes

# Función para crear el dataset de TensorFlow
def create_tf_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

# Cargar y preparar el dataset
data_dir = '../datasets_labeled/2024_10_24_canicas_dataset'  # Cambia a la ruta de tu dataset
images, labels = load_yolo_annotations(data_dir)
dataset = create_tf_dataset(images, labels)

# Ejemplo de impresión de una imagen y sus cajas
for img, boxes in dataset.take(1):
    print(f'Imagen: {img}, Cajas: {boxes}')
