import os
import glob
import shutil
import random
import zipfile
from PIL import Image

def unzip_dataset(zip_file_path, extract_to_folder):
    """
    Descomprime un archivo ZIP en la carpeta indicada.
    
    :param zip_file_path: Ruta del archivo ZIP.
    :param extract_to_folder: Carpeta donde se extraerán los archivos.
    """
    # Eliminar el directorio de destino si existe
    if os.path.exists(extract_to_folder):
        shutil.rmtree(extract_to_folder)
        print(f"Directorio existente eliminado: {extract_to_folder}")
    
    # Descomprimir el archivo ZIP
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
    print(f"Descomprimido {zip_file_path} en {extract_to_folder}")

def process_image(input_path, output_path, quality, size=None):
    """
    Disminuye la calidad de una imagen y opcionalmente la redimensiona.

    :param input_path: Ruta a la imagen de entrada.
    :param output_path: Ruta de salida de la imagen procesada.
    :param quality: Nivel de calidad de la imagen (1-100).
    :param size: Tupla (ancho, alto) para redimensionar la imagen. Si es None, no se redimensiona.
    """
    img = Image.open(input_path)
    if size:
        img = img.resize(size)  # Resize the image if a new size is provided
    img.save(output_path, quality=quality)

    # Compare sizes
    original_size = os.path.getsize(input_path)
    processed_size = os.path.getsize(output_path)
    
    print(f"Processed {os.path.basename(input_path)}")
    print(f"Original size: {original_size / 1024:.2f} KB")
    print(f"Processed size: {processed_size / 1024:.2f} KB\n")

def create_folders_and_yaml(folder_path):
    """
    Crea las carpetas necesarias ('train', 'val', 'test') con subcarpetas 'images' y 'labels'.
    Además, genera el archivo 'data.yml'.

    :param folder_path: Ruta donde se crearán las carpetas y el archivo YAML.
    """
    # Crear carpetas de entrenamiento, validación y prueba
    for split in ['train', 'val', 'test']:
        images_folder = os.path.join(folder_path, split, 'images')
        labels_folder = os.path.join(folder_path, split, 'labels')
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

    # Crear el archivo data.yml
    yaml_content = f"nc: 4
                    names:
                    0: negra
                    1: blanca
                    2: verde
                    3: azul
                    train: ./train
                    val: ./val
                    test: ./test"
                    
    train: {os.path.join(folder_path, 'train/images')}
    val: {os.path.join(folder_path, 'val/images')}
    test: {os.path.join(folder_path, 'test/images')}

    with open(os.path.join(folder_path, 'data.yml'), 'w') as yaml_file:
        yaml_file.write(yaml_content)
    
    print(f"Archivo data.yml creado en {folder_path}")

def split_dataset(folder_path, quality, size=None):
    """
    Divide un conjunto de imágenes en tres subconjuntos: entrenamiento, validación y prueba.
    Aplica disminución de calidad, redimensionado y renombra las imágenes de forma secuencial.

    :param folder_path: Ruta a la carpeta que contiene las imágenes.
    :param quality: Nivel de calidad para las imágenes (1-100).
    :param size: Tupla (ancho, alto) para redimensionar la imagen. Si es None, se mantiene el tamaño original.
    """
    # Crear las carpetas necesarias y el archivo YAML
    create_folders_and_yaml(folder_path)

    # Obtener todas las imágenes .jpg en la carpeta
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))

    # Barajar las imágenes
    random.shuffle(image_files)

    # Calcular los índices para dividir las imágenes
    total_images = len(image_files)
    train_end = int(total_images * 0.8)
    val_end = train_end + int(total_images * 0.1)

    # Dividir las imágenes en entrenamiento, validación y prueba
    train_images = image_files[:train_end]
    val_images = image_files[train_end:val_end]
    test_images = image_files[val_end:]

    # Procesar y mover las imágenes renombradas a sus respectivas carpetas
    for idx, image in enumerate(train_images, start=1):
        new_name = f"train{idx}.jpg"
        output_image_path = os.path.join(folder_path, 'train/images', new_name)
        process_image(image, output_image_path, quality, size)

    for idx, image in enumerate(val_images, start=1):
        new_name = f"val{idx}.jpg"
        output_image_path = os.path.join(folder_path, 'val/images', new_name)
        process_image(image, output_image_path, quality, size)

    for idx, image in enumerate(test_images, start=1):
        new_name = f"test{idx}.jpg"
        output_image_path = os.path.join(folder_path, 'test/images', new_name)
        process_image(image, output_image_path, quality, size)

    print(f"Total imágenes: {total_images}")
    print(f"Entrenamiento: {len(train_images)}")
    print(f"Validación: {len(val_images)}")
    print(f"Prueba: {len(test_images)}")

    # Eliminar las imágenes originales que están en el directorio raíz
    for image in image_files:
        if os.path.exists(image):
            os.remove(image)
            print(f"Eliminado {image}")

# Ejemplo de uso
zip_file_path = '../../datasets/2024_10_02_dataset.zip'  # Ruta al archivo ZIP
extracted_folder_path = '../../datasets/2024_10_02_canicas_dataset'  # Carpeta de extracción

# Descomprimir el archivo ZIP
unzip_dataset(zip_file_path, extracted_folder_path)

# Dividir y procesar el dataset, renombrar imágenes, y generar las carpetas y el archivo YAML
split_dataset(extracted_folder_path, quality= 100, size=(640, 640))
