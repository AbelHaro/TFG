import os
import zipfile
import shutil
import random

def unzip_file(zip_file, dest_dir):
    """Descomprime el archivo ZIP en el directorio destino."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

def create_directories(base_dir):
    """Crea las carpetas necesarias para train, val y test."""
    os.makedirs(os.path.join(base_dir, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val/labels'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test/images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test/labels'), exist_ok=True)

def split_data(images_dir, labels_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Divide los archivos en los conjuntos de datos train, val y test."""
    # Verificar los archivos disponibles en las carpetas
    print("Archivos en images:", os.listdir(images_dir))
    print("Archivos en labels:", os.listdir(labels_dir))

    images = sorted(os.listdir(images_dir))
    labels = sorted(os.listdir(labels_dir))

    # Verifica que el número de imágenes coincida con el número de etiquetas
    if len(images) != len(labels):
        raise ValueError("El número de imágenes no coincide con el número de etiquetas")

    # Mezcla las imágenes y etiquetas de forma aleatoria
    combined = list(zip(images, labels))
    random.shuffle(combined)

    # Reparte los datos en train, val y test
    total_files = len(combined)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    train_files = combined[:train_size]
    val_files = combined[train_size:train_size + val_size]
    test_files = combined[train_size + val_size:]

    return train_files, val_files, test_files

def copy_files(files, src_images_dir, src_labels_dir, dest_images_dir, dest_labels_dir):
    """Copia los archivos en las carpetas correspondientes."""
    for image, label in files:
        shutil.copy(os.path.join(src_images_dir, image), os.path.join(dest_images_dir, image))
        shutil.copy(os.path.join(src_labels_dir, label), os.path.join(dest_labels_dir, label))

def main(zip_file):
    # Definir el directorio de destino
    dest_dir = zip_file.replace('.zip', '')

    # Crear directorios
    create_directories(dest_dir)

    # Descomprimir el archivo ZIP
    unzip_file(zip_file, dest_dir)

    # Directorios de imágenes y etiquetas
    images_dir = os.path.join(dest_dir, 'images')
    labels_dir = os.path.join(dest_dir, 'labels')

    # Dividir los datos en train, val y test
    train_files, val_files, test_files = split_data(images_dir, labels_dir)

    # Copiar los archivos en las carpetas correspondientes
    copy_files(train_files, images_dir, labels_dir, os.path.join(dest_dir, 'train/images'), os.path.join(dest_dir, 'train/labels'))
    copy_files(val_files, images_dir, labels_dir, os.path.join(dest_dir, 'val/images'), os.path.join(dest_dir, 'val/labels'))
    copy_files(test_files, images_dir, labels_dir, os.path.join(dest_dir, 'test/images'), os.path.join(dest_dir, 'test/labels'))

    print("Datos distribuidos correctamente.")

if __name__ == "__main__":
    # Asegúrate de pasar el archivo ZIP como argumento
    zip_file = "../datasets_labeled/2024_10_21_canicas.zip"
    main(zip_file)
