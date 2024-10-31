import os
import random
import shutil
import zipfile
from pathlib import Path

# Función para dividir archivos en train, test y val
def split_data(file_list, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(file_list)

    n_total = len(file_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]

    return train_files, val_files, test_files

# Función para copiar archivos a las carpetas correspondientes
def copy_files(file_list, src_images_dir, src_labels_dir, dest_images_dir, dest_labels_dir):
    for file_name in file_list:
        # Generar rutas de origen
        src_image_path = os.path.join(src_images_dir, f"{file_name}.jpg")
        src_label_path = os.path.join(src_labels_dir, f"{file_name}.txt")

        # Verificar que los archivos existan antes de copiarlos
        if os.path.exists(src_image_path):
            # Generar rutas de destino
            dest_image_path = os.path.join(dest_images_dir, f"{file_name}.jpg")
            if os.path.exists(src_label_path):
                dest_label_path = os.path.join(dest_labels_dir, f"{file_name}.txt")

            # Copiar imagen y label
            shutil.copy(src_image_path, dest_image_path)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)
        else:
            print(f"Archivo {file_name} no encontrado en images o labels.")

# Función principal
def organize_data_from_zip(zip_path, output_dir):
    # Obtener el nombre base del archivo ZIP sin la extensión
    zip_name = Path(zip_path).stem
    extracted_dir = Path(output_dir) / zip_name
    new_extracted_dir = Path(output_dir) / f"{zip_name}_extracted"

    # Crear la carpeta extraída
    os.makedirs(new_extracted_dir, exist_ok=True)

    # Descomprimir el archivo ZIP en la carpeta con el nombre original
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

    # Definir rutas de las carpetas descomprimidas (asumimos que las carpetas images y labels están dentro de la raíz)
    images_dir = extracted_dir / "dataset/images"
    labels_dir = extracted_dir / "dataset/labels"

    # Asegurarse de que las carpetas images y labels existan
    if not images_dir.exists() or not labels_dir.exists():
        print("Las carpetas images o labels no existen en el archivo zip extraído.")
        return

    # Listar los archivos en la carpeta de imágenes
    image_files = [f.stem for f in images_dir.glob("*.jpg")]

    if not image_files:
        print("No se encontraron archivos de imágenes en la carpeta images.")
        return

    # Dividir los archivos en train, val y test
    train_files, val_files, test_files = split_data(image_files)

    # Crear las carpetas de destino para train, val y test dentro de la nueva carpeta _extracted
    for split in ['train', 'val', 'test']:
        os.makedirs(new_extracted_dir / split / "images", exist_ok=True)
        os.makedirs(new_extracted_dir / split / "labels", exist_ok=True)

    # Copiar los archivos correspondientes
    copy_files(train_files, images_dir, labels_dir, new_extracted_dir / 'train' / 'images', new_extracted_dir / 'train' / 'labels')
    copy_files(val_files, images_dir, labels_dir, new_extracted_dir / 'val' / 'images', new_extracted_dir / 'val' / 'labels')
    copy_files(test_files, images_dir, labels_dir, new_extracted_dir / 'test' / 'images', new_extracted_dir / 'test' / 'labels')

    print(f"Archivos organizados correctamente en {new_extracted_dir}.")

# Ejemplo de uso
zip_path = '../../../Descargas/dataset.zip'  # Ruta al archivo ZIP
output_dir = '../../../Descargas'  # Directorio donde se descomprimirá y organizará
organize_data_from_zip(zip_path, output_dir)
