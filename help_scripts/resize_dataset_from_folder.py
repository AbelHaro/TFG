import os
import glob
import zipfile
from PIL import Image

def process_image(input_path, output_path, size=(640, 640)):
    """
    Redimensiona una imagen a un tamaño dado.

    :param input_path: Ruta a la imagen de entrada.
    :param output_path: Ruta de salida de la imagen procesada.
    :param size: Tupla (ancho, alto) para redimensionar la imagen.
    """
    try:
        img = Image.open(input_path)
        img = img.resize(size)  # Redimensionar la imagen a 640x640
        img.save(output_path)

        # Comparar tamaños de archivos
        original_size = os.path.getsize(input_path)
        processed_size = os.path.getsize(output_path)
        
        print(f"Processed {os.path.basename(input_path)}")
        print(f"Original size: {original_size / 1024:.2f} KB")
        print(f"Processed size: {processed_size / 1024:.2f} KB\n")
    except Exception as e:
        print(f"Error procesando la imagen {input_path}: {e}")

def process_images_in_folder(folder_path, output_folder, size=(640, 640)):
    """
    Procesa todas las imágenes en una carpeta, redimensionándolas a 640x640.

    :param folder_path: Ruta de la carpeta que contiene las imágenes.
    :param output_folder: Ruta donde se guardarán las imágenes procesadas.
    :param size: Tupla (ancho, alto) para redimensionar las imágenes.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Obtener todas las imágenes .jpg en la carpeta y subcarpetas
    image_files = glob.glob(os.path.join(folder_path, '**', '*.jpg'), recursive=True)

    # Procesar cada imagen
    for image_file in image_files:
        # Crear la estructura de carpetas en la carpeta de salida
        relative_path = os.path.relpath(image_file, folder_path)
        output_image_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        
        process_image(image_file, output_image_path, size)

    print(f"Total imágenes procesadas: {len(image_files)}")

# Ejemplo de uso
extracted_folder_path = '../../../Descargas/impostores'  # Carpeta de extracción
output_folder = '../../../Descargas/impostores_formatted'  # Carpeta donde se guardarán las imágenes procesadas

# Descomprimir el archivo ZIP
#unzip_dataset(zip_file_path, extracted_folder_path)

# Procesar las imágenes, redimensionándolas a 640x640
process_images_in_folder(extracted_folder_path, output_folder, size=(640, 640))
