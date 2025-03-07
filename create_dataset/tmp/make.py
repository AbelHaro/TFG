import os
import shutil
import random
from PIL import Image  # Importar Pillow para conversión de imágenes


def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Convertir rutas a absolutas para evitar confusiones
    image_dir = os.path.abspath(image_dir)
    label_dir = os.path.abspath(label_dir)
    output_dir = os.path.abspath(output_dir)

    print(f"Imagenes en: {image_dir}")
    print(f"Labels en: {label_dir}")
    print(f"Salida en: {output_dir}")

    # Crear directorios de salida
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Obtener lista de imágenes (.jpg)
    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    print(f"Total de imágenes encontradas: {len(images)}")

    if not images:
        print("ERROR: No se encontraron imágenes en la carpeta.")
        return

    random.shuffle(images)

    # Calcular tamaños de cada conjunto
    total = len(images)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # Dividir imágenes
    train_images = images[:train_size]
    val_images = images[train_size : train_size + val_size]
    test_images = images[train_size + val_size :]

    # Función para convertir y mover archivos
    def process_files(file_list, split):
        for img in file_list:
            img_path = os.path.join(image_dir, img)
            new_img_name = img.replace(".jpg", ".png")  # Cambia extensión a .png
            new_img_path = os.path.join(output_dir, split, "images", new_img_name)
            label_path = os.path.join(label_dir, img.replace(".jpg", ".txt"))

            print(f"Procesando: {img_path} → {new_img_path}")
            print(f"Etiqueta esperada: {label_path}")

            # Convertir JPG a PNG
            try:
                with Image.open(img_path) as im:
                    im.save(new_img_path, "PNG")
                os.remove(img_path)  # Eliminar el JPG original
                print(f"✅ Convertido y guardado: {new_img_name}")
            except Exception as e:
                print(f"❌ ERROR al convertir {img}: {e}")

            # Mover etiqueta
            if os.path.exists(label_path):
                shutil.move(
                    label_path,
                    os.path.join(output_dir, split, "labels", os.path.basename(label_path)),
                )
                print(f"✅ Movido: {os.path.basename(label_path)} a {split}/labels/")
            else:
                print(f"⚠️ Advertencia: No se encontró etiqueta para {img}")

    # Procesar archivos
    process_files(train_images, "train")
    process_files(val_images, "val")
    process_files(test_images, "test")


# Parámetros de entrada
image_dir = "./images"
label_dir = "./labels"
output_dir = "./dataset"

# Ejecutar script
split_dataset(image_dir, label_dir, output_dir)
