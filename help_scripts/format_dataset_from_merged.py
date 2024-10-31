import os
import random
import shutil
from math import floor

# Rutas de las carpetas de origen y destino
source_images_dir = '../datasets_labeled/2024_10_24_canicas_dataset/images'
source_labels_dir = '../datasets_labeled/2024_10_24_canicas_dataset/labels'
dest_base_dir = '../datasets_labeled/2024_10_24_canicas_dataset'

# Proporciones
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Crear las carpetas destino si no existen
for subset in ['train', 'val', 'test']:
    os.makedirs(os.path.join(dest_base_dir, subset, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_base_dir, subset, 'labels'), exist_ok=True)

# Obtener todas las imágenes (.png)
images = [f for f in os.listdir(source_images_dir) if f.endswith('.png')]

# Barajar aleatoriamente las imágenes
random.shuffle(images)

# Calcular la cantidad de datos para cada subconjunto
total_images = len(images)
train_count = floor(train_ratio * total_images)
val_count = floor(val_ratio * total_images)
test_count = total_images - train_count - val_count

# Dividir los datos
train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

def copy_files(image_list, subset):
    """Función para copiar las imágenes y sus labels (si existen) a las carpetas correspondientes"""
    for img in image_list:
        # Definir las rutas de origen de la imagen y label
        img_src = os.path.join(source_images_dir, img)
        label_src = os.path.join(source_labels_dir, img.replace('.png', '.txt'))
        
        # Definir las rutas de destino
        img_dst = os.path.join(dest_base_dir, subset, 'images', img)
        label_dst = os.path.join(dest_base_dir, subset, 'labels', img.replace('.png', '.txt'))
        
        # Copiar la imagen
        shutil.copy(img_src, img_dst)
        
        # Si el archivo de label existe, también copiarlo
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)

# Copiar los archivos a las carpetas correspondientes
copy_files(train_images, 'train')
copy_files(val_images, 'val')
copy_files(test_images, 'test')

print(f'Total imágenes: {total_images}')
print(f'Train: {train_count}, Val: {val_count}, Test: {test_count}')
