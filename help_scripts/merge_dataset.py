import os
import shutil

# Rutas a las carpetas
base_path = '../../datasets/2024_10_02_canicas_dataset'  # Reemplaza con la ruta correcta
folders = ['train', 'test', 'val']
output_path = os.path.join(base_path, 'merged')
images_output = os.path.join(output_path, 'images')
labels_output = os.path.join(output_path, 'labels')

# Crear carpetas de salida
os.makedirs(images_output, exist_ok=True)
os.makedirs(labels_output, exist_ok=True)

# Contador para renombrar los archivos
counter = 1

# Función para copiar y renombrar los archivos
def process_folder(folder_path):
    global counter
    images_path = os.path.join(folder_path, 'images')
    labels_path = os.path.join(folder_path, 'labels')
    
    # Obtener las listas de imágenes y etiquetas
    image_files = sorted(os.listdir(images_path))
    label_files = sorted(os.listdir(labels_path))
    
    for img_file in image_files:
        img_name = os.path.splitext(img_file)[0]  # Nombre sin extensión
        label_file = f"{img_name}.txt"  # Archivo de etiqueta correspondiente

        # Verificar si el archivo de etiqueta existe
        if label_file in label_files:
            # Definir nuevos nombres
            new_img_name = f"{counter}.jpg"  # Cambia la extensión si es necesario
            new_label_name = f"{counter}.txt"
            
            # Rutas completas de los archivos de salida
            new_img_path = os.path.join(images_output, new_img_name)
            new_label_path = os.path.join(labels_output, new_label_name)
            
            # Copiar los archivos
            shutil.copy(os.path.join(images_path, img_file), new_img_path)
            shutil.copy(os.path.join(labels_path, label_file), new_label_path)
            
            print(f"Copiado {img_file} y {label_file} como {new_img_name} y {new_label_name}")
            
            # Incrementar el contador
            counter += 1
        else:
            print(f"Advertencia: No se encontró etiqueta para {img_file}")

# Procesar todas las carpetas (train, test, val)
for folder in folders:
    process_folder(os.path.join(base_path, folder))

print(f"Proceso completado. Total de archivos procesados: {counter - 1}")
