import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define el diccionario ArUco que queremos usar
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
marker_id = 0  # ID del marcador

# Tamaño total del marcador en píxeles (incluyendo el borde)
final_size = 100  

# Tamaño del marcador sin borde (celda interna)
# Debemos dividir en la cantidad de celdas internas y considerar el borde
border_size = 1  # Tamaño del borde en celdas (default)
marker_size = final_size - 2 * border_size * (final_size // (4 + 2 * border_size))

# Generar el marcador
marker_image = np.zeros((final_size, final_size), dtype=np.uint8)
cv2.aruco.drawMarker(aruco_dict, marker_id, final_size, marker_image)

# Guardar la imagen
cv2.imwrite('aruco_100x100.png', marker_image)

print(f"Marcador guardado en 'aruco_100x100.png' con ID {marker_id} y tamaño {marker_size}x{marker_size} píxeles")