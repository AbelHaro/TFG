import cv2

# Ruta del video
video_path = '../datasets_labeled/videos/test/test_640x640.mp4'

# Abrir el video
cap = cv2.VideoCapture(video_path)

# Verificar si se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total de frames: {total_frames}")

# Ir al último frame (los frames empiezan en 0, así que el último es total_frames - 1)
frame_number = total_frames - 1
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()

if ret:
    # Guardar el frame como imagen
    output_image_path = 'frame_capturado.jpg'
    cv2.imwrite(output_image_path, frame)
    print(f"Frame {frame_number} guardado como {output_image_path}")
else:
    print("Error: No se pudo leer el frame.")

# Liberar el objeto de video
cap.release()
