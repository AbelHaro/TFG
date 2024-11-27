import cv2
import numpy as np

video_path = '../datasets_labeled/videos/video_with_aruco.mp4'

# Definir el diccionario ArUco y el ID del marcador de referencia
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# Tamaño físico conocido del marcador de referencia (en cm)
marker_size_cm = 5.0

# Cargar el video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar marcadores ArUco
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Dibujar los marcadores detectados
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):
            # Obtener las esquinas del marcador detectado
            corners_array = corner[0]
            top_left = corners_array[0]
            top_right = corners_array[1]

            # Calcular la longitud del lado del marcador en píxeles
            side_length_px = np.linalg.norm(top_left - top_right)

            # Convertir de píxeles a tamaño físico
            px_to_cm_ratio = marker_size_cm / side_length_px
            size_cm = side_length_px * px_to_cm_ratio  # Confirmar dimensiones reales

            # Mostrar el tamaño estimado en la imagen
            marker_id = ids[i][0]
            cv2.putText(frame, f"ID: {marker_id} Size: {marker_size_cm:.2f} cm",
                        (int(top_left[0]), int(top_left[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame procesado
    cv2.imshow('Aruco Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
