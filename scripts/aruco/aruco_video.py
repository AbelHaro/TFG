import cv2
import numpy as np
import os

video_path = '../datasets_labeled/videos/aruco_canicas.mp4'
output_path = '../inference_predictions/aruco/output_with_aruco.mp4'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
parameters.minCornerDistanceRate = 0.05

marker_size_cm = 3.527

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se puede abrir el video.")
else:
    print("Video cargado correctamente.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(f"Detected corners: {corners}")

    if ids is not None:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, corner in enumerate(corners):
            corners_array = corner[0]
            top_left = corners_array[0]
            top_right = corners_array[1]
            side_length_px = np.linalg.norm(top_left - top_right)
            px_to_cm_ratio = marker_size_cm / side_length_px
            marker_id = ids[i][0]

            text = f"ID: {marker_id} Size: {marker_size_cm:.2f} cm"
            position = (int(top_left[0]), int(top_left[1]) - 10)

            # Primero dibujar el borde del texto (stroke) en negro
            cv2.putText(
                frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4
            )  # Grosor del borde mayor

            # Luego dibujar el texto en verde encima
            cv2.putText(
                frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )  # Texto en verde con grosor normal
    else:
        # Texto para "No markers detected"
        text = "No markers detected"
        position = (10, 30)

        # Stroke para el texto en rojo
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
