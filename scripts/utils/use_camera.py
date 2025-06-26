import cv2

# Intenta abrir la cámara
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

try:
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        exit()

    # Establece la resolución de la cámara a 640x640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while True:
        # Captura un frame de la cámara
        ret, frame = cap.read()

        if not ret:
            print("Error: No se puede recibir un frame de la cámara.")
            break

        # Muestra el frame en una ventana
        cv2.imshow('Camera Feed', frame)

        # Sale del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Libera la cámara y cierra todas las ventanas al finalizar
    cap.release()
    cv2.destroyAllWindows()
