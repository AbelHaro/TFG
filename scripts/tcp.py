#XAVIER
import socket
import json

HOST = '0.0.0.0'  # Escucha en todas las interfaces
PORT = 5000       # Puerto arbitrario

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)  # Solo un cliente

print(f"Esperando conexión en {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print(f"Conectado a {addr}")

while True:
    # Simulación de datos a enviar
    mensaje = {
        "timestamp": 123456789,
        "sensor": "camara",
        "dato": "defecto_detectado"
    }
    
    # Convertir a JSON y enviar
    json_data = json.dumps(mensaje) + "\n"  # Agregar newline para delimitar mensajes
    conn.sendall(json_data.encode())

    # Recibir respuesta del Pico W
    data = conn.recv(1024)
    if not data:
        break
    respuesta = json.loads(data.decode())
    print("Respuesta del Pico:", respuesta)

conn.close()

################################################################################################
#RASPBERRY
import network
import socket
import json
import time

SSID = "tu_red_wifi"
PASSWORD = "tu_contraseña"

# Conectar a WiFi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)
while not wlan.isconnected():
    time.sleep(1)

print("Conectado a WiFi")

# Conectar al servidor TCP
HOST = "IP_DE_LA_JETSON"
PORT = 5000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while True:
    # Recibir datos en formato JSON
    data = s.recv(1024).decode()
    if not data:
        break

    json_data = json.loads(data)  # Convertir a diccionario
    print("Recibido:", json_data)

    # Responder con JSON
    respuesta = {
        "estado": "OK",
        "mensaje": "Datos recibidos"
    }
    s.sendall(json.dumps(respuesta).encode())

s.close()
