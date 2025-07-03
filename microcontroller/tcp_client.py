import socket
import network
import time
import machine
from servo_controller import setup_servo, move_servo

def run_tcp_client():
    # Configurar la conexión Wi-Fi
    ssid = 'UPV-PSK'
    password = 'R4sb3rr4ndo!'

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    print(f"[DEBUG] wlan object: {wlan}")
    print(f"[DEBUG] wlan.active(): {wlan.active()}")
    print(f"[DEBUG] wlan.isconnected(): {wlan.isconnected()}")
    print("[WIFI] Intentando conectar...")

    wlan.connect(ssid, password)

    attempt = 0
    max_attempts = 100000

    while not wlan.isconnected() and attempt < max_attempts:
        print(f"[WIFI] Conectando a Wi-Fi... intento {attempt + 1}")
        print(f"[DEBUG] wlan.status(): {wlan.status()}")
        print(f"[DEBUG] wlan.isconnected(): {wlan.isconnected()}")
        time.sleep(1)
        attempt += 1

    if wlan.isconnected():
        print('[WIFI] ✅ Conectado a Wi-Fi:', wlan.ifconfig())
    else:
        print('[WIFI] ❌ No se pudo conectar a la red Wi-Fi.')
        return

    # Configurar el cliente TCP
    SERVER_IP = '158.42.215.223'
    SERVER_PORT = 8765

    led = machine.Pin("LED", machine.Pin.OUT)
    servo = setup_servo()

    # Intentar conectar al servidor TCP
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((SERVER_IP, SERVER_PORT))
            print(f"[TCP_CLIENT] ✅ Conectado al servidor {SERVER_IP}:{SERVER_PORT}")
            break
        except OSError as e:
            print(f"[TCP_CLIENT] ❌ Error de conexión TCP: {e}. Reintentando en 5 segundos...")
            time.sleep(5)

    try:
        # Enviar un mensaje inicial al servidor
        message = "Hola desde Pico W"
        client_socket.send(message.encode())

        while True:
            response = client_socket.recv(1024)
            if not response:
                break

            response_str = response.decode().strip()
            print(f"[TCP_CLIENT] Respuesta del servidor: {response_str}")

            if response_str:
                led.on()
                move_servo(servo)
                time.sleep(1)
                led.off()

        print("[TCP_CLIENT] Conexión cerrada")

    except Exception as e:
        print(f"[TCP_CLIENT] ❌ Error durante la comunicación: {e}")

    finally:
        client_socket.close()
