import socket
import argparse


def handle_send(client_socket, message: str):
    """Enviar un único mensaje"""
    try:
        client_socket.sendall(message.encode())

    except Exception as e:
        client_socket.close()
        print(f"Error en el envío: {e}")


def tcp_server(host: str, port: int):
    """Servidor TCP que acepta una única conexión a la vez y vuelve a estar disponible tras cierre."""

    print("[TCP_SERVER] Inicializando servidor, esperando conexión...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Servidor escuchando en {host}:{port}...")

    while True:
        try:
            client_socket, client_address = server_socket.accept()
            print(f"Conexión recibida de {client_address}")

            message = client_socket.recv(1024).decode()
            print(f"Mensaje recibido: {message}")

            return client_socket, server_socket

        except Exception as e:
            print(f"Error en el servidor: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCP Full-Duplex Server")
    parser.add_argument('--host', default='0.0.0.0', help="Dirección IP del servidor")
    parser.add_argument('--port', type=int, default=8765, help="Puerto de conexión")
    args = parser.parse_args()

    tcp_server(args.host, args.port)
