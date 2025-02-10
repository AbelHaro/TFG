import socket
import argparse
import threading

def tcp_client(host: str, port: int, message: str):
    try:
        # Crear un socket TCP
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Conectar al servidor
        client_socket.connect((host, port))
        print(f"Conectado a {host}:{port}")
        
        # Enviar mensaje
        client_socket.sendall(message.encode())
        
        # Recibir respuesta
        response = client_socket.recv(1024)
        print(f"Respuesta del servidor: {response.decode()}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cerrar conexión
        client_socket.close()
        print("Conexión cerrada.")

def tcp_server(host: str, port: int):
    try:
        # Crear un socket TCP
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"Escuchando en {host}:{port}...")

        while True:
            # Aceptar conexiones
            client_socket, client_address = server_socket.accept()
            print(f"Conexión recibida de {client_address}")
            
            # Recibir mensaje
            message = client_socket.recv(1024).decode()
            print(f"Mensaje recibido: {message}")
            
            # Enviar respuesta
            response = "Hola, cliente!"
            client_socket.sendall(response.encode())
            
            # Cerrar la conexión
            client_socket.close()
            print(f"Conexión con {client_address} cerrada.")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server_socket.close()
        print("Servidor cerrado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCP Client/Server")
    parser.add_argument('--mode', choices=['client', 'server'], required=True, help="Modo de operación: client o server")
    parser.add_argument('--host', default='0.0.0.0', help="Dirección IP del servidor (solo para cliente)")
    parser.add_argument('--port', type=int, default=8765, help="Puerto de conexión")
    parser.add_argument('--message', default="Hola, servidor!", help="Mensaje a enviar (solo para cliente)")

    args = parser.parse_args()

    if args.mode == 'client':
        tcp_client(args.host, args.port, args.message)
    elif args.mode == 'server':
        tcp_server(args.host, args.port)
