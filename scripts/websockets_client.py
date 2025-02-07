import asyncio
import websockets
import json
import time

async def send_messages():
    uri = "ws://10.236.45.32:8765"  # Conectar al servidor WebSocket
    async with websockets.connect(uri) as websocket:
        while True:
            # Crear un mensaje en formato JSON
            message = {
                "timestamp": time.time(),
                "client_id": "client_1",
                "message": "Hola, servidor"
            }
            
            # Convertir a JSON y enviar
            await websocket.send(json.dumps(message))

            # Recibir la respuesta del servidor
            response = await websocket.recv()
            print(f"Respuesta del servidor: {response}")

            await asyncio.sleep(1)  # Esperar 1 segundo antes de enviar el siguiente mensaje

asyncio.run(send_messages())
