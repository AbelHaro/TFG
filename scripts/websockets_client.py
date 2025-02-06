import asyncio
import websockets

async def send_messages():
    uri = "ws://localhost:8765"  # Conectar al servidor WebSocket
    async with websockets.connect(uri) as websocket:
        while True:
            await websocket.send("Hola, servidor")
            response = await websocket.recv()
            print(f"Respuesta del servidor: {response}")
            await asyncio.sleep(1)  # Esperar 1 segundo antes de enviar el siguiente mensaje

asyncio.run(send_messages())
