import asyncio
import websockets

async def handle_client(websocket, path):
    async for message in websocket:
        print(f"Cliente dice: {message}")
        await websocket.send(f"Mensaje recibido: {message}")

start_server = websockets.serve(handle_client, "0.0.0.0", 8765)  # Permite conexiones externas
    
asyncio.get_event_loop().run_until_complete(start_server)
print("Servidor WebSocket en ejecución en ws://localhost:8765")
asyncio.get_event_loop().run_forever()
