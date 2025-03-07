import asyncio
import websockets
import json


async def handle_client(websocket, path):
    async for message in websocket:
        data = json.loads(message)  # Convertir de JSON a diccionario
        print(f"Cliente {data['client_id']} dice: {data['message']}")

        # Responder con JSON
        response = {"status": "OK", "received": data}
        await websocket.send(json.dumps(response))


start_server = websockets.serve(handle_client, "0.0.0.0", 8765)  # Permite conexiones externas

asyncio.get_event_loop().run_until_complete(start_server)
print("Servidor WebSocket en ejecución en ws://localhost:8765")
asyncio.get_event_loop().run_forever()
