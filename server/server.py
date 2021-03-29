import asyncio
import websockets
import lzstring
from databases import Database
import datetime

database = Database("sqlite:///data.db")

async def stream_client_data(websocket, path):
    print("test")
    identifier = await websocket.recv()
    print(f"< {identifier}")
    while True:
        message = await websocket.recv()
        await database.execute(query = "INSERT INTO Saves(userId, storeTime, compressedSaveData) VALUES (:userId, :storeTime, :compressedSaveData)", values =
            {"userId" : identifier, "storeTime" : datetime.datetime.now().isoformat(), "compressedSaveData" : message}
        )

async def main():
    await database.connect()
    await database.execute(query = "CREATE TABLE IF NOT EXISTS Saves(userId TEXT, storeTime TEXT, compressedSaveData TEXT)")
    await websockets.serve(stream_client_data, "localhost", 3006)

asyncio.get_event_loop().run_until_complete(main())
print("test2")
asyncio.get_event_loop().run_forever()