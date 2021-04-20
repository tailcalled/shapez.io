import asyncio
import websockets
import lzstring
from databases import Database
import datetime
import os

dbpath = "/data.db"
if os.getcwd().endswith("/server"):
    print("Sending data to local database")
    dbpath = "../../../data.db"

database = Database("sqlite://" + dbpath)

async def stream_client_data(websocket, path):
    print("test")
    identifier = await websocket.recv()
    print(f"< {identifier}")
    count = await database.fetch_all(query = "SELECT COUNT(*) FROM Saves WHERE userId = :userId", values={"userId" : identifier})
    print(f"Count: " + str(count))
    while True:
        events = await websocket.recv()
        message = await websocket.recv()
        print(events)
        await database.execute(query = "INSERT INTO Events(userId, storeTime, eventData) VALUES (:userId, :storeTime, :eventData)", values =
            {"userId" : identifier, "storeTime" : datetime.datetime.now().isoformat(), "eventData" : events}
        )
        await database.execute(query = "INSERT INTO Saves(userId, storeTime, compressedSaveData) VALUES (:userId, :storeTime, :compressedSaveData)", values =
            {"userId" : identifier, "storeTime" : datetime.datetime.now().isoformat(), "compressedSaveData" : message}
        )

async def main():
    await database.connect()
    await database.execute(query = "CREATE TABLE IF NOT EXISTS Saves(userId TEXT, storeTime TEXT, compressedSaveData TEXT)")
    await database.execute(query = "CREATE TABLE IF NOT EXISTS Events(userId TEXT, storeTime TEXT, eventData TEXT)")
    await websockets.serve(stream_client_data, "127.0.0.1", 3006)

asyncio.get_event_loop().run_until_complete(main())
print("test2")
asyncio.get_event_loop().run_forever()