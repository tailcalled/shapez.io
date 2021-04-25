import asyncio
import websockets
import lzstring
from databases import Database
import datetime
import os
import sys

dbpath = "/data.db"
if os.getcwd().endswith("/server"):
    print("Sending data to local database")
    dbpath = "../../../data.db"

database = Database("sqlite://" + dbpath)

async def stream_client_data(websocket, path):
    print("test", flush=True)
    identifier = await websocket.recv()
    print(f"< {identifier}", flush=True)
    count = await database.fetch_all(query = "SELECT COUNT(*) FROM Saves WHERE userId = :userId", values={"userId" : identifier})
    print(f"Count: " + str(count), flush=True)
    while True:
        msg_id = await websocket.recv()
        msg_payload = await websocket.recv()
        await websocket.send(msg_id)
        if msg_payload.startswith("EVENT:"):
            event = msg_payload[len("EVENT:"):]
            print(event, flush=True)
            try:
                await database.execute(query = "INSERT INTO Events(userId, storeTime, eventData) VALUES (:userId, :storeTime, :eventData)", values =
                    {"userId" : identifier, "storeTime" : datetime.datetime.now().isoformat(), "eventData" : event}
                )
            except:
                print("Error: ", sys.exc_info()[0], flush=True)
        elif msg_payload.startswith("SAVE:"):
            save_data = msg_payload[len("SAVE:"):]
            print("sent save data", flush=True)
            try:
                await database.execute(query = "INSERT INTO Saves(userId, storeTime, compressedSaveData) VALUES (:userId, :storeTime, :compressedSaveData)", values =
                    {"userId" : identifier, "storeTime" : datetime.datetime.now().isoformat(), "compressedSaveData" : save_data}
                )
            except:
                print("Error: ", sys.exc_info()[0], flush=True)

async def main():
    await database.connect()
    await database.execute(query = "CREATE TABLE IF NOT EXISTS Saves(userId TEXT, storeTime TEXT, compressedSaveData TEXT)")
    await database.execute(query = "CREATE TABLE IF NOT EXISTS Events(userId TEXT, storeTime TEXT, eventData TEXT)")
    await websockets.serve(stream_client_data, "127.0.0.1", 3006)

asyncio.get_event_loop().run_until_complete(main())
print("test2", flush=True)
asyncio.get_event_loop().run_forever()