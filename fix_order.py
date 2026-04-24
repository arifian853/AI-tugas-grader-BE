import asyncio
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    MONGO_URI = os.getenv("MONGODB_URI")
    DB_NAME = os.getenv("DB_NAME")
    
    db_client = AsyncIOMotorClient(MONGO_URI, tlsCAFile=certifi.where())
    db = db_client[DB_NAME]
    
    # Read name.csv
    df = pd.read_csv("name.csv")
    
    for idx, row in df.iterrows():
        nama = row['nama']
        await db.students.update_one({"nama": nama}, {"$set": {"order": idx}})
        await db.grades.update_one({"nama": nama}, {"$set": {"order": idx}})
        
    print("Database order updated successfully!")

if __name__ == "__main__":
    asyncio.run(main())
