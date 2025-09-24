import os
import logging
import json
import random
import datetime
from pymongo import MongoClient

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "ner_service",
            "message": record.getMessage(),
            "request_id": str(random.randint(1000, 9999))
        }
        return json.dumps(log)

# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)

# 设置日志 handler
handler = logging.FileHandler("logs/app.log")
handler.setFormatter(JsonFormatter())

logger = logging.getLogger()  # root logger
logger.addHandler(handler)
logger.setLevel(logging.INFO)

client = MongoClient("mongodb://localhost:27017/")
db = client["resume_db"]
collection = db["resumes"]

def save_resume(user_id: str, file_name: str, data: dict):
    doc = {
        "user_id": user_id,
        "file_name": file_name,
        "upload_time": datetime.datetime.utcnow(),
        "parsed_data": data,
    }
    collection.insert_one(doc)
    logger.info("Saved resume to MongoDB: %s", file_name)

def load_resume(user_id: str, file_name: str) -> dict | None:
    doc = collection.find_one({"user_id": user_id, "file_name": file_name})
    return doc["parsed_data"] if doc else None
