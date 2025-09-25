import datetime
import json
import logging
import os
import random

from pymongo import MongoClient

from app.utils.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

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
