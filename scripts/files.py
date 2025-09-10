import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CLASSIFIED_DIR = "./data/classified"
FAISS_DIR = "./data/faiss"
os.makedirs(CLASSIFIED_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# -------------------------
# 文件保存/加载
# -------------------------
def save_json(file_name: str, data: dict):
    path = os.path.join(CLASSIFIED_DIR, f"{file_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved JSON to %s", path)

def load_json(file_name: str) -> dict | None:
    path = os.path.join(CLASSIFIED_DIR, f"{file_name}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# -------------------------
# FAISS 保存 & 加载
# -------------------------
def save_faiss(file_name: str, db: FAISS):
    save_path = os.path.join(FAISS_DIR, file_name)
    os.makedirs(save_path, exist_ok=True)
    db.save_local(save_path)
    logger.info("Saved FAISS db to %s", save_path)

def load_faiss(file_name: str, embeddings_model=None) -> FAISS | None:
    save_path = os.path.join(FAISS_DIR, file_name)
    if os.path.exists(save_path):
        if embeddings_model is None:
            embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(save_path, embeddings_model, allow_dangerous_deserialization=True)
    return None

# -------------------------
# embed 保存 & 加载
# -------------------------
import numpy as np

def save_embeddings(file_name, embs):
    np.save(f"./data/classified/{file_name}_embs.npy", embs)

def load_embeddings(file_name):
    path = f"./data/classified/{file_name}_embs.npy"
    if os.path.exists(path):
        return np.load(path)
    return None
