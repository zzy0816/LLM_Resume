import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LC_Document
from langchain_community.vectorstores import FAISS
from utils import normalize_category
from parser import CATEGORY_FIELDS
from doc import semantic_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# FAISS 构建
# -------------------------
def build_faiss(structured_resume: dict, embeddings_model=None):
    """
    根据段落列表构建 FAISS 向量数据库：
    - work_experience, education, other: 正常 semantic_split
    - projects: 保证每个项目整体入库（标题+内容），不切碎
    """
    docs = []

    for cat in ["work_experience", "projects", "education", "other"]:
        for entry in structured_resume.get(cat, []):
            if cat == "projects":
                # --- 保证项目整体 ---
                text = f"{entry.get('project_title','')}\n{entry.get('project_content','')}".strip()
                if not text:
                    continue
                meta_cat = normalize_category(cat)
                docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                logger.info("[FAISS INSERT] cat=%s, project_block=%s", meta_cat, text[:80])
            else:
                # --- 其他类正常拆分 ---
                fields_to_use = CATEGORY_FIELDS.get(cat, ["description"])
                text_fields = [str(entry.get(f, "")) for f in fields_to_use if entry.get(f)]
                if not text_fields:
                    text_fields = [str(entry.get("description", "")) or str(entry)]
                text = " ".join(text_fields).strip()
                if not text:
                    continue

                chunks = semantic_split(text)
                if not chunks:
                    chunks = [text]

                for sc in chunks:
                    meta_cat = normalize_category(cat)
                    docs.append(LC_Document(page_content=sc, metadata={"category": meta_cat}))
                    logger.info("[FAISS INSERT] cat=%s, snippet=%s", meta_cat, sc[:80])

    logger.info("Total docs to insert into FAISS: %d", len(docs))

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not docs:
        logger.warning("[FAISS WARN] no docs to insert into FAISS (docs list empty)")
        return None

    db = FAISS.from_documents(docs, embeddings_model)
    logger.info("FAISS database built with %d docs", len(docs))
    return db
