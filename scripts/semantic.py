import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LC_Document
from langchain_community.vectorstores import FAISS
from utils import normalize_category

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
    docs = []

    for cat in ["work_experience", "projects", "education", "other"]:
        for entry in structured_resume.get(cat, []):
            meta_cat = normalize_category(cat)
            
            # projects 整体入库
            if cat == "projects":
                text = f"{entry.get('project_title','')}\n{entry.get('project_content','')}".strip()
                if not text:
                    continue
                docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                logger.info("[FAISS INSERT] cat=%s, project_block=%s", meta_cat, text[:80])
            else:
                # education / work / other 整段入库，不拆分
                text = entry.get("description", "").strip()
                if not text:
                    continue
                docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                logger.info("[FAISS INSERT] cat=%s, snippet=%s", meta_cat, text[:80])

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not docs:
        logger.warning("[FAISS WARN] no docs to insert into FAISS (docs list empty)")
        return None

    db = FAISS.from_documents(docs, embeddings_model)
    logger.info("FAISS database built with %d docs", len(docs))
    return db
