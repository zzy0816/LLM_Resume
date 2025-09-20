import logging
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LC_Document
from langchain_community.vectorstores import FAISS
from utils import normalize_category

logger = logging.getLogger(__name__)
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_faiss(structured_resume: dict, embeddings_model=None):
    docs = []

    logger.info(f"[FAISS DEBUG] Starting build_faiss for resume: {structured_resume.get('email')}")

    for cat in ["work_experience", "projects", "education", "other"]:
        entries = structured_resume.get(cat, [])
        logger.info(f"[FAISS DEBUG] Processing category '{cat}' with {len(entries)} entries")

        for i, entry in enumerate(entries):
            meta_cat = normalize_category(cat)
            logger.info(f"[FAISS DEBUG] Entry {i} raw content: {entry}")

            if isinstance(entry, dict):
                if cat == "projects":
                    # 尝试多种字段名
                    text = (entry.get("project_title") or entry.get("title") or "") + "\n" + \
                           (entry.get("project_content") or entry.get("highlights", "") or "")
                    text = text.strip()
                    if not text:
                        logger.warning(f"[FAISS WARN] Empty project text for entry {i} in category '{cat}'")
                        continue
                    docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                    logger.info(f"[FAISS INSERT] cat={meta_cat}, text_preview={text[:80]}")
                else:
                    text_fields = []
                    for key in ["description", "role", "company", "degree", "school"]:
                        val = entry.get(key)
                        if val and isinstance(val, str):
                            text_fields.append(val.strip())
                    text = "\n".join(text_fields)
                    if not text:
                        logger.warning(f"[FAISS WARN] Empty text for entry {i} in category '{cat}'")
                        continue
                    docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                    logger.info(f"[FAISS INSERT] cat={meta_cat}, snippet={text[:80]}")
            elif isinstance(entry, str):
                text = entry.strip()
                if not text:
                    logger.warning(f"[FAISS WARN] Empty string entry {i} in category '{cat}'")
                    continue
                docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                logger.info(f"[FAISS INSERT] cat={meta_cat}, snippet={text[:80]}")
            else:
                logger.warning(f"[FAISS WARN] Unknown entry type {type(entry)} for entry {i} in category '{cat}'")

    if not docs:
        logger.warning("[FAISS WARN] No docs generated, FAISS DB will be empty")
        return None

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings_model)
    logger.info(f"[FAISS INFO] FAISS database built with {len(docs)} docs")
    return db

if __name__ == "__main__":
    import pprint

    # 模拟一个结构化简历
    test_resume = {
        "name": "Zhenyu Zhang",
        "email": "Zhang.zhenyu6@northeastern.edu",
        "phone": "+18602347101",
        "work_experience": [
            {"company": "OpenAI", "title": "Research Scientist", "description": "Worked on LLM research | Jan 2023 - Present"}
        ],
        "projects": [
            {"project_title": "Recommendation System", "project_content": "Built a recommendation system using PyTorch and Python"}
        ],
        "education": [
            {"school": "Northeastern University", "degree": "Master", "grad_date": "2025", "description": "Studied AI and ML"}
        ],
        "skills": ["Python", "PyTorch", "TensorFlow", "SQL", "Pandas"],
        "other": ["Volunteer at local community center"]
    }

    print("[TEST] Starting FAISS build test...")
    db = build_faiss(test_resume)

    if db:
        # 方法1：用 docstore 长度
        # print("[TEST] FAISS build successful, number of docs:", len(db.docstore))

        # 方法2：也可以用 index_to_docstore_id
        print("[TEST] FAISS build successful, number of docs:", len(db.index_to_docstore_id))
    else:
        print("[TEST] FAISS build returned None")

    # 查询测试
    test_query = "work_experience"
    from query import query_dynamic_category
    results = query_dynamic_category(db, test_resume, test_query, top_k=3)
    pprint.pprint(results)
