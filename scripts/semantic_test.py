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
    user_email = structured_resume.get("email", "unknown")
    logger.info(f"[FAISS DEBUG] Starting build_faiss for resume: {user_email}")

    categories = ["work_experience", "projects", "education", "skills", "other"]
    cat_map = {cat: cat for cat in categories}

    for cat in categories:
        entries = structured_resume.get(cat, [])
        logger.info(f"[FAISS DEBUG] Processing category '{cat}' with {len(entries)} entries")

        if not entries:
            continue

        if cat == "skills" and isinstance(entries, list):
            text = "\n".join([str(s).strip() for s in entries if s])
            if text:
                docs.append(LC_Document(page_content=text, metadata={"category": cat_map[cat]}))
                logger.info(f"[FAISS INSERT] cat={cat_map[cat]}, snippet={text[:80]}")
            continue

        for i, entry in enumerate(entries):
            meta_cat = cat_map[cat]
            text = ""

            if cat == "projects" and isinstance(entry, dict):
                title_text = entry.get("project_title") or entry.get("title") or ""
                title_text = title_text.strip()
                highlights = entry.get("highlights", [])
                highlights_text = "\n".join([h.strip() for h in highlights if h.strip()])

                if title_text and highlights_text:
                    text = title_text + "\n" + highlights_text
                elif title_text:
                    text = title_text
                elif highlights_text:
                    text = highlights_text
                else:
                    text = None

            elif isinstance(entry, dict):
                text_fields = []
                for key in ["description", "role", "company", "degree", "school"]:
                    val = entry.get(key)
                    if val and isinstance(val, str):
                        val = val.strip()
                        if val:
                            text_fields.append(val)
                text = "\n".join(text_fields).strip() or None

            elif isinstance(entry, str):
                text = entry.strip() or None

            if not text:
                text = f"[{meta_cat} 未提供内容]"
                logger.warning(f"[FAISS WARN] Entry {i} in category '{cat}' is empty, using placeholder.")

            docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
            logger.info(f"[FAISS INSERT] cat={meta_cat}, snippet={text[:80]}")

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
