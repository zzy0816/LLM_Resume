import logging
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LC_Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------
# FAISS 构建
# ------------------------
def build_faiss(structured_resume: dict, embeddings_model=None):
    docs = []
    user_email = structured_resume.get("email", "unknown")
    logger.info(f"[FAISS DEBUG] Starting build_faiss for resume: {user_email}")

    categories = ["work_experience", "projects", "education", "skills", "other"]

    for cat in categories:
        entries = structured_resume.get(cat, [])
        if not entries:
            continue

        for i, entry in enumerate(entries):
            text = ""
            meta = {"category": cat}

            # 文本构建
            if cat == "projects" and isinstance(entry, dict):
                title = entry.get("project_title") or entry.get("title") or ""
                highlights = "\n".join(entry.get("highlights", []))
                text = "\n".join([title, highlights]).strip()
            elif isinstance(entry, dict):
                text = entry.get("description") or ""
                # 保存原始结构化字段到 metadata
                for k in ["company","position","location","start_date","end_date",
                          "school","degree","grad_date"]:
                    if k in entry:
                        meta[k] = entry[k]
            elif isinstance(entry, str):
                text = entry.strip()

            if not text:
                text = f"[{cat} 未提供内容]"

            docs.append(LC_Document(page_content=text, metadata=meta))

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings_model)
    logger.info(f"[FAISS INFO] FAISS DB built with {len(docs)} docs")
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
    from app.pipline.query import query_dynamic_category
    results = query_dynamic_category(db, test_resume, test_query, top_k=3)
    pprint.pprint(results)
