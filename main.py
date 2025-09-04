import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel

from scripts.upload_llm import (
    read_docx_paragraphs,
    classify_paragraphs,
    build_faiss_with_category,
    summarize_full_category,
    query_dynamic_category,
    save_classification,
    load_classification,
    save_faiss,
    load_faiss,
)
from scripts.storage_client import StorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Analysis API")

# 请求体定义
class ResumeRequest(BaseModel):
    file_name: str

class QueryRequest(BaseModel):
    file_name: str
    query: str

@app.post("/analyze_resume")
def analyze_resume(req: ResumeRequest):
    """
    输入简历文件名（存储在 MinIO），下载并分析
    """
    client = StorageClient()
    downloads_dir = "./downloads"
    os.makedirs(downloads_dir, exist_ok=True)
    local_resume_path = os.path.join(downloads_dir, req.file_name)

    # 下载简历到本地
    client.read_file(req.file_name, local_resume_path)

    # 尝试加载已有分类
    classified = load_classification(req.file_name)
    if classified is None:
        # 没有则重新解析
        paragraphs = read_docx_paragraphs(local_resume_path)
        classified = classify_paragraphs(paragraphs)
        save_classification(req.file_name, classified)

    # 尝试加载已有 FAISS
    db = load_faiss(req.file_name)
    if db is None:
        db = build_faiss_with_category(classified)
        save_faiss(req.file_name, db)

    # 总结
    categories = ["WorkExperience", "Project", "Education", "Skills", "Other"]
    report = {cat: summarize_full_category(classified, cat) for cat in categories}

    return {"message": "简历分析完成", "report": report}

@app.post("/query_resume")
def query_resume(req: QueryRequest):
    """
    根据用户问题查询向量库（按 file_name 隔离）
    """
    db = load_faiss(req.file_name)
    if db is None:
        return {"error": f"没有找到 {req.file_name} 的向量库，请先调用 /analyze_resume"}

    result_text = query_dynamic_category(
        db, req.query, top_k=5, use_category_filter=True
    )
    return {"query": req.query, "answer": result_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
