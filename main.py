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
)
from scripts.storage_client import StorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Analysis API")

# 全局变量（可复用向量库）
faiss_db = None
classified = None


# 请求体定义
class ResumeRequest(BaseModel):
    file_name: str


class QueryRequest(BaseModel):
    query: str


@app.post("/analyze_resume")
def analyze_resume(req: ResumeRequest):
    """
    输入简历文件名（存储在 MinIO），下载并分析
    """
    global faiss_db, classified

    client = StorageClient()
    downloads_dir = r"D:\project\LLM_Resume\downloads"
    os.makedirs(downloads_dir, exist_ok=True)
    local_resume_path = os.path.join(downloads_dir, req.file_name)

    # 下载简历
    client.read_file(req.file_name, local_resume_path)

    # 读取 + 分类
    paragraphs = read_docx_paragraphs(local_resume_path)
    classified = classify_paragraphs(paragraphs)

    # 构建向量库
    faiss_db = build_faiss_with_category(classified)

    # 总结
    categories = ["WorkExperience", "Project", "Education", "Skills", "Other"]
    report = {}
    for cat in categories:
        report[cat] = summarize_full_category(classified, cat)

    return {"message": "简历分析完成", "report": report}


@app.post("/query_resume")
def query_resume(req: QueryRequest):
    """
    根据用户问题查询向量库
    """
    global faiss_db
    if faiss_db is None:
        return {"error": "请先调用 /analyze_resume 分析简历"}

    result_text = query_dynamic_category(
        faiss_db, req.query, top_k=5, use_category_filter=True
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
