import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel

from scripts.upload_llm import (
    read_docx_paragraphs,
    parse_resume_to_structured,
    build_faiss,
    query_dynamic_category,
    save_json,
    load_json,
    save_faiss,
    load_faiss,
    main_pipeline
)
from scripts.storage_client import StorageClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Analysis API")

# -------------------------
# 请求体定义
# -------------------------
class ResumeRequest(BaseModel):
    file_name: str

class QueryRequest(BaseModel):
    file_name: str
    query: str

# -------------------------
# 分析简历接口
# -------------------------
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

    # 尝试加载已有结构化 JSON
    structured_resume = main_pipeline(req.file_name, mode="exact")

    # 生成报告（按类别汇总）
    report = {
        "WorkExperience": structured_resume.get("work_experience", []),
        "Project": structured_resume.get("projects", []),
        "Education": structured_resume.get("education", []),
        "Skills": structured_resume.get("skills", [])
    }

    return {"message": "简历分析完成", "report": report}

# -------------------------
# 查询简历接口
# -------------------------
@app.post("/query_resume")
def query_resume(req: QueryRequest):
    """
    根据用户问题查询向量库（按 file_name 隔离）
    """
    db = load_faiss(req.file_name)
    structured_resume = load_json(req.file_name)

    if db is None or structured_resume is None:
        return {"error": f"没有找到 {req.file_name} 的向量库或结构化数据，请先调用 /analyze_resume"}

    result = query_dynamic_category(
        db=db,
        structured_resume=structured_resume,
        query=req.query,
        top_k=5,
        use_category_filter=True
    )

    return {"query": req.query, "answer": result}

# -------------------------
# 本地启动
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
