import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel
from app.utils.files import load_faiss
from app.storage.db import load_resume
from app.pipline.query import query_dynamic_category
from app.pipline.pipline import main_pipeline
from app.storage.storage_client import StorageClient

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
    file_names: Union[str, List[str]]  # 支持单文件或列表

class QueryRequest(BaseModel):
    file_name: str
    query: str

# -------------------------
# 分析简历接口
# -------------------------
@app.post("/analyze_resume")
def analyze_resume(req: ResumeRequest):
    """
    输入简历文件名（单个或列表，存储在 MinIO），下载并分析
    """
    client = StorageClient()
    downloads_dir = "./downloads"
    os.makedirs(downloads_dir, exist_ok=True)

    # 统一成列表处理
    file_list = req.file_names if isinstance(req.file_names, list) else [req.file_names]

    # 下载文件到本地
    for file_name in file_list:
        local_resume_path = os.path.join(downloads_dir, file_name)
        client.read_file(file_name, local_resume_path)

    # 调用批量处理主流程
    all_results = main_pipeline(file_list, mode="exact")

    # 生成报告（按文件名+邮箱汇总）
    reports = {}
    for file_name, structured_resume in all_results.items():
        key = f"{file_name} ({structured_resume.get('email', 'N/A')})"  # 文件名+邮箱
        reports[key] = {
            "basic_info": {
                "name": structured_resume.get("name", "N/A"),
                "email": structured_resume.get("email", "N/A"),
                "phone": structured_resume.get("phone", "N/A"),
            },
            "work_experience": structured_resume.get("work_experience", []),
            "projects": structured_resume.get("projects", []),
            "education": structured_resume.get("education", []),
            "skills": structured_resume.get("skills", [])
        }

    return {"message": "简历分析完成", "reports": reports}

# -------------------------
# 查询简历接口
# -------------------------
@app.post("/query_resume")
def query_resume(req: QueryRequest):
    """
    根据用户问题查询向量库（按 file_name 隔离）
    """
    db = load_faiss(req.file_name)
    structured_resume = load_resume(user_id=None, file_name=req.file_name)  # user_id 可留空，用 file_name 唯一标识

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
