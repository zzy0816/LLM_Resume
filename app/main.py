import logging
import os
import sys

import uvicorn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel

from app.pipline.pipline import main_pipeline
from app.qre.query import query_dynamic_category
from app.storage.db import load_resume
from app.storage.storage_client import StorageClient
from app.utils.files import load_faiss, setup_logging
from app.utils.utils import sanitize_filename
from app.utils.files import FAISS_DIR, CLASSIFIED_DIR

setup_logging()
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
class ResumeRequest(BaseModel):
    file_names: Union[str, List[str]]  # 单文件或列表


@app.post("/analyze_resume")
def analyze_resume(req: ResumeRequest):
    """
    输入简历文件名（单个或列表，存储在 MinIO），下载并分析
    """
    client = StorageClient()
    downloads_dir = "./downloads"
    os.makedirs(downloads_dir, exist_ok=True)

    # 统一成列表处理
    file_list = (
        req.file_names
        if isinstance(req.file_names, list)
        else [req.file_names]
    )

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
            "skills": structured_resume.get("skills", []),
        }

    return {"message": "简历分析完成", "reports": reports}


# -------------------------
# 查询简历接口
# -------------------------
@app.post("/query_resume")
@app.post("/query_resume")
def query_resume(req: QueryRequest):
    """
    根据用户问题查询向量库（按 file_name 隔离）
    """

    # 1. 将用户传入的原始文件名统一成 pipeline 保存名
    safe_file_name = sanitize_filename(req.file_name)  # "Resume(AI).docx" -> "Resume_AI_.pdf"

    # 2. 计算 JSON 和 FAISS 的绝对路径
    json_path = os.path.join(CLASSIFIED_DIR, f"{safe_file_name}.json")
    faiss_path = os.path.join(FAISS_DIR, safe_file_name)

    # 3. 检查文件是否存在
    if not os.path.exists(json_path):
        return {
            "error": f"没有找到 {req.file_name} 的结构化 JSON，请先调用 /analyze_resume"
        }
    if not os.path.exists(faiss_path):
        return {
            "error": f"没有找到 {req.file_name} 的向量库，请先调用 /analyze_resume"
        }

    # 4. 加载 JSON
    structured_resume = load_resume(None, safe_file_name)

    # 5. 加载 FAISS
    db = load_faiss(safe_file_name)
    if db is None:
        return {"error": f"没有找到 {req.file_name} 的向量库，请先调用 /analyze_resume"}

    # 6. 执行查询
    results = query_dynamic_category(
        db=db,
        structured_resume=structured_resume,
        query=req.query,
        top_k=5,
        use_category_filter=True,
    )

    # 只返回结果数组
    return {"query": req.query, "answer": results["results"]}

# -------------------------
# 本地启动
# -------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # 替换成你的 FastAPI 实例路径
        host="0.0.0.0",
        port=8000,
        log_config=None,  # 禁用 Uvicorn 默认 logging 配置
        log_level="info",
    )
