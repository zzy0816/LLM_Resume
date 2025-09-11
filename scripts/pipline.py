import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import json
from files import load_faiss, load_json, save_faiss
from doc import read_document_paragraphs
from semantic import build_faiss
from parser import parse_resume_to_structured
from utils import auto_fill_fields
from query import query_dynamic_category, fill_query_exact
from db import save_resume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------
# 批量简历处理
# -------------------------
def main_pipeline(file_names: list[str], mode: str = "exact") -> dict[str, dict]:
    """
    批量简历处理：
    file_names: 简历文件名列表
    mode: exact 模式
    返回 {user_email: structured_resume} 字典
    """
    results = {}

    for file_name in file_names:
        file_path = f"./downloads/{file_name}"

        # 1️⃣ 尝试加载缓存 JSON
        structured_resume = load_json(file_name)
        if structured_resume is None:
            paragraphs = read_document_paragraphs(file_path)
            structured_resume = parse_resume_to_structured(paragraphs, file_name=file_name)
            structured_resume = auto_fill_fields(structured_resume)

        # 获取 user_email（Email 或文件名）
        user_email = structured_resume.get("email") or file_name

        # 2️⃣ 构建或加载 FAISS
        db = load_faiss(file_name)
        if db is None:
            db = build_faiss(structured_resume)
            save_faiss(file_name, db)

        # 3️⃣ 查询 FAISS 并生成 query_results
        queries = ["工作经历", "项目经历", "教育经历", "技能"]
        query_results = {}
        for q in queries:
            res = query_dynamic_category(db, structured_resume, q, top_k=10)
            query_results[q] = res.get("results", [])

        # 4️⃣ 使用 query 结果填充结构化 JSON
        if mode == "exact":
            structured_resume = fill_query_exact(structured_resume, query_results)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # 5️⃣ 保存最终 JSON（只保存一次，避免重复记录）
        save_resume(user_id=user_email, file_name=file_name + "_faiss_confirmed", data=structured_resume)

        # 保存结果到返回字典
        results[user_email] = structured_resume

    return results

if __name__ == "__main__":
    # 批量文件处理示例
    files_to_process = ["Resume(AI).docx", "Resume(AI).pdf"]
    all_results = main_pipeline(files_to_process, mode="exact")

    for user_email, structured_resume in all_results.items():
        logger.info(f"\n===== FINAL STRUCTURED RESUME JSON for {user_email} =====")
        logger.info(json.dumps(structured_resume, ensure_ascii=False, indent=2))
