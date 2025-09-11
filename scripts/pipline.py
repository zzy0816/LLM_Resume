import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import json
from files import load_faiss, load_json, save_faiss, save_json
from doc import read_docx_paragraphs
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
# 主流程示例
# -------------------------
def main_pipeline(file_name: str, mode: str = "exact") -> dict:
    """
    简历处理主流程：
    1. 读取 Word 段落
    2. 段落语义拆分
    3. 解析成结构化字典
    4. 自动补全缺失字段
    5. 如果提供 FAISS，则进行语义补全
    返回最终结构化字典
    """
    file_path = f"./downloads/{file_name}"

    # 1️⃣ 加载或解析简历
    structured_resume = load_json(file_name)
    if structured_resume is None:
        paragraphs = read_docx_paragraphs(file_path)
        structured_resume = parse_resume_to_structured(paragraphs, file_name=file_name)
        structured_resume = auto_fill_fields(structured_resume)
        save_json(file_name, structured_resume)

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

    # 5️⃣ 保存最终 JSON
    save_json(file_name + "_faiss_confirmed", structured_resume)

    return structured_resume

if __name__ == "__main__":
    file_name = "Resume(AI).docx"
    result = main_pipeline(file_name, mode="exact")

    logger.info("\n===== FINAL STRUCTURED RESUME JSON =====")
    logger.info(json.dumps(result, ensure_ascii=False, indent=2))
