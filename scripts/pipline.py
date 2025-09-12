import sys
import os
import logging
import json
from files import load_faiss, load_json, save_faiss
from doc import read_document_paragraphs
from semantic import build_faiss
from parser import parse_resume_to_structured
from utils import auto_fill_fields, extract_basic_info
from query import query_dynamic_category, fill_query_exact
from db import save_resume

# ---------------------------
# 日志配置
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------
# 批量简历处理
# ---------------------------
def main_pipeline(file_names: list[str], mode: str = "exact") -> dict[str, dict]:
    results = {}

    for file_name in file_names:
        file_path = f"./downloads/{file_name}"
        logger.info(f"DEBUG: processing file {file_name}")

        # 1️⃣ 尝试加载缓存 JSON
        structured_resume = load_json(file_name)
        if structured_resume is None:
            paragraphs = read_document_paragraphs(file_path)
            full_text = "\n".join(paragraphs)
            logger.info(f"DEBUG: total paragraphs = {len(paragraphs)}")
            logger.info(f"DEBUG: first 5 paragraphs: {paragraphs[:5]}")

            # --- 先抓全局基础信息 ---
            basic_info = extract_basic_info(full_text)
            logger.info(f"DEBUG: basic_info extracted: {basic_info}")

            # 初始化结构化 JSON，保证 name/email/phone 不丢
            structured_resume = {
                "name": basic_info.get("name"),
                "email": basic_info.get("email"),
                "phone": basic_info.get("phone"),
                "basic_info": basic_info,  # 新增字段
                "education": [],
                "work_experience": [],
                "projects": [],
                "skills": [],
                "other": []
            }

            # 解析简历
            parsed_resume = parse_resume_to_structured(paragraphs, file_name=file_name)
            logger.info(f"DEBUG: parsed_resume name = {parsed_resume.get('name')}")

            # 自动填充
            parsed_resume = auto_fill_fields(parsed_resume)
            logger.info(f"DEBUG: after auto_fill_fields name = {parsed_resume.get('name')}")

            # 合并 parsed_resume 到 structured_resume，但保留 basic_info
            for key in parsed_resume:
                if key not in ["name", "email", "phone"]:
                    structured_resume[key] = parsed_resume[key]

            # 再次覆盖 basic_info，保证正确
            structured_resume["basic_info"]["name"] = basic_info.get("name") or structured_resume["basic_info"].get("name")
            structured_resume["basic_info"]["email"] = basic_info.get("email") or structured_resume["basic_info"].get("email")
            structured_resume["basic_info"]["phone"] = basic_info.get("phone") or structured_resume["basic_info"].get("phone")

            logger.info(f"DEBUG: final structured_resume after merge: {structured_resume['basic_info']}")

        # 获取 user_email（Email 或文件名）
        user_email = structured_resume.get("email") or file_name

        # 2️⃣ 构建或加载 FAISS
        db = load_faiss(file_name)
        if db is None:
            logger.info(f"DEBUG: FAISS not found, building for {file_name}")
            db = build_faiss(structured_resume)
            save_faiss(file_name, db)
            logger.info(f"DEBUG: FAISS saved for {file_name}")

        # 3️⃣ 查询 FAISS 并生成 query_results
        queries = ["工作经历", "项目经历", "教育经历", "技能"]
        query_results = {}
        for q in queries:
            res = query_dynamic_category(db, structured_resume, q, top_k=10)
            query_results[q] = res.get("results", [])
            logger.info(f"DEBUG: query '{q}' results: {query_results[q]}")

        # 4️⃣ 使用 query 结果填充结构化 JSON
        if mode == "exact":
            structured_resume = fill_query_exact(structured_resume, query_results)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # 5️⃣ 保存最终 JSON
        save_resume(user_id=user_email, file_name=file_name + "_faiss_confirmed", data=structured_resume)
        logger.info(f"DEBUG: saved resume for {user_email}")

        results[user_email] = structured_resume

    return results

if __name__ == "__main__":
    files_to_process = ["Resume(AI).docx"]
    all_results = main_pipeline(files_to_process, mode="exact")

    for user_email, structured_resume in all_results.items():
        logger.info(f"\n===== FINAL STRUCTURED RESUME JSON for {user_email} =====")
        logger.info(json.dumps(structured_resume, ensure_ascii=False, indent=2))
