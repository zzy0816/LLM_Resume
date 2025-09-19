import sys
import os
import logging
import json
from files import load_faiss, load_json, save_faiss, save_json
from doc import read_document_paragraphs
from semantic import build_faiss
from parser import parse_resume_to_structured
from utils import auto_fill_fields, extract_basic_info, rule_based_filter, validate_and_clean
from query import query_dynamic_category, fill_query_exact
from db import save_resume

# --------------------------- 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------- 批量简历处理
def sanitize_filename(file_name: str) -> str:
    """移除文件名中可能导致路径问题的字符"""
    return file_name.replace("(", "_").replace(")", "_").replace(" ", "_")

def main_pipeline(file_names: list[str], mode: str = "exact") -> dict[str, dict]:
    results = {}

    for file_name in file_names:
        file_path = f"./downloads/{file_name}"
        logger.info(f"DEBUG: processing file {file_name}")

        safe_name = sanitize_filename(file_name)

        # 1️⃣ 尝试加载缓存 JSON
        structured_resume = load_json(safe_name)
        if structured_resume is None:
            paragraphs = read_document_paragraphs(file_path)
            full_text = "\n".join(paragraphs)
            logger.info(f"DEBUG: total paragraphs = {len(paragraphs)}")
            logger.info(f"DEBUG: first 5 paragraphs: {paragraphs[:5]}")

            # --- 先抓全局基础信息 ---
            basic_info = extract_basic_info(full_text)
            logger.info(f"DEBUG: basic_info extracted: {basic_info}")

            structured_resume = {
                "name": None,   # 让 parser 来决定
                "email": basic_info.get("email"),
                "phone": basic_info.get("phone"),
                "basic_info": basic_info,
                "education": [],
                "work_experience": [],
                "projects": [],
                "skills": [],
                "other": []
            }

            # 用 parser 结果覆盖
            parsed_resume = parse_resume_to_structured(paragraphs)

            if parsed_resume.get("name"):  
                structured_resume["name"] = parsed_resume["name"]


            # 解析简历
            if parsed_resume is None:
                logger.warning(f"parse_resume_to_structured returned None for {file_name}, using empty structured_resume")
                parsed_resume = {}

            # 自动填充
            parsed_resume = auto_fill_fields(parsed_resume or {})
            
            # 合并 parsed_resume 到 structured_resume，但保留 basic_info
            for key in parsed_resume:
                if key not in ["name", "email", "phone"]:
                    structured_resume[key] = parsed_resume[key]

            # 再次覆盖 basic_info
            structured_resume["basic_info"]["name"] = basic_info.get("name") or structured_resume["basic_info"].get("name")
            structured_resume["basic_info"]["email"] = basic_info.get("email") or structured_resume["basic_info"].get("email")
            structured_resume["basic_info"]["phone"] = basic_info.get("phone") or structured_resume["basic_info"].get("phone")

            logger.info(f"DEBUG: final structured_resume after merge: {structured_resume['basic_info']}")

        user_email = structured_resume.get("email") or safe_name

        # 2️⃣ 构建或加载 FAISS
        db = load_faiss(safe_name)
        if db is None:
            logger.info(f"DEBUG: FAISS not found, building for {safe_name}")
            db = build_faiss(structured_resume)
            if db is not None:
                save_faiss(safe_name, db)
                logger.info(f"DEBUG: FAISS saved for {safe_name}")
            else:
                logger.warning(f"build_faiss returned None, skipping FAISS save for {safe_name}")

        # 3️⃣ 查询 FAISS 并生成 query_results
        queries = ["工作经历", "项目经历", "教育经历", "技能"]
        query_results = {}
        if db is not None:
            for q in queries:
                res = query_dynamic_category(db, structured_resume, q, top_k=10)
                raw_results = res.get("results", [])
                logger.info(f"DEBUG: raw query '{q}' results: {raw_results}")

                # ✅ 二次过滤，去掉不相关内容
                filtered = rule_based_filter(q, raw_results)
                query_results[q] = filtered
                logger.info(f"DEBUG: filtered query '{q}' results: {query_results[q]}")
        else:
            logger.warning(f"No FAISS db for {safe_name}, skipping dynamic query")

        # 4️⃣ 使用 query 结果填充结构化 JSON
        if mode == "exact":
            structured_resume = fill_query_exact(structured_resume, query_results)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # ✅ 再次清洗，防止分类错位
        structured_resume = validate_and_clean(structured_resume)

        # 4️⃣ 使用 query 结果填充结构化 JSON
        if mode == "exact":
            structured_resume = fill_query_exact(structured_resume, query_results)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # 5️⃣ 保存最终 JSON
        if structured_resume:
            save_resume(user_id=user_email, file_name=safe_name + "_faiss_confirmed", data=structured_resume)
            save_json(safe_name + "_faiss_confirmed", structured_resume)
            logger.info(f"DEBUG: saved resume for {user_email}")
        else:
            logger.warning(f"No structured_resume to save for {file_name}")

        results[user_email] = structured_resume

    return results

if __name__ == "__main__":
    files_to_process = ["Resume(AI).pdf"]
    all_results = main_pipeline(files_to_process, mode="exact")

    for user_email, structured_resume in all_results.items():
        logger.info(f"\n===== FINAL STRUCTURED RESUME JSON for {user_email} =====")
        logger.info(json.dumps(structured_resume, ensure_ascii=False, indent=2))
