import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import re
from utils import normalize_category, normalize_skills ,extract_skills_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------
# 查询接口
# -------------------------
def detect_query_category(query:str):
    """
    分类归一化
    """
    query_lower = query.lower()
    if any(k in query_lower for k in ["work","experience","career","job","employment","工作经历"]):
        return "work_experience"
    elif any(k in query_lower for k in ["project","built","developed","created","项目"]):
        return "projects"
    elif any(k in query_lower for k in ["education","degree","university","school","bachelor","master","教育"]):
        return "education"
    elif any(k in query_lower for k in ["skill","skills","python","tensorflow","ml","pytorch","sql","技能"]):
        return "skills"
    else:
        return None
    
def query_dynamic_category(db, structured_resume, query: str, top_k=10, use_category_filter=True):
    """
    基于 FAISS 查询指定类别段落（严格类别过滤）
    根据查询语义动态匹配分类：
    - 返回与 query 最相关的结构化条目列表
    - query 可以是 "工作经历", "项目经历", "教育经历", "技能" 等
    """
    docs = db.similarity_search(query, k=top_k*5)
    logger.debug("[QUERY DEBUG] retrieved %d docs for query='%s'", len(docs), query)

    candidate_paras = []
    target_category = normalize_category(detect_query_category(query))

    if use_category_filter and target_category:
        for doc in docs:
            doc_cat = normalize_category(doc.metadata.get("category", "other"))

            if target_category == "education":
                doc_lower = doc.page_content.lower()
                has_school = any(tok in doc_lower for tok in ["university","college","学院","大学"])
                has_degree = any(tok in doc_lower for tok in ["bachelor","master","phd","bs","ms","mba","学士","硕士","博士"])
                if doc_cat == "education" and (has_school or has_degree):
                    candidate_paras.append(doc.page_content)

            elif target_category == "skills":
                skill_keywords = [
                    "python","sql","pandas","numpy","scikit","sklearn","tensorflow",
                    "pytorch","keras","docker","kubernetes","aws","gcp","azure",
                    "spark","hadoop","tableau","powerbi","llm","llama","hugging"
                ]
                if doc_cat == "skills":
                    candidate_paras.append(doc.page_content)
                elif doc_cat == "other" and len(doc.page_content) < 150:
                    if any(k in doc.page_content.lower() for k in skill_keywords):
                        candidate_paras.append(doc.page_content)

            else:
                if doc_cat == target_category:
                    candidate_paras.append(doc.page_content)

            if len(candidate_paras) >= top_k:
                break

        if not candidate_paras:
            logger.warning("No candidate paragraphs found for query='%s' with strict category filter.", query)
            return {"query": query, "results": []}
    else:
        candidate_paras = [doc.page_content for doc in docs[:top_k]]

    for i, p in enumerate(candidate_paras):
        logger.info("[QUERY RESULT] %d. %s", i+1, p[:140])

    return {"query": query, "results": candidate_paras}

def fill_query_exact(structured: dict, query_results: dict) -> dict:
    """
    使用 query_results 完全覆盖原 JSON 对应类别，
    保留基础信息 (name/email/phone)
    根据 query 精准返回相关字段内容
    - 如果 query 是技能列表，返回匹配技能
    - 如果 query 是其他类别，返回对应列表
    """
    # 保留基础信息
    base_info = {k: structured.get(k) for k in ["name", "email", "phone"]}

    # 初始化空结构
    new_structured = {
        "name": base_info.get("name"),
        "email": base_info.get("email"),
        "phone": base_info.get("phone"),
        "education": [],
        "work_experience": [],
        "projects": [],
        "skills": [],
        "other": []
    }

    # ---- 工作经历 ----
    for para in query_results.get("工作经历", []):
        parts = [p.strip() for p in para.split("|")]
        entry = {
            "company": parts[0] if len(parts) > 0 else "N/A",
            "title": parts[1] if len(parts) > 1 else "N/A",
            "start_date": "Unknown",
            "end_date": "Unknown",
            "description": para
        }
        # 尝试解析时间范围
        date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})\s*[-–]\s*(Present|\d{4})", para, re.I)
        if date_match:
            entry["start_date"] = date_match.group(1)
            entry["end_date"] = date_match.group(2)
        new_structured["work_experience"].append(entry)

    # ---- 教育经历 ----
    for para in query_results.get("教育经历", []):
        parts = [p.strip() for p in para.split("|")]
        entry = {
            "school": parts[0] if len(parts) > 0 else "N/A",
            "degree": parts[1] if len(parts) > 1 else "N/A",
            "grad_date": "N/A",
            "description": para
        }
        # 尝试抽取年份
        year_match = re.search(r"\b(19|20)\d{2}\b", para)
        if year_match:
            entry["grad_date"] = year_match.group()
        new_structured["education"].append(entry)

    # ---- 项目经历 ----
    for para in query_results.get("项目经历", []):
        entry = {
            "project_title": None,
            "start_date": "Unknown",
            "end_date": "Present",
            "project_content": para
        }
        # 尝试提取标题
        title_match = re.match(r"^(.*?)[:\-]", para)
        if title_match:
            entry["project_title"] = title_match.group(1).strip()
        # 尝试提取日期
        date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})\s*[-–]\s*(Present|\d{4})", para, re.I)
        if date_match:
            entry["start_date"] = date_match.group(1)
            entry["end_date"] = date_match.group(2)
        new_structured["projects"].append(entry)


    # ---- 技能 ----
    for para in query_results.get("技能", []):
        extracted = extract_skills_from_text(para)
        new_structured["skills"].extend(extracted)

    # 技能去重 & 规范化
    new_structured["skills"] = normalize_skills(new_structured["skills"])

    # ---- 其他 ----
    for para in query_results.get("其他", []):
        new_structured["other"].append({"description": para})

    return new_structured
