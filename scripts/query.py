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
def detect_query_category(query: str):
    """
    分类归一化，只根据 query 判断类别
    """
    query_lower = query.lower()
    if any(k in query_lower for k in ["work", "experience", "career", "job", "employment", "工作经历"]):
        return "work_experience"
    elif any(k in query_lower for k in ["project", "projects", "项目经历", "项目"]):
        return "projects"
    elif any(k in query_lower for k in ["education", "degree", "university", "school", "bachelor", "master", "教育"]):
        return "education"
    elif any(k in query_lower for k in ["skill", "skills", "python", "tensorflow", "ml", "pytorch", "sql", "技能"]):
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
    使用 query_results 完全覆盖原 JSON 对应类别
    - 保留基础信息 (name/email/phone)
    - 工作经历：合并所有任务描述
    - 独立项目：按换行合并成 project_title + project_content
    - 技能列表规范化
    """
    # 基础信息
    base_info = {k: structured.get(k) for k in ["name", "email", "phone"]}

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

    # ---- 教育经历 ----
    edu_date_pattern = re.compile(
        r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?"
        r"(\d{4})",
        re.I
    )

    for para in query_results.get("教育经历", []):
        parts = [p.strip() for p in para.split("|")]
        entry = {
            "school": parts[0] if len(parts) > 0 else "N/A",
            "degree": parts[1] if len(parts) > 1 else "N/A",
            "grad_date": "Unknown",
            "description": para
        }
        match = edu_date_pattern.search(para)
        if match:
            month, year = match.groups()
            entry["grad_date"] = f"{month} {year}" if month else year

        new_structured["education"].append(entry)

    # ---- 工作经历 ----
    work_date_pattern = re.compile(
        r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?"
        r"(\d{4})\s*[-–]\s*"
        r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?"
        r"(Present|\d{4})",
        re.I
    )

    for para in query_results.get("工作经历", []):
        parts = [p.strip() for p in para.split("|")]
        entry = {
            "company": parts[0] if len(parts) > 0 else "N/A",
            "title": parts[1] if len(parts) > 1 else "N/A",
            "start_date": "Unknown",
            "end_date": "Present",
            "description": para
        }
        match = work_date_pattern.search(para)
        if match:
            start_month, start_year, end_month, end_year = match.groups()
            entry["start_date"] = f"{start_month} {start_year}" if start_month else start_year
            entry["end_date"] = f"{end_month} {end_year}" if end_month else end_year

        new_structured["work_experience"].append(entry)

    # ---- 独立项目 ----
    for para in query_results.get("项目经历", []):
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        if not lines:
            continue

        # 规则改动：第一行如果不以动词开头，作为标题
        if re.match(r"^(built|created|used|collected|led|fine\-tuned)", lines[0].lower()):
            title = ""
            content = "\n".join(lines)
        else:
            title = lines[0]
            content = "\n".join(lines[1:])

        entry = {
            "project_title": title if title else content[:60],  # 没标题用前60字符
            "start_date": "Unknown",
            "end_date": "Present",
            "project_content": content
        }
        new_structured["projects"].append(entry)

    # ---- 技能 ----
    for para in query_results.get("技能", []):
        extracted = extract_skills_from_text(para)
        new_structured["skills"].extend(extracted)

    new_structured["skills"] = normalize_skills(new_structured["skills"])

    # ---- 其他 ----
    for para in query_results.get("其他", []):
        new_structured["other"].append({"description": para})

    return new_structured
