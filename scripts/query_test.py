import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import re
from utils import normalize_category, normalize_skills, extract_skills_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------
# 查询接口
# -------------------------
def detect_query_category(query: str):
    query_lower = query.lower()
    if any(k in query_lower for k in ["work", "experience", "career", "job", "employment", "工作经历"]):
        return "work_experience"
    elif any(k in query_lower for k in ["project", "projects", "项目经历", "项目"]):
        return "projects"
    elif any(k in query_lower for k in ["education", "degree", "university", "school", "bachelor", "master", "教育"]):
        return "education"
    elif any(k in query_lower for k in ["skill", "skills", "python", "tensorflow", "ml", "pytorch", "sql", "技能"]):
        return "skills"
    elif any(k in query_lower for k in ["other", "其他"]):
        return "other"
    else:
        return None

def query_dynamic_category(db, structured_resume, query: str, top_k=10, use_category_filter=True):
    """
    查询指定类别段落（可严格类别过滤）
    返回 {"query": query, "results": [...] }
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

# -------------------------
# 多类别整合查询（安全版）
# -------------------------
def query_all_categories(db, structured_resume, top_k=10):
    queries = ["工作经历", "项目经历", "教育经历", "技能", "其他"]
    all_results = {}
    for q in queries:
        try:
            res = query_dynamic_category(db, structured_resume, q, top_k=top_k)
            paras = res.get("results", [])
            # 如果严格过滤没有结果，用非严格模式 fallback
            if not paras:
                res = query_dynamic_category(db, structured_resume, q, top_k=top_k, use_category_filter=False)
                paras = res.get("results", [])
            all_results[q] = paras
        except Exception as e:
            logger.warning(f"[QUERY ALL] Error querying '{q}': {e}")
            all_results[q] = []
    return all_results


# -------------------------
# 填充结构化 JSON（安全版）
# -------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import re
from utils import normalize_category, normalize_skills, extract_skills_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------
# 安全版填充函数
# -------------------------
def fill_query_exact(structured: dict, query_results: dict) -> dict:
    """
    使用 query_results 完全覆盖原 JSON 对应类别
    对所有类别支持字符串或 dict 输入，避免 AttributeError
    支持 N/A 前缀自动跳过
    """
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

    def safe_text(para):
        """保证返回字符串内容，并去掉 N/A 前缀"""
        if isinstance(para, dict):
            text = str(para.get("description", "")).strip()
        else:
            text = str(para).strip()
        return re.sub(r"^(N/A\s*)", "", text, flags=re.I).strip()

    # 教育经历
    edu_date_pattern = re.compile(r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+)?(\d{4})", re.I)
    edu_paras = query_results.get("教育经历", []) or structured.get("education", [])
    for para in edu_paras:
        text = safe_text(para)
        if not text:
            continue
        parts = [p.strip() for p in text.split("|")]
        entry = {
            "school": parts[0] if len(parts) > 0 else "N/A",
            "degree": parts[1] if len(parts) > 1 else "N/A",
            "grad_date": "Unknown",
            "description": text
        }
        match = edu_date_pattern.search(text)
        if match:
            month, year = match.groups()
            entry["grad_date"] = f"{month} {year}" if month else year
        new_structured["education"].append(entry)

    # 工作经历
    work_date_pattern = re.compile(
        r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+)?(\d{4})\s*[-–]\s*(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+)?(Present|\d{4})",
        re.I
    )
    work_paras = query_results.get("工作经历", []) or structured.get("work_experience", [])
    for para in work_paras:
        text = safe_text(para)
        if not text:
            continue
        parts = [p.strip() for p in text.split("|")]
        entry = {
            "company": parts[0] if len(parts) > 0 else "N/A",
            "title": parts[1] if len(parts) > 1 else "N/A",
            "start_date": "Unknown",
            "end_date": "Present",
            "description": text
        }
        match = work_date_pattern.search(text)
        if match:
            start_month, start_year, end_month, end_year = match.groups()
            entry["start_date"] = f"{start_month} {start_year}" if start_month else start_year
            entry["end_date"] = f"{end_month} {end_year}" if end_month else end_year
        new_structured["work_experience"].append(entry)

    # 项目经历
    proj_paras = query_results.get("项目经历", []) or structured.get("projects", [])
    for para in proj_paras:
        text = safe_text(para)
        if not text:
            continue
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            continue
        if re.match(r"^(built|created|used|collected|led|fine\-tuned)", lines[0].lower()):
            title = ""
            content = "\n".join(lines)
        else:
            title = lines[0]
            content = "\n".join(lines[1:])
        entry = {
            "project_title": title if title else content[:60],
            "start_date": "Unknown",
            "end_date": "Present",
            "project_content": content
        }
        new_structured["projects"].append(entry)

    # 技能
    skills_paras = query_results.get("技能", []) or structured.get("skills", [])
    for para in skills_paras:
        text = safe_text(para)
        if not text:
            continue
        extracted = extract_skills_from_text(text.lower())
        new_structured["skills"].extend(extracted)
    if not new_structured["skills"]:
        new_structured["skills"] = structured.get("skills", [])

    # 其他
    other_paras = query_results.get("其他", []) or structured.get("other", [])
    for para in other_paras:
        text = safe_text(para)
        if not text:
            continue
        new_structured["other"].append({"description": text})

    return new_structured


# -------------------------
# 测试
# -------------------------
if __name__ == "__main__":
    # 模拟 FAISS 返回带 N/A
    query_results = {
        "工作经历": ["N/A | OpenAI | Research Scientist | Jan 2023 – Present"],
        "项目经历": ["N/A\nBuilt a QA system using PyTorch and Python"],
        "教育经历": ["N/A | Northeastern University | Master | 2025"],
        "技能": ["N/A | Python\nSQL\nPandas"],
        "其他": ["N/A | Some other info unrelated"]
    }

    # 原始结构化 resume
    structured_resume = {
        "name": "Zhenyu Zhang",
        "email": "zhang@example.com",
        "phone": "+1234567890",
        "education": [],
        "work_experience": [],
        "projects": [],
        "skills": [],
        "other": []
    }

    filled_resume = fill_query_exact(structured_resume, query_results)

    import json
    print(json.dumps(filled_resume, ensure_ascii=False, indent=2))
    