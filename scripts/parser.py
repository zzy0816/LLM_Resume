import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import re
from utils import CATEGORY_FIELDS, normalize_category, normalize_skills , auto_fill_fields, extract_basic_info
from ner import run_ner_batch
from utils_parser import semantic_fallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def classify_paragraphs(paragraph: str, structured: dict, ner_results=None, sem_cat=None):
    para_clean = paragraph.strip().replace("\r", " ").replace("\n", " ")
    if not para_clean:
        return "other", {}

    para_lower = para_clean.lower()

    # ---- 基础信息 ----
    info = extract_basic_info(para_clean)
    if info:
        structured["email"] = structured.get("email") or info.get("email")
        structured["phone"] = structured.get("phone") or info.get("phone")
        return "basic_info", {}

    if any(k in para_lower for k in ["linkedin", "github", "电话", "邮箱"]):
        return "basic_info", {}

    # ---- 初始化 data ----
    def init_data_for_category(cat, text):
        fields = CATEGORY_FIELDS.get(cat, ["description"])
        d = {f: None for f in fields}
        if "description" in fields:
            d["description"] = text
        if "skills" in fields:
            d["skills"] = []
        return d

    category = None
    edu_keywords = ["university", "college", "学院", "大学", "bachelor", "master", "phd", "ma", "ms", "mba"]
    work_keywords = ["intern", "engineer", "manager", "responsible", "工作", "实习", "任职", "developer", "consultant"]

    if any(k in para_lower for k in edu_keywords):
        category = "education"
        data = init_data_for_category(category, para_clean)
        parts = [p.strip() for p in para_clean.split("|")]
        if len(parts) >= 2:
            data["school"] = parts[0]
            data["degree"] = parts[1]
        for p in parts:
            year_match = re.search(r"\b(19|20)\d{2}\b", p)
            if year_match:
                data["grad_date"] = data["grad_date"] or year_match.group()
    elif any(k in para_lower for k in work_keywords):
        category = "work_experience"
        data = init_data_for_category(category, para_clean)
        parts = [p.strip() for p in para_clean.split("|")]
        if len(parts) >= 2:
            data["company"] = parts[0]
            data["title"] = parts[1]
        for p in parts:
            date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{4})\s*–\s*(Present|[0-9]{4})", p, re.I)
            if date_match:
                data["start_date"] = date_match.group(1)
                data["end_date"] = date_match.group(2)
    else:
        category = None
        data = init_data_for_category("other", para_clean)

    # ---- 使用批量 NER 结果 ----
    if ner_results:
        try:
            for ent in ner_results:
                label = ent["entity_group"].lower()
                val = ent["word"].strip()
                if label == "per" and structured.get("name") is None:
                    structured["name"] = val
                elif label == "org" and "company" in data and not data["company"]:
                    data["company"] = val
                elif label == "title" and "title" in data and not data["title"]:
                    data["title"] = val
                elif label == "edu" and "school" in data and not data["school"]:
                    data["school"] = val
                elif label == "degree" and "degree" in data and not data["degree"]:
                    data["degree"] = val
                elif label == "skill" and "skills" in data and val not in data["skills"]:
                    data["skills"].append(val)
                elif label == "date":
                    if "start_date" in data and not data["start_date"]:
                        data["start_date"] = val
                    elif "end_date" in data and not data["end_date"]:
                        data["end_date"] = val
        except Exception as e:
            logger.warning(f"[NER ERROR] {e}")

    # ---- 技能关键词补全 ----
    skill_keywords = [
        "python","sql","pandas","numpy","scikit","sklearn","tensorflow",
        "pytorch","keras","docker","kubernetes","aws","gcp","azure",
        "spark","hadoop","tableau","powerbi","llm","llama","hugging"
    ]
    if "skills" in data:
        for kw in skill_keywords:
            if kw in para_lower and kw not in data["skills"]:
                data["skills"].append(kw.upper() if kw in ["sql","llm","aws","hugging"] else kw.capitalize())

    # ---- fallback 使用批量结果 ----
    if not category:
        category = normalize_category(sem_cat or "other")
        data = init_data_for_category(category, para_clean)

    return normalize_category(category), data

# -------------------------
# parse_resume_to_structured
# -------------------------
def parse_resume_to_structured(paragraphs: list, file_name: str = None):
    structured = {
        "name": None, "email": None, "phone": None,
        "education": [], "work_experience": [], "projects": [], "skills": [], "other": []
    }

    # 批量 NER 和语义 fallback
    ner_results_batch = run_ner_batch(paragraphs)
    semantic_cats = semantic_fallback(paragraphs, file_name=file_name)

    for para, ner_results, sem_cat in zip(paragraphs, ner_results_batch, semantic_cats):
        category, data = classify_paragraphs(para, structured, ner_results, sem_cat)
        if category == "basic_info":
            continue
        elif category == "work_experience":
            structured["work_experience"].append(data)
        elif category == "projects":
            structured["projects"].append(data)
        elif category == "education":
            structured["education"].append(data)
        elif category == "skills":
            structured["skills"].extend(data.get("skills", []))
        else:
            structured["other"].append(data)

    structured["skills"] = normalize_skills(structured["skills"])
    structured = auto_fill_fields(structured)
    return structured
