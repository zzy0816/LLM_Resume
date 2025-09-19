import sys
import os
import re
import logging
from typing import List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    CATEGORY_FIELDS,
    normalize_category,
    normalize_skills,
    auto_fill_fields,
    extract_basic_info,
    extract_skills_from_text
)
from ner import run_ner_batch

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -------------------------
# 预处理段落，保持 Skills 行独立
# -------------------------
def preprocess_paragraphs(paragraphs: List[str]) -> List[str]:
    merged = []
    buffer = []
    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        if "project" in para_clean.lower() or "项目" in para_clean:
            if buffer:
                merged.extend(buffer)
                buffer = []
            merged.append(para_clean)
        elif para_clean.lower().startswith("skills:"):
            if buffer:
                merged.extend(buffer)
                buffer = []
            merged.append(para_clean)
        else:
            buffer.append(para_clean)
    if buffer:
        merged.extend(buffer)
    logger.debug(f"Preprocessed paragraphs: {merged}")
    return merged

# -------------------------
# 分类段落
# -------------------------
def classify_paragraph(paragraph: str, structured: dict, ner_results=None):
    para_clean = paragraph.strip()
    para_lower = para_clean.lower()

    # ---- 基础信息 ----
    info = extract_basic_info(para_clean)
    if info:
        structured["name"] = structured.get("name") or info.get("name")
        structured["email"] = structured.get("email") or info.get("email")
        structured["phone"] = structured.get("phone") or info.get("phone")

    def init_data(cat):
        fields = CATEGORY_FIELDS.get(cat, ["description"])
        d = {f: None for f in fields}
        if "description" in fields:
            d["description"] = para_clean
        if "skills" in fields:
            d["skills"] = []
        return d

    # ---- 工作经历 ----
    # 工作经历一般有 title | company | start–end
    if "email" not in para_lower and "phone" not in para_lower:
        work_match = re.match(
            r"(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)(?:–|-|to)(.*)", para_clean, re.I
        )
        if work_match:
            title, company, start, end = work_match.groups()
            category = "work_experience"
            data = init_data(category)
            data["title"] = (title or "").strip()
            data["company"] = (company or "").strip()
            data["start_date"] = (start or "").strip()
            data["end_date"] = (end or "").strip()
            return category, data

    # ---- 教育经历 ----
    # 教育经历必须包含学位关键字或学校关键字
    if any(k in para_clean.lower() for k in ["university", "college", "school", "bachelor", "master", "phd"]):
        edu_match = re.match(r"(.*?)\s*\|\s*(.*?)\s*\|?\s*(\d{4})?", para_clean)
        if edu_match:
            school, degree, year = edu_match.groups()
            category = "education"
            data = init_data(category)
            data["school"] = (school or "").strip()
            data["degree"] = (degree or "").strip()
            if year:
                data["grad_date"] = year.strip()
            return category, data

    # ---- 项目经历 ----
    if re.search(r"\b(Built|Created|Developed|Led|Designed|Implemented)\b", para_clean, re.I):
        category = "projects"
        data = init_data(category)
        title_match = re.match(
            r"(.*?)\s*(Built|Created|Developed|Led|Designed|Implemented)", para_clean, re.I
        )
        data["project_title"] = title_match.group(1).strip() if title_match else " ".join(para_clean.split()[:7])
        data["project_content"] = para_clean
        return category, data

    # ---- 技能 ----
    if para_lower.startswith("skills:"):
        category = "skills"
        data = init_data(category)
        skills_text = para_clean[len("skills:"):].strip()
        skills_list = [s.strip() for s in re.split(r",|;", skills_text) if s.strip()]
        data["skills"].extend(skills_list)
        if ner_results:
            for ent in ner_results:
                if ent["entity_group"].lower() == "skill":
                    val = ent["word"].strip()
                    if val not in data["skills"]:
                        data["skills"].append(val)
        return category, data

    # ---- NER 补充其他技能 ----
    if ner_results:
        data = init_data("other")
        for ent in ner_results:
            label = ent["entity_group"].lower()
            val = ent["word"].strip()
            if label == "skill" and "skills" in data:
                if val not in data["skills"]:
                    data["skills"].append(val)
        return "other", data

    # 默认其他
    category = "other"
    data = init_data(category)
    return category, data

# -------------------------
# 完整解析
# -------------------------
def parse_resume_to_structured(paragraphs: List[str]):
    structured = {
        "name": None,
        "email": None,
        "phone": None,
        "education": [],
        "work_experience": [],
        "projects": [],
        "skills": [],
        "other": []
    }

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue

        para_lower = para_clean.lower()

        # ---- 基础信息 ----
        if not structured["email"] and "@" in para_clean:
            email_match = re.search(r"[\w\.-]+@[\w\.-]+", para_clean)
            if email_match:
                structured["email"] = email_match.group(0)

        if not structured["phone"] and re.search(r"\+?\d[\d\s\-()]{6,}", para_clean):
            phone_match = re.search(r"\+?\d[\d\s\-()]{6,}", para_clean)
            structured["phone"] = phone_match.group(0)

        if not structured["name"] and "|" in para_clean and "@" in para_clean:
            structured["name"] = para_clean.split("|")[0].strip()

        # ---- 教育经历 ----
        if any(k in para_lower for k in ["university", "college", "school", "bachelor", "master", "phd"]):
            parts = [p.strip() for p in para_clean.split("|")]
            edu_entry = {
                "school": parts[0] if len(parts) > 0 else "",
                "degree": parts[1] if len(parts) > 1 else "",
                "grad_date": parts[-1] if len(parts) > 2 else "Unknown",
                "description": para_clean
            }
            structured["education"].append(edu_entry)
            continue

        # ---- 工作经历 ----
        if any(kw in para_lower for kw in ["llc", "inc", "company", "intern", "engineer", "analyst"]):
            parts = [p.strip() for p in para_clean.split("|")]
            work_entry = {
                "company": parts[1] if len(parts) > 1 else "",
                "position": parts[0] if len(parts) > 0 else "",
                "start_date": parts[2].split("–")[0].strip() if len(parts) > 2 and "–" in parts[2] else "Unknown",
                "end_date": parts[2].split("–")[-1].strip() if len(parts) > 2 and "–" in parts[2] else "Present",
                "description": para_clean
            }
            structured["work_experience"].append(work_entry)
            continue

        # ---- 项目经历 ----
        if any(kw in para_lower for kw in ["project", "built", "developed", "created", "implemented", "designed"]):
            proj_entry = {
                "project_title": para_clean.split("|")[0][:50],
                "project_content": para_clean,
                "start_date": "Unknown",
                "end_date": "Present"
            }
            structured["projects"].append(proj_entry)
            continue

        # ---- 技能 ----
        if para_lower.startswith("skills") or para_lower.startswith("languages & tools"):
            skills_text = para_clean.split(":", 1)[-1]
            skills_list = [s.strip().lower() for s in re.split(r",|;", skills_text) if s.strip()]
            structured["skills"].extend(skills_list)
            continue

        # ---- 兜底放 other ----
        structured["other"].append({"description": para_clean})

    # 去重技能
    structured["skills"] = sorted(set(structured["skills"]))
    return structured

# -------------------------
# 测试
# -------------------------
if __name__ == "__main__":
    test_paragraphs = [
        "Zhenyu Zhang | Email: Zhang.zhenyu6@northeastern.edu | Phone: +1860234-7101",
        "Northeastern University | Master of Professional Study in Applied Machine Intelligence | 2025",
        "University of Connecticut | Bachelor of Art | 2022",
        "Data Science Intern | Google LLC | Jun 2024 – Aug 2024",
        "YouTube Recommendation System Built a recommendation model using DNN and LightGBM...",
        "Skills: Python, SQL, TensorFlow, PyTorch"
    ]
    result = parse_resume_to_structured(test_paragraphs)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
