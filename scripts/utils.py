import re
import logging

# -------------------------
# 分类段落
# -------------------------
CATEGORY_FIELDS = {
    "work_experience": ["company","position","location","start_date","end_date","description", " highlights"],
    "education": ["school","degree","grad_date","description"],
    "projects": ["title","highlights"],
    "skills": ["skills"],
    "other": ["description"]
}

# -------------------------
# 分类归一化
# -------------------------
def normalize_category(cat: str) -> str:
    mapping = {
        # ---- 工作经历 ----
        "work": "work_experience",
        "workexperience": "work_experience",
        "work_experience": "work_experience",
        "professionalexperience": "work_experience",
        "industryexperience": "work_experience",
        "experience": "work_experience",

        # ---- 项目经历 ----
        "projects": "projects",
        "project": "projects",
        "projectexperience": "projects",

        # ---- 教育 ----
        "education": "education",
        "edu": "education",

        # ---- 技能 ----
        "skills": "skills",
        "skill": "skills",

        # ---- 其他 ----
        "other": "other"
    }
    key = cat.lower().replace(" ", "").replace("_", "")
    return mapping.get(key, "other")

# -------------------------
# 技能标准化词典
# -------------------------
SKILL_NORMALIZATION = {
    "hugging face": "HuggingFace",
    "huggingface": "HuggingFace",
    "langchain": "LangChain",
    "ollama": "Ollama",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "Scikit-learn",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "sql": "SQL",
    "llm": "LLM",
    "aws": "AWS"
}

def normalize_skills(skills: list) -> list:
    """安全技能标准化"""
    result = set()
    for s in skills:
        if not isinstance(s, str):
            continue
        s_clean = s.strip()
        if not s_clean:
            continue
        s_lower = s_clean.lower()
        if s_lower in SKILL_NORMALIZATION and SKILL_NORMALIZATION[s_lower]:
            result.add(SKILL_NORMALIZATION[s_lower])
        else:
            result.add(s_clean)
    return sorted(result)

def auto_fill_fields(structured_resume: dict) -> dict:
    """自动补全字段 & 技能（安全处理，不覆盖顶层 skills）"""
    logging.basicConfig(level=logging.DEBUG)

    # 定义允许保留的字段
    allowed_fields = {
        "education": {"school", "degree", "grad_date", "description"},
        "work_experience": {"company", "position", "location", "start_date", "end_date", "description", "highlights"},
        "projects": {"title", "highlights"},
        "other": {"description"},
    }

    for cat, fields in allowed_fields.items():
        entries = structured_resume.get(cat, [])
        new_entries = []

        for i, entry in enumerate(entries):
            if isinstance(entry, str):
                entry = {"description": entry}
            elif not isinstance(entry, dict):
                entry = {"description": str(entry)}

            # 丢掉不在允许范围内的字段
            entry = {k: v for k, v in entry.items() if k in fields}

            # 补关键字段
            for f in fields:
                if f not in entry or entry[f] in [None, ""]:
                    if f == "start_date":
                        entry[f] = "Unknown"
                    elif f == "end_date":
                        entry[f] = "Present"
                    else:
                        entry[f] = ""

            new_entries.append(entry)

        structured_resume[cat] = new_entries

    # 顶层技能
    if "skills" in structured_resume and isinstance(structured_resume["skills"], list):
        structured_resume["skills"] = normalize_skills(
            [s for s in structured_resume["skills"] if isinstance(s, str) and s.strip()]
        )

    return structured_resume

# -------------------------
# 正则兜底
# -------------------------
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
phone_pattern = r"(\+?\d[\d\s\-\(\)]{7,20})"

# utils.py
def extract_basic_info(text: str) -> dict:
    result = {"name": None, "email": None, "phone": None}

    # 只提取 email
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    if email_match:
        result["email"] = email_match.group(0)

    # 只提取 phone
    phone_match = re.search(r"\+?\d[\d\s\-()]{6,}", text)
    if phone_match:
        result["phone"] = phone_match.group(0)

    # 不处理 name，留给 parser.py 单独判断
    return result

# -------------------------
# 技能从文本中提取
# -------------------------
def extract_skills_from_text(text: str) -> list:
    if not isinstance(text, str):
        return []
    # 优先按 ":" 分割（常见 "Platforms & Tech: Ollama, Hugging Face"）
    if ":" in text:
        after = text.split(":", 1)[1]
    else:
        after = text
    # 按常见分隔符切分
    parts = re.split(r"[,\|;/、;]+", after)
    skills = []
    for p in parts:
        p = p.strip()
        # 忽略太短或包含描述性关键词的项
        if len(p) <= 1:
            continue
        if any(k in p.lower() for k in ["platform", "tech", "languages", "tools", "frameworks"]):
            continue
        # 去掉末尾多余的句号/中文句号
        p = p.rstrip("。.")
        if p:
            skills.append(p)
    return skills

def rule_based_filter(category: str, items: list) -> list[str]:
    """
    对 FAISS 检索到的候选段落做二次过滤
    返回纯文本字符串列表
    """
    keywords = {
        "教育经历": ["university", "college", "school", "gpa", "bachelor", "master", "phd", "degree", "教育", "学院", "大学"],
        "工作经历": ["company", "inc", "llc", "engineer", "analyst", "intern", "experience", "工作", "实习", "任职"],
        "项目经历": ["project", "built", "developed", "designed", "implemented", "system", "tool", "项目"],
        "技能": ["python", "sql", "tensorflow", "pytorch", "keras", "docker", "kubernetes",
                 "aws", "gcp", "azure", "spark", "hadoop", "tableau", "powerbi",
                 "sklearn", "scikit-learn", "pandas", "numpy", "seaborn", "langchain"]
    }

    filtered = []
    for item in items:
        # dict -> text
        if isinstance(item, dict):
            text = str(item.get("text", ""))
        else:
            text = str(item)
        
        text_low = text.lower()
        if any(k.lower() in text_low for k in keywords.get(category, [])):
            filtered.append(text)  # 注意这里返回的是字符串，而不是 dict

    return filtered

from copy import deepcopy

# utils.py 中新增
def merge_dicts(d1: dict, d2: dict) -> dict:
    """
    深度合并两个字典。
    - 如果 key 对应值都是 dict，则递归合并
    - 如果 key 对应值都是 list，则合并去重
    - 其他类型，使用 d2 覆盖 d1
    """
    import copy
    result = copy.deepcopy(d1)

    for key, value in d2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                # 合并去重
                result[key] = list({*result[key], *value})
            else:
                result[key] = value
        else:
            result[key] = value
    return result

# parser_fixed.py
import re
import json
import logging
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ACTION_RE = re.compile(r"\b(Built|Created|Developed|Led|Designed|Implemented)\b", re.I)
POSITION_KEYWORDS = ["intern", "engineer", "manager", "analyst", "consultant", "scientist", "developer", "research"]
COMPANY_KEYWORDS = ["llc", "inc", "company", "corp", "ltd", "co.", "technolog", "university", "school"]

def preprocess_paragraphs(paragraphs: List[str]) -> List[str]:
    out = []
    for p in paragraphs:
        if not p:
            continue
        text = " ".join(p.split())
        # handle "...2022Skills" like cases
        text = re.sub(r'(?<=\d)(?=Skills\b)', ' ', text, flags=re.I)
        out.append(text)
    logger.debug("Preprocessed paragraphs: %s", out[:12])
    return out

def extract_email(text: str) -> Optional[str]:
    m = re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None

def extract_phone(text: str) -> Optional[str]:
    candidates = re.findall(r'(\+?\d[\d\-\s\(\)]{6,}\d)', text)
    return max([c.strip() for c in candidates], key=len) if candidates else None

def parse_date_range(date_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not date_str:
        return None, None
    parts = re.split(r'\s*(?:–|-|—|to)\s*', date_str)
    if len(parts) == 1:
        yrs = re.findall(r'\b(19|20)\d{2}\b', date_str)
        if yrs:
            return yrs[0], yrs[-1] if len(yrs) > 1 else yrs[0]
        return date_str.strip(), None
    start = parts[0].strip() or None
    end = parts[1].strip() or None
    return start, end

def is_project_title(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if low in ("project", "projects", "项目", "project experience", "project:"):
        return False
    if "|" in text or "@" in text or re.search(r'\b(19|20)\d{2}\b', text):
        return False
    words = text.split()
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha_words:
        return False
    cap_count = sum(1 for w in alpha_words if w[0].isupper())
    ratio = cap_count / len(alpha_words)
    return ratio >= 0.6 and len(words) <= 8

def is_work_line(text: str) -> bool:
    if "|" not in text:
        return False
    parts = [p.strip().lower() for p in text.split("|")]
    # 同时包含公司和职位才算工作经历
    has_pos = any(any(k in p for k in POSITION_KEYWORDS) for p in parts)
    has_comp = any(any(k in p for k in COMPANY_KEYWORDS) for p in parts)
    # 必须至少有一个年份才算（防止误判）
    has_year = any(re.search(r"(19|20)\d{2}", p) for p in parts)
    return has_pos and has_comp and has_year

def parse_work_line(text: str):
    parts = [p.strip() for p in text.split("|")]
    company, position, location, date_range = None, None, None, None
    for p in parts:
        low = p.lower()
        if any(k in low for k in POSITION_KEYWORDS):
            position = p
        elif any(k in low for k in COMPANY_KEYWORDS):
            company = p
        elif re.search(r"\b[A-Z]{2}\b", p):  # 州缩写作为地点
            location = p
        elif re.search(r"(19|20)\d{2}", p):
            date_range = p
    start, end = parse_date_range(date_range) if date_range else (None, None)
    return {
        "company": company,
        "position": position,
        "location": location,
        "start_date": start,
        "end_date": end,
        "description": text,
        "highlights": []
    }

def parse_education_line(text: str):
    parts = [p.strip() for p in text.split("|")]
    school = parts[0] if parts else text
    degree = None
    grad_date = None
    for p in parts[1:]:
        if any(word in p.lower() for word in ["bachelor", "master", "phd", "degree", "art", "science", "study"]):
            degree = p
        yr = re.search(r'\b(19|20)\d{2}\b', p)
        if yr:
            grad_date = yr.group(0)
    return {"school": school, "degree": degree, "grad_date": grad_date, "description": text}

def validate_and_clean(structured: dict) -> dict:
    cleaned = {
        "name": structured.get("name"),
        "email": structured.get("email"),
        "phone": structured.get("phone"),
        "education": [],
        "work_experience": [],
        "projects": [],
        "skills": structured.get("skills", []),
        "other": []
    }

    # 教育经历
    cleaned["education"].extend([e for e in structured.get("education", []) if e])

    # 工作经历
    cleaned["work_experience"].extend([w for w in structured.get("work_experience", []) if w])

    # 项目 - 过滤掉伪项目
    for p in structured.get("projects", []):
        if not p:
            continue
        title = (p.get("title") or "").lower()
        if any(x in title for x in ["platforms & tech", "frameworks", "skills", "sourced code"]):
            cleaned["other"].append(p)
        else:
            cleaned["projects"].append(p)

    # other 里提取技能
    new_other = []
    for o in structured.get("other", []):
        desc = o if isinstance(o, str) else o.get("description", "")
        if desc and any(k in desc.lower() for k in ["frameworks", "libraries", "tech", "tools"]):
            # 把冒号后的部分拆成技能
            parts = desc.split(":")[-1]
            skills = [s.strip() for s in parts.split(",") if s.strip()]
            cleaned["skills"].extend(skills)
        else:
            new_other.append(o)
    cleaned["other"] = new_other

    # 兜底：other 里搬运到 work_experience
    moved, remain = [], []
    for o in cleaned["other"]:
        desc = o if isinstance(o, str) else o.get("description", "")
        if "|" in desc and (any(k in desc.lower() for k in COMPANY_KEYWORDS + POSITION_KEYWORDS) and re.search(r"(19|20)\d{2}", desc)):
            moved.append(parse_work_line(desc))
        else:
            remain.append(o)
    cleaned["work_experience"].extend(moved)
    cleaned["other"] = remain

    # 最后统一标准化技能
    cleaned["skills"] = normalize_skills(cleaned["skills"])

    return cleaned

# -----------------------
# 测试 auto_fill_fields
# -----------------------
if __name__ == "__main__":
    test_resume = {
        "name": "Zhenyu Zhang",
        "email": "Zhang.zhenyu6@northeastern.edu",
        "phone": "+1860234-7101",
        "education": [{"school": "Northeastern University", "degree": "Master of Science in Computer Science", "grad_date": "2025", "description": "Northeastern University | Master of Science in Computer Science | 2025"}],
        "work_experience": [{"title": "Data Science Intern", "company": "Google LLC", "start_date": "Jun 2024", "end_date": "Aug 2024", "description": "Data Science Intern | Google LLC | Jun 2024 – Aug 2024", "Professional Experience": None, "Industry Experience": None, "Experience": None}],
        "projects": [{"project_title": "YouTube Recommendation System Built a", "start_date": None, "end_date": None, "project_content": "YouTube Recommendation System Built a recommendation model using DNN and LightGBM...", "Projects": None, "Project Experience": None}],
        "skills": ["Python", "SQL", "TensorFlow", "PyTorch"],
        "other": [{"description": "Zhenyu Zhang | Email: Zhang.zhenyu6@northeastern.edu | Phone: +1860234-7101"}]
    }

    filled_resume = auto_fill_fields(test_resume)
    print("\nFinal structured resume:\n", filled_resume)
