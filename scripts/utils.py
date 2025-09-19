import re

# -------------------------
# 分类段落
# -------------------------
CATEGORY_FIELDS = {
    "work_experience": ["title","company","start_date","end_date","description", "Professional Experience", "Industry Experience", "Experience"],
    "education": ["school","degree","grad_date","description"],
    "projects": ["project_title","start_date","end_date","project_content", "Projects", "Project Experience"],
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

import logging

def auto_fill_fields(structured_resume: dict) -> dict:
    """自动补全字段 & 技能（安全处理，不覆盖顶层 skills）"""
    logging.basicConfig(level=logging.DEBUG)

    # 只处理条目列表，不处理顶层 skills
    CATEGORIES_TO_FILL = ["education", "work_experience", "projects", "other"]

    for cat in CATEGORIES_TO_FILL:
        fields = CATEGORY_FIELDS.get(cat, [])
        entries = structured_resume.get(cat, [])
        logging.debug(f"Processing category: {cat} with fields: {fields}")
        new_entries = []

        for i, entry in enumerate(entries):
            logging.debug(f"Processing entry {i}: {entry} (type={type(entry)})")

            # 字符串包装为 dict
            if isinstance(entry, str):
                entry = {f: None for f in fields}
                if "description" in fields:
                    entry["description"] = entry
                if "skills" in fields:
                    entry["skills"] = []
            elif not isinstance(entry, dict):
                # 非法类型
                entry = {f: None for f in fields}
                if "description" in fields:
                    entry["description"] = str(entry)
                if "skills" in fields:
                    entry["skills"] = []

            # 补全字段
            for f in fields:
                if f not in entry or entry[f] is None:
                    if f == "skills":
                        entry[f] = entry.get("skills") or []
                    elif f in ["start_date", "grad_date"]:
                        entry[f] = "Unknown"
                    elif f == "end_date":
                        entry[f] = "Present"
                    else:
                        entry[f] = "N/A"

            # 条目内部技能标准化
            if "skills" in entry and isinstance(entry["skills"], list):
                entry["skills"] = normalize_skills([s for s in entry["skills"] if isinstance(s, str) and s.strip()])

            logging.debug(f"Completed entry: {entry}")
            new_entries.append(entry)

        structured_resume[cat] = new_entries
        logging.debug(f"After processing category {cat}: {structured_resume[cat]}")

    # 顶层技能标准化（仅标准化，不覆盖原有列表）
    if "skills" in structured_resume and isinstance(structured_resume["skills"], list):
        top_skills = [s for s in structured_resume["skills"] if isinstance(s, str) and s.strip()]
        logging.debug(f"Top-level skills before normalize: {top_skills}")
        structured_resume["skills"] = normalize_skills(top_skills)
        logging.debug(f"Top-level skills after normalize: {structured_resume['skills']}")

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

import re

def rule_based_filter(category: str, candidates: list[str]) -> list[str]:
    """
    对 FAISS 检索到的候选段落做二次过滤，避免噪声进入
    """
    keywords = {
        "教育经历": ["university", "college", "school", "gpa", "bachelor", "master", "phd", "degree", "教育", "学院", "大学"],
        "工作经历": ["company", "inc", "llc", "engineer", "analyst", "intern", "experience", "工作", "实习", "任职"],
        "项目经历": ["project", "built", "developed", "designed", "implemented", "system", "tool", "项目"],
        "技能": ["python", "sql", "tensorflow", "pytorch", "keras", "docker", "kubernetes",
                 "aws", "gcp", "azure", "spark", "hadoop", "tableau", "powerbi",
                 "sklearn", "scikit-learn", "pandas", "numpy", "seaborn", "langchain"]
    }
    result = []
    for c in candidates:
        c_low = c.lower()
        if any(k in c_low for k in keywords.get(category, [])):
            result.append(c)
    return result


def validate_and_clean(structured: dict) -> dict:
    """
    对最终 structured_resume 做清理，避免分类错误
    """
    cleaned = structured.copy()

    # 教育经历过滤：必须包含大学/学院关键词
    edu_valid = []
    for edu in cleaned.get("education", []):
        text = (edu.get("school") or "") + " " + (edu.get("description") or "")
        if re.search(r"(university|college|学院|大学)", text, re.I):
            edu_valid.append(edu)
    cleaned["education"] = edu_valid

    # 工作经历过滤：必须包含 company/engineer/intern/工作 等关键词
    work_valid = []
    for work in cleaned.get("work_experience", []):
        text = (work.get("company") or "") + " " + (work.get("description") or "")
        if re.search(r"(company|engineer|intern|analyst|工作|实习|任职)", text, re.I):
            work_valid.append(work)
    cleaned["work_experience"] = work_valid

    # 项目经历过滤：必须包含 project/项目
    proj_valid = []
    for proj in cleaned.get("projects", []):
        text = (proj.get("project_title") or "") + " " + (proj.get("project_content") or "")
        if re.search(r"(project|项目)", text, re.I):
            proj_valid.append(proj)
    cleaned["projects"] = proj_valid

    # 技能去重 + 白名单
    skill_whitelist = set([
        "python","sql","pandas","numpy","scikit-learn","sklearn","tensorflow",
        "pytorch","keras","docker","kubernetes","aws","gcp","azure",
        "spark","hadoop","tableau","powerbi","llm","huggingface","langchain",
        "seaborn"
    ])
    skills = cleaned.get("skills", [])
    skills = [s.strip() for s in skills if s and s.lower() in skill_whitelist]
    cleaned["skills"] = sorted(set(skills), key=lambda x: skills.index(x))

    return cleaned

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
