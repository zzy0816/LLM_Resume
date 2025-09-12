import re

# -------------------------
# 分类段落
# -------------------------
CATEGORY_FIELDS = {
    "work_experience": ["title","company","start_date","end_date","description"],
    "education": ["school","degree","grad_date","description"],
    "projects": ["project_title","start_date","end_date","project_content"],
    "skills": ["skills"],
    "other": ["description"]
}

# -------------------------
# 分类归一化
# -------------------------
def normalize_category(cat: str) -> str:
    mapping = {
        "work": "work_experience",
        "workexperience": "work_experience",
        "work_experience": "work_experience",
        "projects": "projects",
        "project": "projects",
        "education": "education",
        "edu": "education",
        "skills": "skills",
        "skill": "skills",
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
    skills_set = set()
    for s in skills:
        if not isinstance(s, str):
            continue
        s_clean = s.strip()
        if not s_clean:
            continue

        s_lower = s_clean.lower()

        # 如果在标准化词典里，直接替换
        if s_lower in SKILL_NORMALIZATION:
            skills_set.add(SKILL_NORMALIZATION[s_lower])
            continue

        # 否则做基本规则化
        if s_clean.isupper():
            skills_set.add(s_clean)
        else:
            skills_set.add(s_clean.capitalize())

    return sorted(list(skills_set))

# -------------------------
# 自动补全工作/教育/项目字段
# -------------------------
def auto_fill_fields(structured_resume: dict) -> dict:
    """
    对解析后的结构化信息进行补全：
    - 工作经历、教育经历、项目经历补全缺失字段
    - 技能字段统一标准化
    """
    for cat, fields in CATEGORY_FIELDS.items():
        new_entries = []
        for entry in structured_resume.get(cat, []):
            # 如果 entry 是字符串，则包装为 dict
            if isinstance(entry, str):
                entry_dict = {f: None for f in fields}
                if "description" in fields:
                    entry_dict["description"] = entry
                if "skills" in fields:
                    entry_dict["skills"] = []
                entry = entry_dict
            # 如果 entry 是 dict，就确保包含所有字段
            if isinstance(entry, dict):
                for f in fields:
                    if f not in entry or entry[f] is None:
                        if f == "skills":
                            entry[f] = []
                        elif f in ["start_date", "grad_date"]:
                            entry[f] = "Unknown"
                        elif f == "end_date":
                            entry[f] = "Present"
                        else:
                            entry[f] = "N/A"
                new_entries.append(entry)
            else:
                # 非法类型，统一转为 description dict
                entry_dict = {f: None for f in fields}
                if "description" in fields:
                    entry_dict["description"] = str(entry)
                for f in fields:
                    if entry_dict[f] is None:
                        if f == "skills":
                            entry_dict[f] = []
                        elif f in ["start_date", "grad_date"]:
                            entry_dict[f] = "Unknown"
                        elif f == "end_date":
                            entry_dict[f] = "Present"
                        else:
                            entry_dict[f] = "N/A"
                new_entries.append(entry_dict)
        structured_resume[cat] = new_entries

    # 技能标准化
    if "skills" in structured_resume:
        structured_resume["skills"] = [s for s in structured_resume["skills"] if isinstance(s, str)]
        structured_resume["skills"] = normalize_skills(structured_resume["skills"])

    return structured_resume

# -------------------------
# 正则兜底
# -------------------------
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
phone_pattern = r"(\+?\d[\d\s\-\(\)]{7,20})"

def extract_basic_info(text: str) -> dict:
    result = {}
    # 提取 email
    email_match = re.search(email_pattern, text)
    if email_match:
        result["email"] = email_match.group()
    # 提取电话
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        phone_clean = re.sub(r"[\s\(\)]", "", phone_match.group())
        result["phone"] = phone_clean
    # 兜底提取名字（假设首行或 email 上方）
    if "email" in result:
        first_line = text.split("\n")[0].strip()
        if first_line and not re.search(email_pattern, first_line):
            name_match = re.match(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)", first_line)
            if name_match:
                result["name"] = name_match.group(1)
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
