import copy
import json
import logging
import os
import random
import re
import sys
from typing import List, Optional, Tuple
from typing import Iterable

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "ner_service",
            "message": record.getMessage(),
            "request_id": str(random.randint(1000, 9999)),
        }
        return json.dumps(log)


# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)

# 设置日志 handler
handler = logging.FileHandler("logs/app.log")
handler.setFormatter(JsonFormatter())

logger = logging.getLogger()  # root logger
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ACTION_RE = re.compile(
    r"\b(Built|Created|Developed|Led|Designed|Implemented)\b", re.I
)
POSITION_KEYWORDS = [
    "intern",
    "engineer",
    "manager",
    "analyst",
    "consultant",
    "scientist",
    "developer",
    "research",
]
COMPANY_KEYWORDS = [
    "llc",
    "inc",
    "company",
    "corp",
    "ltd",
    "co.",
    "technolog",
    "university",
    "school",
]


POSITION_KEYWORDS = [
    "intern",
    "engineer",
    "manager",
    "analyst",
    "consultant",
    "scientist",
    "developer",
    "research",
]
COMPANY_KEYWORDS = [
    "llc",
    "inc",
    "company",
    "corp",
    "ltd",
    "co.",
    "technolog",
    "university",
    "school",
]

# -------------------------
# 分类段落
# -------------------------
CATEGORY_FIELDS = {
    "work_experience": [
        "company",
        "position",
        "location",
        "start_date",
        "end_date",
        "description",
        " highlights",
    ],
    "education": ["school", "degree", "grad_date", "description"],
    "projects": ["title", "highlights"],
    "skills": ["skills"],
    "other": ["description"],
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
        "other": "other",
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
    "aws": "AWS",
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
        "work_experience": {
            "company",
            "position",
            "location",
            "start_date",
            "end_date",
            "description",
            "highlights",
        },
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
    if "skills" in structured_resume and isinstance(
        structured_resume["skills"], list
    ):
        structured_resume["skills"] = normalize_skills(
            [
                s
                for s in structured_resume["skills"]
                if isinstance(s, str) and s.strip()
            ]
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
        if any(
            k in p.lower()
            for k in ["platform", "tech", "languages", "tools", "frameworks"]
        ):
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
        "教育经历": [
            "university",
            "college",
            "school",
            "gpa",
            "bachelor",
            "master",
            "phd",
            "degree",
            "教育",
            "学院",
            "大学",
        ],
        "工作经历": [
            "company",
            "inc",
            "llc",
            "engineer",
            "analyst",
            "intern",
            "experience",
            "工作",
            "实习",
            "任职",
        ],
        "项目经历": [
            "project",
            "built",
            "developed",
            "designed",
            "implemented",
            "system",
            "tool",
            "项目",
        ],
        "技能": [
            "python",
            "sql",
            "tensorflow",
            "pytorch",
            "keras",
            "docker",
            "kubernetes",
            "aws",
            "gcp",
            "azure",
            "spark",
            "hadoop",
            "tableau",
            "powerbi",
            "sklearn",
            "scikit-learn",
            "pandas",
            "numpy",
            "seaborn",
            "langchain",
        ],
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
            filtered.append(text)  # 这里返回的是字符串，而不是 dict

    return filtered


def merge_dicts(d1: dict, d2: dict) -> dict:
    """
    深度合并两个字典。
    - 如果 key 对应值都是 dict，则递归合并
    - 如果 key 对应值都是 list，则合并去重
    - 其他类型，使用 d2 覆盖 d1
    """

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


def preprocess_paragraphs(paragraphs: List[str]) -> List[str]:
    out = []
    for p in paragraphs:
        if not p:
            continue
        text = " ".join(p.split())
        # handle "...2022Skills" like cases
        text = re.sub(r"(?<=\d)(?=Skills\b)", " ", text, flags=re.I)
        out.append(text)
    logger.debug("Preprocessed paragraphs: %s", out[:12])
    return out


def extract_email(text: str) -> Optional[str]:
    m = re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def extract_phone(text: str) -> Optional[str]:
    candidates = re.findall(r"(\+?\d[\d\-\s\(\)]{6,}\d)", text)
    return (
        max([c.strip() for c in candidates], key=len) if candidates else None
    )


# ----------------- 日期解析 -----------------
def parse_date_range(
    date_range: Optional[str], next_line: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    支持 'Sep 2022 – Jul' 或 'Jun 2021 – Aug 2022' 或 'May 2025 – Present'。
    如果 end 没有年份，则推断和 start 同年或加1年。
    """
    if not date_range:
        return None, None
    date_range = date_range.replace("–", "-").strip()
    parts = [p.strip() for p in date_range.split("-")]
    start, end = parts[0] if parts else None, (
        parts[1] if len(parts) > 1 else None
    )

    # --- 处理 end 为月份但无年份 ---
    if start and end:
        start_match = re.search(r"([A-Za-z]+)\s*(\d{4})", start)
        end_match = re.search(r"([A-Za-z]+)\s*(\d{4})", end)

        if start_match:
            start_month, start_year = start_match.group(1), int(
                start_match.group(2)
            )
        else:
            start_month, start_year = None, None

        if end_match:
            end_month, end_year = end_match.group(1), int(end_match.group(2))
        else:
            end_month = end.strip()
            end_year = start_year  # 推断同年

            # 如果 end_month 是 "Present"，跳过月份比较
            if end_month != "Present":
                month_order = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                if (
                    start_month
                    and end_month
                    and month_order.index(end_month)
                    < month_order.index(start_month)
                ):
                    end_year += 1

        end = (
            f"{end_month} {end_year}"
            if end_month and end_year and end_month != "Present"
            else end
        )

    # 如果 next_line 是年份，也可以覆盖 end
    if next_line:
        next_line = next_line.strip()
        if re.match(r"^(19|20)\d{2}$", next_line):
            if end and end != "Present":
                end = f"{end.split()[0]} {next_line}"
            else:
                end = next_line

    return start, end


def is_project_title(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if low in (
        "project",
        "projects",
        "项目",
        "project experience",
        "project:",
    ):
        return False
    if "|" in text or "@" in text or re.search(r"\b(19|20)\d{2}\b", text):
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


def parse_work_line(text: str, next_line: Optional[str] = None):
    parts = [p.strip() for p in text.split("|")]
    company, position, location, date_range = None, None, None, None
    for p in parts:
        low = p.lower()
        if any(k in low for k in POSITION_KEYWORDS):
            position = p
        elif any(k in low for k in COMPANY_KEYWORDS):
            company = p
        elif re.search(r"\b[A-Z]{2}\b", p):  # 州缩写
            location = p
        elif re.search(r"(19|20)\d{2}", p) or re.search(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", p
        ):
            date_range = p
    start, end = parse_date_range(date_range, next_line)
    return {
        "company": company,
        "position": position,
        "location": location,
        "start_date": start,
        "end_date": end,
        "description": text,
        "highlights": [],
    }


def parse_education_line(text: str):
    parts = [p.strip() for p in text.split("|")]
    school = parts[0] if parts else text
    degree = None
    grad_date = None
    for p in parts[1:]:
        if any(
            word in p.lower()
            for word in [
                "bachelor",
                "master",
                "phd",
                "degree",
                "art",
                "science",
                "study",
            ]
        ):
            degree = p
        # 支持月份 + 年份
        yr = re.search(
            r"((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*)?\d{4}", p
        )
        if yr:
            grad_date = yr.group(0)
    return {
        "school": school,
        "degree": degree,
        "grad_date": grad_date,
        "description": text,
    }


def validate_and_clean(structured: dict) -> dict:
    cleaned = {
        "name": structured.get("name"),
        "email": structured.get("email"),
        "phone": structured.get("phone"),
        "education": [],
        "work_experience": [],
        "projects": [],
        "skills": structured.get("skills", []),
        "other": [],
    }

    # 教育经历
    cleaned["education"].extend(
        [e for e in structured.get("education", []) if e]
    )

    # 工作经历
    cleaned["work_experience"].extend(
        [w for w in structured.get("work_experience", []) if w]
    )

    # 项目 - 过滤掉伪项目
    for p in structured.get("projects", []):
        if not p:
            continue
        title = (p.get("title") or "").lower()
        if any(
            x in title
            for x in [
                "platforms & tech",
                "frameworks",
                "skills",
                "sourced code",
            ]
        ):
            cleaned["other"].append(p)
        else:
            cleaned["projects"].append(p)

    # other 里提取技能
    new_other = []
    for o in structured.get("other", []):
        desc = o if isinstance(o, str) else o.get("description", "")
        if desc and any(
            k in desc.lower()
            for k in ["frameworks", "libraries", "tech", "tools"]
        ):
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
        if "|" in desc and (
            any(
                k in desc.lower() for k in COMPANY_KEYWORDS + POSITION_KEYWORDS
            )
            and re.search(r"(19|20)\d{2}", desc)
        ):
            moved.append(parse_work_line(desc))
        else:
            remain.append(o)
    cleaned["work_experience"].extend(moved)
    cleaned["other"] = remain

    # 最后统一标准化技能
    cleaned["skills"] = normalize_skills(cleaned["skills"])

    return cleaned


def fix_resume_dates(structured_resume: dict) -> dict:
    """
    修复教育和工作经历日期：
    1. 教育经历：优先使用 description 中 Month Year，否则用年份。
    2. 工作经历：优先使用 description 中 Month Year，end_date 会根据 highlights 推断，保证 end_date >= start_date。
    3. highlights 中纯年份或 "Present" 会被清理。
    """
    if not structured_resume:
        return {}

    # ---- 教育经历 ----
    month_year_pattern = (
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(19|20)\d{2}"
    )
    year_pattern = r"(19|20)\d{2}"

    for edu in structured_resume.get("education", []):
        desc = edu.get("description", "")
        desc_clean = re.sub(r"[\r\n]+", " ", desc)
        desc_clean = re.sub(r"\s+", " ", desc_clean).strip()

        match = re.search(month_year_pattern, desc_clean)
        if match:
            edu["grad_date"] = match.group(0)
        else:
            match_year = re.search(year_pattern, desc_clean)
            if match_year:
                edu["grad_date"] = match_year.group(0)

    # ---- 工作经历 ----
    work_exp = structured_resume.get("work_experience", [])
    structured_resume["work_experience"] = fix_work_dates(work_exp)

    return structured_resume


def fix_work_dates(work_experience: list) -> list:
    month_pattern = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    year_pattern = r"(19|20)\d{2}"

    for job in work_experience:
        desc = job.get("description", "")
        highlights = job.get("highlights", [])

        # 合并换行并压缩空格
        desc_clean = re.sub(r"[\r\n]+", " ", desc)
        desc_clean = re.sub(r"\s+", " ", desc_clean).strip()

        # 收集 highlights 年份
        highlight_years = []
        cleaned_highlights = []
        for h in highlights:
            h_strip = h.strip()
            if re.fullmatch(r"\d{4}", h_strip):
                highlight_years.append(int(h_strip))
            else:
                cleaned_highlights.append(h)
                highlight_years.extend(
                    [int(y) for y in re.findall(year_pattern, h_strip)]
                )
        job["highlights"] = cleaned_highlights

        # 提取 description 中所有 Month Year
        month_year_matches = re.findall(
            rf"{month_pattern}\s+{year_pattern}", desc_clean
        )
        # 提取 description 中只有 Month 的部分
        months_only_matches = re.findall(
            rf"{month_pattern}(?=\s*(–|$))", desc_clean
        )

        # --- start_date ---
        start_date = job.get("start_date")
        if not start_date or str(start_date).lower() in ["null", "n/a", ""]:
            start_date = (
                month_year_matches[0]
                if month_year_matches
                else f"{months_only_matches[0] if months_only_matches else 'Jan'} 2020"
            )

        # --- end_date ---
        end_date = job.get("end_date")
        if not end_date or str(end_date).lower() in ["null", "n/a", ""]:
            if len(month_year_matches) >= 2:
                end_date = month_year_matches[1]
            elif months_only_matches:
                # Month-only，年份 = start_date 年 +1
                start_m, start_y = start_date.split()
                start_y = int(start_y)
                # 找到 end 月份，如果只有一个 month-only，用 start month +1 年
                end_m = (
                    months_only_matches[1]
                    if len(months_only_matches) > 1
                    else months_only_matches[0]
                )
                end_date = f"{end_m} {start_y + 1}"
            elif highlight_years:
                end_date = f"{start_date.split()[0]} {max(highlight_years)}"
            else:
                end_date = "Present"

        # --- 确保 end_date >= start_date ---
        try:
            start_m, start_y = start_date.split()
            start_y = int(start_y)
            if end_date.lower() != "present":
                end_m, end_y = end_date.split()
                end_y = int(end_y)
                if end_y < start_y:
                    end_y = start_y + 1
                    end_date = f"{end_m} {end_y}"
        except Exception:
            pass

        job["start_date"] = start_date
        job["end_date"] = end_date

    return work_experience

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "ner_service",
            "message": record.getMessage(),
            "request_id": str(random.randint(1000, 9999)),
        }
        return json.dumps(log, ensure_ascii=False)
    
def setup_logging(
    log_file: str = "logs/app.log",
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    root_level: int = logging.INFO,
    noisy_libs: Iterable[str] | None = ("transformers", "tqdm"),
    clear_existing_handlers: bool = True,
) -> None:
    """
    统一初始化 logging：
      - 清理已有 handler（避免重复打印）
      - 文件日志为 JSON（便于 ELK 等处理）
      - 控制台日志为人类可读格式（便于调试）
      - 可选降噪指定第三方库的日志级别

    使用：
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("start")
    注意：请在其它可能配置 logging 的模块导入之前调用此函数（例如 transformers）。
    """
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logger = logging.getLogger()
    # 设置 root level（决定 handler 是否接收消息）
    logger.setLevel(root_level)

    # 清理已有 handler（关键，避免重复输出）
    if clear_existing_handlers and logger.hasHandlers():
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # 文件 handler -> JSON
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    # 控制台 handler -> 可读格式
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 可选：降低某些 noisy 第三方库的日志级别
    if noisy_libs:
        for lib in noisy_libs:
            try:
                logging.getLogger(lib).setLevel(logging.WARNING)
            except Exception:
                pass
