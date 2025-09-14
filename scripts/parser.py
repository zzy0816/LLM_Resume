import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import re
from utils import CATEGORY_FIELDS, normalize_category, normalize_skills, auto_fill_fields, SKILL_NORMALIZATION, extract_basic_info, extract_skills_from_text
from ner import run_ner_batch
from utils_parser import semantic_fallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def parse_projects_blocks(raw_lines: list[str]) -> list[dict]:
    """
    将原始文本行拆分为项目块（title + content）
    - 标题行：不以动词开头，长度小于100
    - 内容行：以 '-' 开头或动词开头的描述
    """
    projects = []
    current_title = None
    current_content = []

    action_verbs = ("built", "created", "used", "collected", "led", "fine-tuned", "developed")

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # 新标题行条件：不以动作动词开头，长度<100
        if not line.lower().startswith(action_verbs) and len(line) <= 100:
            # 保存前一个项目
            if current_title:
                projects.append({
                    "project_title": current_title,
                    "project_content": "\n".join(current_content)
                })
            current_title = line
            current_content = []
        else:
            current_content.append(line)

    # 保存最后一个项目
    if current_title:
        projects.append({
            "project_title": current_title,
            "project_content": "\n".join(current_content)
        })

    return projects

# -------------------------
# preprocess_paragraphs_for_projects
# -------------------------
def preprocess_paragraphs_for_projects(paragraphs: list[str]) -> list[str]:
    """
    将连续的项目段落合并，生成按项目拆分的段落列表
    - 项目标题：长度<=100 且不以动作动词开头，或包含 'project' / '项目'
    - 内容行：以 '-' 开头或动作动词开头
    """
    merged_paragraphs = []
    ACTION_VERBS = ("built", "created", "used", "collected", "led", "fine-tuned", "developed")

    skip_next = 0
    for i, para in enumerate(paragraphs):
        if skip_next:
            skip_next -= 1
            continue

        para_clean = para.strip()
        if not para_clean:
            continue

        para_lower = para_clean.lower()
        # 判断是否为项目段落起始
        is_project_start = ("project" in para_lower) or ("项目" in para_lower) or \
                           (len(para_clean) <= 100 and not para_lower.startswith(ACTION_VERBS))

        if is_project_start:
            project_lines = [para_clean]
            j = i + 1
            while j < len(paragraphs):
                next_para = paragraphs[j].strip()
                if not next_para:
                    j += 1
                    continue
                next_lower = next_para.lower()
                # 遇到非项目段落或教育/工作标题就停止
                if any(k in next_lower for k in ["university","college","bachelor","master","phd",
                                                 "intern","engineer","manager","工作","实习","任职"]):
                    break
                project_lines.append(next_para)
                j += 1

            # 使用 parse_projects_blocks 生成项目块
            projects_blocks = parse_projects_blocks(project_lines)
            for blk in projects_blocks:
                merged_paragraphs.append(blk["project_title"] + "\n" + blk["project_content"])

            skip_next = len(project_lines) - 1
        else:
            merged_paragraphs.append(para_clean)

    return merged_paragraphs

# -------------------------
# classify_paragraphs
# -------------------------
def classify_paragraphs(paragraph: str, structured: dict, ner_results=None, sem_cat=None):
    para_clean = paragraph.strip().replace("\r", " ").replace("\n", " ")
    if not para_clean:
        return "other", {}

    para_lower = para_clean.lower()

    # ---- 基础信息 ----
    from utils import extract_basic_info
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

    # ---- 判断类别 ----
    edu_keywords = ["university", "college", "学院", "大学", "bachelor", "master", "phd", "ma", "ms", "mba"]
    work_keywords = ["intern", "engineer", "manager", "responsible", "工作", "实习", "任职", "developer", "consultant"]
    proj_keywords = ["project", "built", "created", "developed", "led", "designed", "implemented"]

    category = None
    data = init_data_for_category("other", para_clean)

    if any(k in para_lower for k in edu_keywords):
        category = "education"
        data = init_data_for_category(category, para_clean)
        parts = [p.strip() for p in para_clean.split("|")]
        if len(parts) >= 2:
            data["school"] = parts[0]
            data["degree"] = parts[1]

        date_pattern = re.compile(
            r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?"
            r"(\d{4})",
            re.I
        )
        match = date_pattern.search(para_clean)
        if match:
            month, year = match.groups()
            data["grad_date"] = f"{month} {year}" if month else year
    elif any(k in para_lower for k in proj_keywords):
        category = "projects"
        data = init_data_for_category(category, para_clean)
        # 尝试抽取标题
        title_match = re.match(r"^(.*?)[:\-]", para_clean)
        if title_match:
            data["project_title"] = title_match.group(1).strip()
        # 尝试抽取日期
        date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})\s*–\s*(Present|\d{4})", para_clean, re.I)
        if date_match:
            data["start_date"] = date_match.group(1)
            data["end_date"] = date_match.group(2)
        # 内容
        data["project_content"] = para_clean
    elif any(k in para_lower for k in work_keywords):
        category = "work_experience"
        data = init_data_for_category(category, para_clean)
        parts = [p.strip() for p in para_clean.split("|")]
        if len(parts) >= 2:
            data["company"] = parts[0]
            data["title"] = parts[1]

        # 使用统一正则解析日期
        date_pattern = re.compile(
            r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?"
            r"(\d{4})\s*[-–]\s*"
            r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?"
            r"(Present|\d{4})",
            re.I
        )
        match = date_pattern.search(para_clean)
        if match:
            start_month, start_year, end_month, end_year = match.groups()
            data["start_date"] = f"{start_month} {start_year}" if start_month else start_year
            data["end_date"] = f"{end_month} {end_year}" if end_month else end_year
    else:
        category = "other"
        data = init_data_for_category(category, para_clean)

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
            import logging
            logging.warning(f"[NER ERROR] {e}")

    # ---- 技能关键词补全 ----
    skill_keywords = [
        "python","sql","pandas","numpy","scikit","sklearn","tensorflow",
        "pytorch","keras","docker","kubernetes","aws","gcp","azure",
        "spark","hadoop","tableau","powerbi","llm","llama","hugging"
    ]
    if "skills" in data:
        for kw in skill_keywords:
            if kw in para_lower and kw not in data["skills"]:
                data["skills"].append(SKILL_NORMALIZATION.get(kw, kw.upper() if kw in ["sql","llm","aws","hugging"] else kw.capitalize()))

    return normalize_category(category), data

# -------------------------
# parse_resume_to_structured
# -------------------------
def parse_resume_to_structured(paragraphs: list, file_name: str = None):
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

    paragraphs = preprocess_paragraphs_for_projects(paragraphs)
    ner_results_batch = run_ner_batch(paragraphs)
    semantic_cats = semantic_fallback(paragraphs, file_name=file_name)

    current_work = None
    for para, ner_results, sem_cat in zip(paragraphs, ner_results_batch, semantic_cats):
        para_clean = para.strip().replace("\r", " ").replace("\n", " ")
        if not para_clean:
            continue

        # ---- 基础信息 ----
        info = extract_basic_info(para_clean)
        if info:
            structured["email"] = structured.get("email") or info.get("email")
            structured["phone"] = structured.get("phone") or info.get("phone")
            structured["name"] = structured.get("name") or info.get("name")
            continue
        if any(k in para_clean.lower() for k in ["linkedin", "github", "电话", "邮箱"]):
            continue

        para_lower = para_clean.lower()

        # ---- 教育经历 ----
        edu_keywords = ["university", "college", "学院", "大学", "bachelor", "master", "phd", "ma", "ms", "mba"]
        if any(k in para_lower for k in edu_keywords):
            parts = [p.strip() for p in para_clean.split("|")]
            entry = {
                "school": parts[0] if len(parts) > 0 else "N/A",
                "degree": parts[1] if len(parts) > 1 else "N/A",
                "grad_date": "Unknown",
                "description": para_clean
            }

            # 增强日期匹配：支持完整月份可选 + 年份
            date_pattern = re.compile(
                r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?"
                r"(\d{4})",
                re.I
            )
            match = date_pattern.search(para_clean)
            if match:
                month, year = match.groups()
                entry["grad_date"] = f"{month} {year}" if month else year

            structured["education"].append(entry)

        # ---- 工作经历 ----
        work_keywords = ["intern", "engineer", "manager", "responsible", "工作", "实习", "任职", "developer", "consultant"]
        if any(k in para_lower for k in work_keywords):
            parts = [p.strip() for p in para_clean.split("|")]
            current_work = {
                "company": parts[0] if len(parts) > 0 else "N/A",
                "title": parts[1] if len(parts) > 1 else "N/A",
                "start_date": "Unknown",
                "end_date": "Present",
                "description": para_clean
            }

            # 增强日期匹配：支持完整月份名或缩写 + 年份
            date_pattern = re.compile(
                r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?"
                r"(\d{4})\s*[-–]\s*"
                r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?"
                r"(Present|\d{4})",
                re.I
            )

            match = date_pattern.search(para_clean)
            if match:
                start_month, start_year, end_month, end_year = match.groups()
                start_date = f"{start_month} {start_year}" if start_month else start_year
                end_date = f"{end_month} {end_year}" if end_month else end_year
                current_work["start_date"] = start_date
                current_work["end_date"] = end_date

            structured["work_experience"].append(current_work)

        # ---- 独立项目经历 ----
        project_keywords = ["project", "项目"]

        if any(k in para_lower for k in project_keywords) or sem_cat == "project":
            # 收集所有连续项目段落
            project_lines = [para_clean]
            idx = paragraphs.index(para) + 1
            while idx < len(paragraphs):
                next_para = paragraphs[idx].strip()
                if not next_para:
                    idx += 1
                    continue
                # 遇到非项目段落或技能/工作/教育则停止
                next_lower = next_para.lower()
                if any(k in next_lower for k in ["university","college","bachelor","master","phd","intern","engineer","manager","工作","实习","任职"]):
                    break
                project_lines.append(next_para)
                idx += 1

            # 使用 parse_projects_blocks 生成项目块
            projects_blocks = parse_projects_blocks(project_lines)
            structured["projects"].extend(projects_blocks)

            # 跳过已经收集的段落
            for _ in range(len(project_lines) - 1):
                next(paragraphs, None)
            continue

        # ---- 技能 ----
        skill_keywords = [
            "python","sql","pandas","numpy","scikit","sklearn","tensorflow",
            "pytorch","keras","docker","kubernetes","aws","gcp","azure",
            "spark","hadoop","tableau","powerbi","llm","llama","hugging"
        ]
        if any(k in para_lower for k in skill_keywords):
            extracted = extract_skills_from_text(para_clean)
            structured["skills"].extend(extracted)
            continue

        # ---- NER 补充 ----
        if ner_results:
            try:
                for ent in ner_results:
                    label = ent["entity_group"].lower()
                    val = ent["word"].strip()
                    if label == "per" and structured.get("name") is None:
                        structured["name"] = val
                    # 其他字段同原逻辑
            except Exception as e:
                logging.warning(f"[NER ERROR] {e}")

        # ---- 其他 ----
        structured["other"].append({"description": para_clean})

    # 去重 & 标准化技能
    structured["skills"] = normalize_skills(structured["skills"])
    structured = auto_fill_fields(structured)
    return structured
