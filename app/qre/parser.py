import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import re
import json
import logging
from typing import List

from app.utils.utils import (
    preprocess_paragraphs,
    extract_email,
    extract_phone,
    is_work_line,
    parse_education_line,
    parse_work_line,
    is_project_title,
)
from app.qre.ner import run_ner_batch 
import logging, json, random, time, os

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "ner_service",
            "message": record.getMessage(),
            "request_id": str(random.randint(1000, 9999))
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

ACTION_RE = re.compile(r"\b(Built|Created|Developed|Led|Designed|Implemented)\b", re.I)
POSITION_KEYWORDS = ["intern", "engineer", "manager", "analyst", "consultant", "scientist", "developer", "research"]
COMPANY_KEYWORDS = ["llc", "inc", "company", "corp", "ltd", "co.", "technolog", "university", "school"]


def parse_resume_to_structured(paragraphs: List[str]) -> dict:
    paragraphs = preprocess_paragraphs(paragraphs)

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
        
    # -------- 1. 先用规则解析 --------

    current_section = None
    last_work = None
    last_project = None

    for idx, raw in enumerate(paragraphs):
        text = raw.strip()
        if not text:
            continue
        low = text.lower()

        # -------------------- 基本信息 --------------------
        if not structured["email"]:
            em = extract_email(text)
            if em:
                structured["email"] = em
                logger.debug("Extracted email: %s", em)
        if not structured["phone"]:
            ph = extract_phone(text)
            if ph:
                structured["phone"] = ph
                logger.debug("Extracted phone: %s", ph)
        if not structured["name"]:
            if re.match(r'^[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3}$', text):
                structured["name"] = text
                logger.debug("Extracted name: %s", text)
                continue  # 姓名行通常只包含姓名

        # -------------------- 工作行 --------------------
        if is_work_line(text):
            work_entry = parse_work_line(text)
            structured["work_experience"].append(work_entry)
            last_work = work_entry
            current_section = "work_experience"
            logger.info("Detected work_experience: %s | %s", work_entry.get("position"), work_entry.get("company"))
            continue

        # -------------------- 教育行 --------------------
        if any(k in low for k in ["university", "college", "school"]) or ("|" in text and any(word in low for word in ["master", "bachelor", "phd", "degree"])):
            edu_entry = parse_education_line(text)
            structured["education"].append(edu_entry)
            current_section = "education"
            logger.info("Detected education: %s", edu_entry.get("school"))
            continue

        # -------------------- Section headers --------------------
        if any(k in low for k in ["work experience", "professional experience", "experience", "工作经历"]):
            current_section = "work_experience"
            continue
        if any(k in low for k in ["education", "education:", "学校"]):
            current_section = "education"
            continue
        if any(k in low for k in ["project", "projects", "项目", "project experience"]):
            current_section = "projects"
            continue
        if any(k in low for k in ["skill", "skills", "languages & tools", "languages and tools", "语言"]):
            current_section = "skills"
            if ":" in text:
                right = text.split(":", 1)[1]
                skills = [s.strip() for s in re.split(r",|;", right) if s.strip()]
                structured["skills"].extend(skills)
            continue

        # -------------------- 项目标题 --------------------
        if is_project_title(text):
            # 防止误把工作结束时间/Highlights当作项目
            if current_section == "work_experience" and last_work and len(text.split()) < 6:
                last_work["highlights"].append(text)
                continue
            proj = {"title": text, "highlights": []}
            structured["projects"].append(proj)
            last_project = proj
            current_section = "projects"
            logger.info("Detected project title: %s", text)
            continue

        # -------------------- 动作句 --------------------
        if ACTION_RE.search(text):
            # 忽略纯年份或 "Present" 等
            if re.match(r'^(19|20)\d{2}$', text) or text.strip().lower() == "present":
                continue
            # 当前在 work_experience
            if current_section == "work_experience" and last_work:
                last_work["highlights"].append(text)
                continue
            # 当前在 projects
            if current_section == "projects" and last_project:
                last_project["highlights"].append(text)
                continue
            # 回溯上一行
            if idx > 0:
                prev = paragraphs[idx-1].strip()
                if is_work_line(prev):
                    if not (structured["work_experience"] and structured["work_experience"][-1]["description"] == prev):
                        work_entry = parse_work_line(prev)
                        structured["work_experience"].append(work_entry)
                        last_work = work_entry
                    last_work["highlights"].append(text)
                    continue
                if is_project_title(prev):
                    # 回填到已有 project
                    for pj in reversed(structured["projects"]):
                        if pj.get("title") == prev:
                            pj["highlights"].append(text)
                            last_project = pj
                            break
                    else:
                        proj = {"title": prev, "highlights": [text]}
                        structured["projects"].append(proj)
                        last_project = proj
                    continue
            # fallback 当作独立 project
            proj = {"title": text[:60], "highlights": [text]}
            structured["projects"].append(proj)
            last_project = proj
            current_section = "projects"
            continue

        # -------------------- Skills --------------------
        if re.match(r'(?i)skills\s*[:\-]', text) or re.match(r'(?i)languages\s*&\s*tools', text) or "languages" in low:
            right = text.split(":", 1)[-1] if ":" in text else text
            skills = [s.strip() for s in re.split(r',|;|\n', right) if s.strip()]
            structured["skills"].extend(skills)
            current_section = "skills"
            continue

        # -------------------- 工作结束日期修正 --------------------
        date_match = re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*[-–]$', text)
        if date_match and last_work and idx+1 < len(paragraphs):
            last_work["end_date"] = paragraphs[idx+1].strip()
            continue

        # -------------------- Section兜底 --------------------
        if current_section == "work_experience" and last_work:
            last_work["highlights"].append(text)
            continue
        if current_section == "projects" and last_project:
            last_project["highlights"].append(text)
            continue
        if current_section == "education" and structured["education"]:
            structured["education"][-1].setdefault("description", "")
            structured["education"][-1]["description"] += (" " + text)
            continue

        # -------------------- 其他 --------------------
        if ("phone" in low or "email" in low) and (extract_email(text) or extract_phone(text)):
            continue
        structured["other"].append(text)

    # -------- 2. 再跑 NER --------
    ner_results = run_ner_batch(paragraphs)

    for para_entities in ner_results:
        for ent in para_entities:
            label = ent["entity_group"]
            text = ent["word"]

            if label == "NAME" and not structured["name"]:
                structured["name"] = text
            elif label == "EMAIL" and not structured["email"]:
                structured["email"] = text
            elif label == "PHONE" and not structured["phone"]:
                structured["phone"] = text
            elif label == "SKILL":
                structured["skills"].append(text)
            elif label == "ORG":
                # 如果当前解析的 work_experience 里没有公司，就补充
                if structured["work_experience"]:
                    last = structured["work_experience"][-1]
                    if not last.get("company"):
                        last["company"] = text
            elif label in ["EDUCATION", "SCHOOL"]:
                if not structured["education"] or not structured["education"][-1].get("school"):
                    structured["education"].append({"school": text})

    # -------------------- 3. 去重 & 默认值 --------------------
    structured["skills"] = sorted(set([s for s in structured["skills"] if s]))
    for we in structured["work_experience"]:
        we.setdefault("highlights", [])
    for pj in structured["projects"]:
        pj.setdefault("highlights", [])

    # -------------------- 清理无效条目 --------------------
    # 删除 projects 中 highlights 为空的条目
    structured["projects"] = [pj for pj in structured["projects"] if pj.get("highlights")]

    # 删除 other 中空字符串
    structured["other"] = [o for o in structured["other"] if o and o.strip()]

    return structured

# ---------------- 测试 ----------------
if __name__ == "__main__":
    test_paragraphs = [
        "Zhenyu Zhang",
        "Phone: +1 (860) 234-7101 | Email: Zhang.zhenyu6@northeastern.edu | Linkedin Profile | GithubCareer Goal",
        "Northeastern University | Master of Professional Study in Applied Machine Intelligence | 2025",
        "University of Connecticut | Bachelor of Art | 2022",
        "Data Science Intern | Google LLC | Jun 2024 – Aug 2024",
        "Created a token counter in Ollama to efficiently track and manage token usage.",
        "project",
        "YouTube Recommendation System",
        "Built a recommendation model using DNN and LightGBM...",
        "Skills: Python, SQL, TensorFlow, PyTorch"
    ]

    out = parse_resume_to_structured(test_paragraphs)
    print(json.dumps(out, indent=2, ensure_ascii=False))
