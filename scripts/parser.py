import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import json
import logging
from typing import List

from utils import preprocess_paragraphs, extract_email, extract_phone, is_work_line, parse_education_line, parse_work_line, is_project_title

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

    current_section = None
    last_work = None
    last_project = None

    for idx, raw in enumerate(paragraphs):
        text = raw.strip()
        if not text:
            continue
        low = text.lower()

        # 基本信息抽取（不跳过后续逻辑，除非行确实只是姓名）
        if not structured["email"]:
            em = extract_email(text)
            if em:
                structured["email"] = em
                logger.debug("Extracted email: %s (from '%s')", em, text)
        if not structured["phone"]:
            ph = extract_phone(text)
            if ph:
                structured["phone"] = ph
                logger.debug("Extracted phone: %s (from '%s')", ph, text)
        if not structured["name"]:
            if re.match(r'^[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3}$', text):
                structured["name"] = text
                logger.debug("Extracted name: %s", text)
                # 通常姓名行只是姓名，跳到下一行
                continue

        # 优先：明确的工作行（含 '|' 且匹配职位或公司关键词）
        if is_work_line(text):
            work_entry = parse_work_line(text)
            structured["work_experience"].append(work_entry)
            last_work = work_entry
            current_section = "work_experience"
            logger.info("Detected work_experience: %s | %s", work_entry.get("position"), work_entry.get("company"))
            continue

        # 教育行（含 university/college/school 或含 '|' 且看起来是学校）
        if any(k in low for k in ["university", "college", "school"]) or ("|" in text and any(word in text.lower() for word in ["master", "bachelor", "phd", "degree"])):
            edu_entry = parse_education_line(text)
            structured["education"].append(edu_entry)
            current_section = "education"
            logger.info("Detected education: %s", edu_entry.get("school"))
            continue

        # 明确的 section header 切换
        if any(k in low for k in ["work experience", "professional experience", "experience", "工作经历"]):
            current_section = "work_experience"
            logger.debug("Switch to section work_experience @ line %d", idx)
            continue
        if any(k in low for k in ["education", "education:", "学校"]):
            current_section = "education"
            logger.debug("Switch to section education @ line %d", idx)
            continue
        if any(k in low for k in ["project", "projects", "项目", "project experience"]):
            current_section = "projects"
            logger.debug("Switch to section projects @ line %d", idx)
            if low.strip() in ("project", "projects", "项目", "project experience", "project:"):
                continue
        if any(k in low for k in ["skill", "skills", "languages & tools", "languages and tools", "语言"]):
            current_section = "skills"
            if ":" in text:
                right = text.split(":", 1)[1]
                skills = [s.strip() for s in re.split(r",|;", right) if s.strip()]
                structured["skills"].extend(skills)
                logger.info("Detected skills inline: %s", skills)
            continue

        # 项目标题识别
        if is_project_title(text):
            proj = {"title": text, "highlights": []}
            structured["projects"].append(proj)
            last_project = proj
            current_section = "projects"
            logger.info("Detected project title: %s", text)
            continue

        # 动作句（Built/Created/...）
        if ACTION_RE.search(text):
            # 情形一：当前在 work_experience 且有 last_work（优先）
            if current_section == "work_experience" and last_work:
                last_work["highlights"].append(text)
                logger.info("Added work highlight to %s: %s", last_work.get("company"), text)
                continue
            # 情形二：当前在 projects 且 last_project 存在
            if current_section == "projects" and last_project:
                last_project["highlights"].append(text)
                logger.info("Added project highlight to %s: %s", last_project.get("title"), text)
                continue
            # 情形三：回溯上一行是否为 work_line（如果上一步被误判或漏判）
            if idx > 0:
                prev = paragraphs[idx-1].strip()
                if is_work_line(prev):
                    # 如果还没被解析成 work_experience（安全回填）
                    if not (structured["work_experience"] and structured["work_experience"][-1]["description"] == prev):
                        work_entry = parse_work_line(prev)
                        structured["work_experience"].append(work_entry)
                        last_work = work_entry
                        logger.info("Backfilled work_experience from previous line: %s", prev)
                    last_work["highlights"].append(text)
                    logger.info("Backlinked work highlight to previous work: %s", text)
                    continue
                # 如果上一行是项目标题，回填到该 project
                if is_project_title(prev):
                    # find the project with that title
                    for pj in reversed(structured["projects"]):
                        if pj.get("title") == prev:
                            pj["highlights"].append(text)
                            last_project = pj
                            logger.info("Backlinked project highlight to previous title '%s': %s", prev, text)
                            break
                    else:
                        # 找不到则新建 project
                        proj = {"title": prev, "highlights": [text]}
                        structured["projects"].append(proj)
                        last_project = proj
                        logger.info("Created project from previous title with highlight: %s", prev)
                    continue
            # 兜底：将该动作句当作独立 project（fallback）
            proj = {"title": text[:60], "highlights": [text]}
            structured["projects"].append(proj)
            last_project = proj
            current_section = "projects"
            logger.info("Detected standalone project (fallback): %s", text[:80])
            continue

        # Skills 行（非 header 情况）
        if re.match(r'(?i)skills\s*[:\-]', text) or re.match(r'(?i)languages\s*&\s*tools', text) or "languages" in low:
            right = text.split(":", 1)[-1] if ":" in text else text
            skills = [s.strip() for s in re.split(r',|;|\n', right) if s.strip()]
            structured["skills"].extend(skills)
            current_section = "skills"
            logger.info("Detected skills: %s", skills)
            continue

        # 兜底：根据当前 section 放置
        if current_section == "work_experience" and last_work:
            last_work["highlights"].append(text)
            logger.debug("Appended to last work highlights: %s", text)
            continue
        if current_section == "projects" and last_project:
            last_project["highlights"].append(text)
            logger.debug("Appended to last project highlights: %s", text)
            continue
        if current_section == "education" and structured["education"]:
            structured["education"][-1].setdefault("description", "")
            structured["education"][-1]["description"] += (" " + text)
            logger.debug("Appended to last education description: %s", text)
            continue

        # 否则放到 other（电话号码/邮箱行也不用重复放）
        if ("phone" in low or "email" in low) and (extract_email(text) or extract_phone(text)):
            logger.debug("Skipped adding contact line to 'other' (already extracted): %s", text)
            continue

        structured["other"].append(text)
        logger.debug("Appended to other: %s", text)

    # 清理与去重
    structured["skills"] = sorted(set([s for s in structured["skills"] if s]))
    for we in structured["work_experience"]:
        we.setdefault("highlights", [])
    for pj in structured["projects"]:
        pj.setdefault("highlights", [])

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
