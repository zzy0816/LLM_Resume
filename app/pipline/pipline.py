import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging, json, re
from app.utils.files import load_faiss, save_faiss, save_json, load_json
from app.qre.doc_read import read_document_paragraphs
from app.qre.parser import parse_resume_to_structured
from app.utils.utils import auto_fill_fields, extract_basic_info, rule_based_filter, validate_and_clean, fix_resume_dates
from app.qre.query import query_dynamic_category, fill_query_exact
from app.storage.db import save_resume
from app.qre.semantic import build_faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------
# 文件名 sanitize
# ------------------------
def sanitize_filename(file_name: str) -> str:
    return file_name.replace("(", "_").replace(")", "_").replace(" ", "_")

# ------------------------
# 恢复 fill_query_exact 结果的结构
# ------------------------
def restore_parsed_structure(structured_resume, original_resume):
    """将 fill_query_exact 结果恢复为原解析结构"""
    # work_experience
    if "work_experience" in structured_resume:
        restored = []
        for i, item in enumerate(structured_resume["work_experience"]):
            if isinstance(item, dict):
                orig_item = original_resume.get("work_experience", [])[i] if i < len(original_resume.get("work_experience", [])) else {}
                restored.append({
                    "company": item.get("company") or orig_item.get("company"),
                    "position": item.get("title") or orig_item.get("position"),
                    "location": item.get("location") or orig_item.get("location"),
                    "start_date": item.get("start_date") or orig_item.get("start_date"),
                    "end_date": item.get("end_date") or orig_item.get("end_date"),
                    "description": item.get("description") or orig_item.get("description"),
                    "highlights": item.get("highlights") or orig_item.get("highlights", [])
                })
        structured_resume["work_experience"] = restored

    # projects
    if "projects" in structured_resume:
        restored = []
        existing_titles = set()
        for orig_item in original_resume.get("projects", []):
            restored.append(orig_item)
            title = orig_item.get("title") or orig_item.get("project_title")
            if title:
                existing_titles.add(title)

        for item in structured_resume.get("projects", []):
            if isinstance(item, dict):
                title = item.get("project_title") or item.get("title")
                if title and title not in existing_titles:
                    restored.append({
                        "title": title,
                        "highlights": item.get("highlights", []),
                        "description": item.get("project_content") or item.get("description", "")
                    })
                    existing_titles.add(title)

        structured_resume["projects"] = restored

    # education
    if "education" in structured_resume:
        restored = []
        for i, item in enumerate(structured_resume["education"]):
            if isinstance(item, dict):
                orig_item = original_resume.get("education", [])[i] if i < len(original_resume.get("education", [])) else {}
                restored.append({
                    "school": item.get("school") or orig_item.get("school"),
                    "degree": item.get("degree") or orig_item.get("degree"),
                    "grad_date": item.get("grad_date") or orig_item.get("grad_date"),
                    "description": item.get("description") or orig_item.get("description")
                })
        structured_resume["education"] = restored

    return structured_resume

# ------------------------
# skills 处理
# ------------------------
def clean_skills(raw_skills: list[str]) -> list[str]:
    """统一技能格式，拆分多技能条目"""
    cleaned = []
    for s in raw_skills:
        if not s:
            continue
        s = s.strip()
        s = re.sub(r"^(Frameworks\s*&\s*Libraries:)", "", s, flags=re.I).strip()
        parts = re.split(r"[,\n]", s)
        parts = [p.strip() for p in parts if p.strip()]
        cleaned.extend(parts)
    return cleaned

# ------------------------
# 回填工作经历时间
# ------------------------
def restore_work_experience(structured_resume, parsed_resume, faiss_results):
    if "work_experience" not in structured_resume:
        return structured_resume

    restored = []
    for i, item in enumerate(structured_resume["work_experience"]):
        if i < len(faiss_results.get("工作经历", [])):
            faiss_entry = getattr(faiss_results["工作经历"][i], "metadata", {})
        else:
            faiss_entry = {}

        orig_item = parsed_resume.get("work_experience", [])[i] if i < len(parsed_resume.get("work_experience", [])) else {}

        restored.append({
            "company": item.get("company") or orig_item.get("company"),
            "position": item.get("title") or orig_item.get("position"),
            "location": item.get("location") or orig_item.get("location"),
            "start_date": faiss_entry.get("start_date") or orig_item.get("start_date") or "Unknown",
            "end_date": faiss_entry.get("end_date") or orig_item.get("end_date") or "Present",
            "description": item.get("description") or orig_item.get("description"),
            "highlights": item.get("highlights") or orig_item.get("highlights", [])
        })

    structured_resume["work_experience"] = restored
    return structured_resume

# ------------------------
# 主 pipeline
# ------------------------
def main_pipeline(file_names: list[str], mode: str = "exact") -> dict[str, dict]:
    results = {}

    for file_name in file_names:
        file_path = f"./downloads/{file_name}"
        logger.info(f"[PIPELINE] Processing file {file_name}")
        safe_name = sanitize_filename(file_name)

        structured_resume = load_json(safe_name)
        parsed_resume = structured_resume.copy() if structured_resume else {}

        if structured_resume is None:
            paragraphs = read_document_paragraphs(file_path)
            full_text = "\n".join(paragraphs)
            basic_info = extract_basic_info(full_text)

            structured_resume = {
                "name": basic_info.get("name"),
                "email": basic_info.get("email"),
                "phone": basic_info.get("phone"),
                "basic_info": basic_info,
                "education": [],
                "work_experience": [],
                "projects": [],
                "skills": [],
                "other": []
            }

            parsed_resume = parse_resume_to_structured(paragraphs) or {}
            parsed_resume = auto_fill_fields(parsed_resume)

            for key, val in parsed_resume.items():
                if key not in ["name", "email", "phone"]:
                    structured_resume[key] = val

        user_email = structured_resume.get("email") or safe_name

        db = load_faiss(safe_name)
        if db is None:
            logger.info(f"[PIPELINE] FAISS not found, building for {safe_name}")
            db = build_faiss(structured_resume)
            if db:
                save_faiss(safe_name, db)

        queries = ["工作经历", "项目经历", "教育经历", "技能"]
        query_results = {}
        if db:
            for q in queries:
                res = query_dynamic_category(db, structured_resume, q, top_k=10)
                filtered = rule_based_filter(q, res.get("results", []))
                query_results[q] = filtered

        structured_resume = fill_query_exact(structured_resume, query_results, parsed_resume)
        structured_resume = restore_parsed_structure(structured_resume, parsed_resume)
        structured_resume = restore_work_experience(structured_resume, parsed_resume, query_results)
        structured_resume = validate_and_clean(structured_resume)
        structured_resume = fix_resume_dates(structured_resume)

        structured_resume["skills"] = clean_skills(query_results.get("技能", []) or structured_resume.get("skills", []))

        structured_resume["other"] = [
            {"description": str(entry.get("description", ""))} if isinstance(entry, dict) else {"description": str(entry)}
            for entry in structured_resume.get("other", [])
        ]

        save_resume(user_id=user_email, file_name=safe_name + "_faiss_confirmed", data=structured_resume)
        save_json(safe_name + "_faiss_confirmed", structured_resume)
        logger.info(f"[PIPELINE] Saved resume for {user_email}")

        results[user_email] = structured_resume

    return results

# ------------------------
# 主函数
# ------------------------
if __name__ == "__main__":
    files_to_process = ["Resume(AI).docx"]
    all_results = main_pipeline(files_to_process, mode="exact")
    for user_email, structured_resume in all_results.items():
        logger.info(f"\n===== FINAL STRUCTURED RESUME JSON for {user_email} =====")
        logger.info(json.dumps(structured_resume, ensure_ascii=False, indent=2))
