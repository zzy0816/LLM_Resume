import logging, json, re
from files import load_faiss, save_faiss, save_json, load_json
from doc import read_document_paragraphs
from parser_test import parse_resume_to_structured
from utils import auto_fill_fields, extract_basic_info, rule_based_filter, validate_and_clean
from query_test import query_dynamic_category, fill_query_exact
from db import save_resume

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LC_Document
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------
# FAISS 构建
# ------------------------
def build_faiss(structured_resume: dict, embeddings_model=None):
    docs = []
    user_email = structured_resume.get("email", "unknown")
    logger.info(f"[FAISS DEBUG] Starting build_faiss for resume: {user_email}")

    categories = ["work_experience", "projects", "education", "skills", "other"]
    cat_map = {cat: cat for cat in categories}

    for cat in categories:
        entries = structured_resume.get(cat, [])
        logger.info(f"[FAISS DEBUG] Processing category '{cat}' with {len(entries)} entries")

        if not entries:
            continue

        if cat == "skills" and isinstance(entries, list):
            text = "\n".join([str(s).strip() for s in entries if s])
            if text:
                docs.append(LC_Document(page_content=text, metadata={"category": cat_map[cat]}))
                logger.info(f"[FAISS INSERT] cat={cat_map[cat]}, snippet={text[:80]}")
            continue

        for i, entry in enumerate(entries):
            meta_cat = cat_map[cat]
            text = ""

            if cat == "projects" and isinstance(entry, dict):
                title_text = entry.get("project_title") or entry.get("title") or ""
                title_text = title_text.strip()
                highlights = entry.get("highlights", [])
                highlights_text = "\n".join([h.strip() for h in highlights if h.strip()])

                if title_text and highlights_text:
                    text = title_text + "\n" + highlights_text
                elif title_text:
                    text = title_text
                elif highlights_text:
                    text = highlights_text
                else:
                    text = None

            elif isinstance(entry, dict):
                text_fields = []
                for key in ["description", "role", "company", "degree", "school"]:
                    val = entry.get(key)
                    if val and isinstance(val, str):
                        val = val.strip()
                        if val:
                            text_fields.append(val)
                text = "\n".join(text_fields).strip() or None

            elif isinstance(entry, str):
                text = entry.strip() or None

            if not text:
                text = f"[{meta_cat} 未提供内容]"
                logger.warning(f"[FAISS WARN] Entry {i} in category '{cat}' is empty, using placeholder.")

            docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
            logger.info(f"[FAISS INSERT] cat={meta_cat}, snippet={text[:80]}")

    if not docs:
        logger.warning("[FAISS WARN] No docs generated, FAISS DB will be empty")
        return None

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings_model)
    logger.info(f"[FAISS INFO] FAISS database built with {len(docs)} docs")
    return db

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
    # projects
    if "projects" in structured_resume:
        restored = []
        existing_titles = set()
        # 先保留原解析的项目
        for orig_item in original_resume.get("projects", []):
            restored.append(orig_item)
            title = orig_item.get("title") or orig_item.get("project_title")
            if title:
                existing_titles.add(title)

        # 再加入 fill_query_exact 新增的项目（避免重复）
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

def clean_skills(raw_skills: list[str]) -> list[str]:
    """统一技能格式，拆分多技能条目"""
    cleaned = []
    for s in raw_skills:
        if not s:
            continue
        # 去掉可能的前缀，如 "Frameworks & Libraries:"
        s = s.strip()
        s = re.sub(r"^(Frameworks\s*&\s*Libraries:)", "", s, flags=re.I).strip()
        # 按逗号或换行拆分
        parts = re.split(r"[,\n]", s)
        parts = [p.strip() for p in parts if p.strip()]
        cleaned.extend(parts)
    return cleaned


# ------------------------
# 主 pipeline
# ------------------------
def main_pipeline(file_names: list[str], mode: str = "exact") -> dict[str, dict]:
    results = {}

    for file_name in file_names:
        file_path = f"./downloads/{file_name}"
        logger.info(f"[PIPELINE] Processing file {file_name}")
        safe_name = sanitize_filename(file_name)

        # 尝试直接加载已解析 JSON
        structured_resume = load_json(safe_name)
        parsed_resume = structured_resume.copy() if structured_resume else {}

        if structured_resume is None:
            # 原始解析
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

        # FAISS 构建和加载
        db = load_faiss(safe_name)
        if db is None:
            logger.info(f"[PIPELINE] FAISS not found, building for {safe_name}")
            db = build_faiss(structured_resume)
            if db:
                save_faiss(safe_name, db)

        # FAISS 查询 + 回填
        queries = ["工作经历", "项目经历", "教育经历", "技能"]
        query_results = {}
        if db:
            for q in queries:
                res = query_dynamic_category(db, structured_resume, q, top_k=10)
                raw_results = res.get("results", [])
                filtered = rule_based_filter(q, raw_results)
                query_results[q] = filtered

        # 使用 test_faiss 的回填逻辑
        structured_resume = fill_query_exact(structured_resume, query_results, parsed_resume)
        structured_resume = restore_parsed_structure(structured_resume, parsed_resume)
        structured_resume = validate_and_clean(structured_resume)

        # skills 处理
        skills_list = query_results.get("技能", []) or structured_resume.get("skills", [])
        structured_resume["skills"] = clean_skills(skills_list)

        # other 字段处理
        structured_resume["other"] = [
            {"description": str(entry.get("description", ""))} if isinstance(entry, dict) else {"description": str(entry)}
            for entry in structured_resume.get("other", [])
        ]

        # 保存
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
