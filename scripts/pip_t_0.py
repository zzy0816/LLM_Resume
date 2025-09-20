import logging, json
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

def sanitize_filename(file_name: str) -> str:
    return file_name.replace("(", "_").replace(")", "_").replace(" ", "_")

def main_pipeline(file_names: list[str], mode: str = "exact") -> dict[str, dict]:
    results = {}

    for file_name in file_names:
        file_path = f"./downloads/{file_name}"
        logger.info(f"DEBUG: processing file {file_name}")
        safe_name = sanitize_filename(file_name)

        structured_resume = load_json(safe_name) if load_json(safe_name) else None

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
            logger.info(f"DEBUG: parsed_resume: {json.dumps(parsed_resume, ensure_ascii=False, indent=2)}")

            parsed_file = f"./data/parsed_resume_{safe_name}.json"
            with open(parsed_file, "w", encoding="utf-8") as f:
                json.dump(parsed_resume, f, ensure_ascii=False, indent=2)
            logger.info(f"DEBUG: saved parsed_resume to {parsed_file}")

            parsed_resume = auto_fill_fields(parsed_resume)
            for key, val in parsed_resume.items():
                if key not in ["name", "email", "phone"]:
                    structured_resume[key] = val

            structured_resume["name"] = structured_resume["name"] or basic_info.get("name")
            structured_resume["email"] = structured_resume["email"] or basic_info.get("email")
            structured_resume["phone"] = structured_resume["phone"] or basic_info.get("phone")

        user_email = structured_resume.get("email") or safe_name

        db = load_faiss(safe_name)
        if db is None:
            logger.info(f"DEBUG: FAISS not found, building for {safe_name}")
            db = build_faiss(structured_resume)
            if db:
                save_faiss(safe_name, db)

        # FAISS 查询
        queries = ["工作经历", "项目经历", "教育经历", "技能"]
        query_results = {}
        if db:
            for q in queries:
                res = query_dynamic_category(db, structured_resume, q, top_k=10)
                raw_results = res.get("results", [])
                filtered = rule_based_filter(q, raw_results)
                query_results[q] = filtered
                logger.info(f"[QUERY DEBUG] query='{q}' -> {len(filtered)} results: {filtered[:3]}")

        # 处理 work, projects, education
        mapping = [("工作经历", "work_experience"),
                   ("项目经历", "projects"),
                   ("教育经历", "education")]
        for cat, key in mapping:
            faiss_list = query_results.get(cat)
            if faiss_list:
                processed = []
                for e in faiss_list:
                    if isinstance(e, str):
                        processed.append({"description": e})
                    elif isinstance(e, dict):
                        processed.append(e)
                structured_resume[key] = processed
            else:
                structured_resume[key] = structured_resume.get(key, [])

        # 处理 skills
        skills_list = query_results.get("技能")
        if skills_list:
            skills = []
            for s in skills_list:
                if isinstance(s, str):
                    skills.extend([line.strip() for line in s.splitlines() if line.strip()])
            structured_resume["skills"] = skills
        else:
            structured_resume["skills"] = structured_resume.get("skills", [])

        # exact 模式处理
        if mode == "exact":
            structured_resume = fill_query_exact(structured_resume, query_results)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # other 字段清理
        structured_resume["other"] = [
            {"description": str(entry.get("description", ""))} if isinstance(entry, dict) else {"description": str(entry)}
            for entry in structured_resume.get("other", [])
        ]

        structured_resume = validate_and_clean(structured_resume)

        save_resume(user_id=user_email, file_name=safe_name + "_faiss_confirmed", data=structured_resume)
        save_json(safe_name + "_faiss_confirmed", structured_resume)
        logger.info(f"DEBUG: saved resume for {user_email}")

        results[user_email] = structured_resume

    return results

if __name__ == "__main__":
    files_to_process = ["Resume(AI).docx"]
    all_results = main_pipeline(files_to_process, mode="exact")
    for user_email, structured_resume in all_results.items():
        logger.info(f"\n===== FINAL STRUCTURED RESUME JSON for {user_email} =====")
        logger.info(json.dumps(structured_resume, ensure_ascii=False, indent=2))
