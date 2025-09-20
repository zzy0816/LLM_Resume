import logging, json
from files import load_faiss, save_faiss, save_json, load_json
from doc import read_document_paragraphs
from semantic_test import build_faiss
from parser_test import parse_resume_to_structured
from utils import auto_fill_fields, extract_basic_info, rule_based_filter, validate_and_clean
from query import query_dynamic_category, fill_query_exact
from db import save_resume

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LC_Document
from langchain_community.vectorstores import FAISS
from utils import normalize_category

logger = logging.getLogger(__name__)
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_faiss(structured_resume: dict, embeddings_model=None):
    docs = []

    for cat in ["work_experience", "projects", "education", "other"]:
        for entry in structured_resume.get(cat, []):
            meta_cat = normalize_category(cat)

            if isinstance(entry, dict):
                # 拼接文本
                if cat == "projects":
                    text = (entry.get("project_title", "") or "") + "\n" + (entry.get("project_content", "") or "")
                    text = text.strip()
                    if not text:
                        continue
                    docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                    logger.info("[FAISS INSERT] cat=%s, project_block=%s", meta_cat, text[:80])
                else:  # work_experience / education / other
                    text_fields = []
                    for key in ["description", "role", "company", "degree", "school"]:
                        val = entry.get(key)
                        if val and isinstance(val, str):
                            text_fields.append(val.strip())
                    text = "\n".join(text_fields)
                    if not text:
                        continue
                    docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                    logger.info("[FAISS INSERT] cat=%s, snippet=%s", meta_cat, text[:80])
            elif isinstance(entry, str):
                text = entry.strip()
                if not text:
                    continue
                docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                logger.info("[FAISS INSERT] cat=%s, snippet=%s", meta_cat, text[:80])

    if not docs:
        logger.warning("[FAISS WARN] no docs to insert into FAISS (docs list empty)")
        return None

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(docs, embeddings_model)
    logger.info("FAISS database built with %d docs", len(docs))
    return db

def sanitize_filename(file_name: str) -> str:
    return file_name.replace("(", "_").replace(")", "_").replace(" ", "_")

def main_pipeline(file_names: list[str], mode: str = "exact") -> dict[str, dict]:
    results = {}

    for file_name in file_names:
        file_path = f"./downloads/{file_name}"
        logger.info(f"DEBUG: processing file {file_name}")

        safe_name = sanitize_filename(file_name)

        structured_resume = None
        try:
            structured_resume = load_json(safe_name)
        except:
            pass

        if structured_resume is None:
            paragraphs = read_document_paragraphs(file_path)
            full_text = "\n".join(paragraphs)
            basic_info = extract_basic_info(full_text)

            # 初始化 JSON
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

            # parser 提取
            parsed_resume = parse_resume_to_structured(paragraphs) or {}
            logger.info(f"DEBUG: parsed_resume: {json.dumps(parsed_resume, ensure_ascii=False, indent=2)}")
            parsed_resume = auto_fill_fields(parsed_resume)

            # 合并 parser 到 structured_resume
            for key, val in parsed_resume.items():
                if key not in ["name", "email", "phone"]:
                    structured_resume[key] = val

            structured_resume["name"] = structured_resume["name"] or basic_info.get("name")
            structured_resume["email"] = structured_resume["email"] or basic_info.get("email")
            structured_resume["phone"] = structured_resume["phone"] or basic_info.get("phone")

        user_email = structured_resume.get("email") or safe_name

        # 构建 / 加载 FAISS
        db = load_faiss(safe_name)
        if db is None:
            logger.info(f"DEBUG: FAISS not found, building for {safe_name}")
            db = build_faiss(structured_resume)
            if db:
                save_faiss(safe_name, db)

        # FAISS 查询并填充
        queries = ["工作经历", "项目经历", "教育经历", "技能"]
        query_results = {}
        if db:
            for q in queries:
                res = query_dynamic_category(db, structured_resume, q, top_k=10)
                raw_results = res.get("results", [])
                filtered = rule_based_filter(q, raw_results)
                query_results[q] = filtered

        if mode == "exact":
            structured_resume = fill_query_exact(structured_resume, query_results)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        structured_resume = validate_and_clean(structured_resume)

        # 保存 JSON 和 Mongo
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
