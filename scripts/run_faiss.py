import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import json
from files import load_json, save_json, load_faiss, save_faiss
from query_test import query_dynamic_category, fill_query_exact
from utils import rule_based_filter, validate_and_clean
from pipline_test import restore_parsed_structure
from db import save_resume
from langchain.schema import Document as LC_Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------
# 测试 FAISS 构建和查询
# ------------------------
def build_test_faiss(structured_resume):
    from sentence_transformers import SentenceTransformer
    docs = []
    for cat in ["work_experience", "projects", "education", "skills", "other"]:
        entries = structured_resume.get(cat, [])
        if not entries:
            continue
        if cat == "skills":
            text = "\n".join([str(s).strip() for s in entries if s])
            if text:
                docs.append(LC_Document(page_content=text, metadata={"category": cat}))
            continue
        for e in entries:
            if isinstance(e, dict):
                text = ""
                if cat == "projects":
                    title_text = e.get("project_title") or e.get("title") or ""
                    highlights_text = "\n".join(e.get("highlights", []))
                    text = "\n".join([title_text, highlights_text]).strip()
                else:
                    text = e.get("description", "") or ""
                if text:
                    docs.append(LC_Document(page_content=text, metadata={"category": cat}))
    if not docs:
        logger.warning("No docs to build FAISS.")
        return None
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings_model)
    logger.info(f"FAISS DB built with {len(docs)} documents")
    return db

# ------------------------
# 主测试函数
# ------------------------
def test_faiss(parsed_json_file: str):
    # 1️⃣ 读取已解析的 JSON
    try:
        with open(parsed_json_file, "r", encoding="utf-8") as f:
            structured_resume = json.load(f)
        logger.info(f"Successfully loaded JSON: {parsed_json_file}")
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return

    parsed_resume = structured_resume.copy()  # 保存原始解析结构用于回填

    # 2️⃣ 构建 FAISS
    db = build_test_faiss(structured_resume)
    if not db:
        logger.error("FAISS build failed")
        return

    # 3️⃣ 查询测试
    queries = ["工作经历", "项目经历", "教育经历", "技能"]
    query_results = {}
    for q in queries:
        res = query_dynamic_category(db, structured_resume, q, top_k=10)
        raw_results = res.get("results", [])
        filtered = rule_based_filter(q, raw_results)
        query_results[q] = filtered
        logger.info(f"Query '{q}' -> {len(filtered)} results")
        for i, item in enumerate(filtered[:3]):
            logger.info(f"  {i+1}: {item[:120]}")

    # 4️⃣ 使用 fill_query_exact 回填
    structured_resume = fill_query_exact(structured_resume, query_results, parsed_resume)
    structured_resume = restore_parsed_structure(structured_resume, parsed_resume)
    structured_resume = validate_and_clean(structured_resume)

    # 5️⃣ 输出结果
    logger.info("===== FAISS CONFIRMED STRUCTURED RESUME =====")
    logger.info(json.dumps(structured_resume, ensure_ascii=False, indent=2))

    # 6️⃣ 保存结果（直接使用标准 open，不调用 save_json）
    output_file = parsed_json_file.replace(".json", "_faiss_test.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_resume, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved FAISS test JSON: {output_file}")

# ------------------------
# 脚本入口
# ------------------------
if __name__ == "__main__":
    import os
    print("CWD:", os.getcwd())
    print("Exists:", os.path.exists(r"data\classified\Resume_AI_.pdf_parsed.json"))

    test_file = r"data\classified\Resume_AI_.pdf_parsed.json"  # 使用已保存的 parsed.json 文件
    test_faiss(test_file)
