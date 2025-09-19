import sys
import os
import re
import logging
from typing import List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    CATEGORY_FIELDS,
    normalize_category,
    normalize_skills,
    auto_fill_fields,
    extract_basic_info,
    extract_skills_from_text
)
from ner import run_ner_batch
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LC_Document
from langchain_community.vectorstores import FAISS
from doc import semantic_split

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -------------------------
# 解析简历，严格区分教育/工作/项目/技能
# -------------------------
import logging
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LC_Document
from langchain_community.vectorstores import FAISS
from utils import normalize_category
from doc import semantic_split

logger = logging.getLogger(__name__)

semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# 简历结构化解析
# -------------------------
def parse_resume_to_structured(paragraphs):
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

    skill_keywords = ["python", "sql", "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "tableau", "langchain", "seaborn"]

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue

        # ---- 基础信息 ----
        if "@" in para_clean and not structured["email"]:
            structured["email"] = para_clean.split(" ")[-1].replace("Email:", "").strip()
        elif "Phone" in para_clean and not structured["phone"]:
            structured["phone"] = para_clean.split(" ")[-1].strip()
        elif not structured["name"]:
            structured["name"] = para_clean.strip()

        # ---- 技能提取 ----
        if "skills" in para_clean.lower() or "languages & tools" in para_clean.lower():
            # 统一用冒号分割
            parts = para_clean.split(":")
            if len(parts) > 1:
                skills_part = parts[1]
            else:
                skills_part = parts[0]
            # 分割逗号、空格等
            skills = [s.strip() for s in skills_part.replace("\n", ",").split(",") if s.strip()]
            # 只保留我们关心的技能
            skills = [s for s in skills if any(k in s.lower() for k in skill_keywords)]
            structured["skills"].extend(skills)
            continue  # 技能段落单独处理，不再归到其他类别

        # ---- 教育经历 ----
        if "university" in para_clean.lower() or "master" in para_clean.lower() or "bachelor" in para_clean.lower():
            structured["education"].append({
                "school": para_clean.split("|")[0].strip(),
                "degree": "|".join(para_clean.split("|")[1:-1]).strip() if len(para_clean.split("|")) > 2 else "",
                "grad_date": para_clean.split("|")[-1].strip() if len(para_clean.split("|")) > 1 else "Unknown",
                "description": para_clean
            })
            continue

        # ---- 工作经历 ----
        if any(kw in para_clean.lower() for kw in ["llc", "inc", "company", "google", "amazon", "microsoft"]):
            structured["work_experience"].append({
                "company": para_clean.split("|")[0].strip(),
                "position": "|".join(para_clean.split("|")[1:-1]).strip() if len(para_clean.split("|")) > 2 else "",
                "start_date": "Unknown",
                "end_date": "Present",
                "description": para_clean
            })
            continue

        # ---- 项目经历 ----
        if any(kw in para_clean.lower() for kw in ["project", "built", "developed"]):
            structured["projects"].append({
                "project_title": para_clean.split("|")[0].strip(),
                "project_content": para_clean,
                "start_date": "Unknown",
                "end_date": "Present"
            })
            continue

    # 去重技能
    structured["skills"] = list(set(structured["skills"]))
    return structured


# -------------------------
# FAISS 构建
# -------------------------
def build_faiss(structured_resume: dict, embeddings_model=None):
    docs = []

    for cat in ["work_experience", "projects", "education", "other"]:
        for entry in structured_resume.get(cat, []):
            meta_cat = normalize_category(cat)
            
            # projects 整体入库
            if cat == "projects":
                text = f"{entry.get('project_title','')}\n{entry.get('project_content','')}".strip()
                if not text:
                    continue
                docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                logger.info("[FAISS INSERT] cat=%s, project_block=%s", meta_cat, text[:80])
            else:
                # education / work / other 整段入库，不拆分
                text = entry.get("description", "").strip()
                if not text:
                    continue
                docs.append(LC_Document(page_content=text, metadata={"category": meta_cat}))
                logger.info("[FAISS INSERT] cat=%s, snippet=%s", meta_cat, text[:80])

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not docs:
        logger.warning("[FAISS WARN] no docs to insert into FAISS (docs list empty)")
        return None

    db = FAISS.from_documents(docs, embeddings_model)
    logger.info("FAISS database built with %d docs", len(docs))
    return db

# -------------------------
# 测试
# -------------------------
if __name__ == "__main__":
    test_paragraphs = [
        "Zhenyu Zhang | Email: Zhang.zhenyu6@northeastern.edu | Phone: +1860234-7101",
        "Northeastern University | Master of Professional Study in Applied Machine Intelligence | 2025",
        "University of Connecticut | Bachelor of Art | 2022",
        "Data Science Intern | Google LLC | Jun 2024 – Aug 2024",
        "YouTube Recommendation System Built a recommendation model using DNN and LightGBM...",
        "Skills: Python, SQL, TensorFlow, PyTorch"
    ]
    structured = parse_resume_to_structured(test_paragraphs)
    db = build_faiss(structured)
    import json
    print(json.dumps(structured, indent=2, ensure_ascii=False))
