import os
import re
import json
import logging
import faiss
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------- Data Models -----------------
@dataclass
class ResumeEntry:
    title: Optional[str] = None
    organization: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

@dataclass
class StructuredResume:
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = None
    work_experience: List[ResumeEntry] = None
    education: List[ResumeEntry] = None
    projects: List[ResumeEntry] = None
    other: List[str] = None

# ----------------- Utils -----------------
def save_json(file_name: str, data: Dict[str, Any]):
    path = os.path.join("./data/classified", f"{file_name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved structured resume to %s", path)

def load_json(file_name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join("./data/classified", f"{file_name}.json")
    if os.path.exists(path):
        logger.info("⚠️ 使用缓存 JSON: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def read_docx_paragraphs(file_path: str) -> List[str]:
    import docx
    doc = docx.Document(file_path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    logger.info("Loaded %d paragraphs from %s", len(paras), file_path)
    return paras

# ----------------- Regex Extractors -----------------
def extract_email(text: str) -> Optional[str]:
    m = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return m.group(0) if m else None

def extract_phone(text: str) -> Optional[str]:
    m = re.search(r"\+?\d[\d\-\s]{7,}\d", text)
    return m.group(0) if m else None

# ----------------- Core Parsing -----------------
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def parse_resume_to_structured(paragraphs: List[str]) -> Dict[str, Any]:
    structured = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "work_experience": [],
        "education": [],
        "projects": [],
        "other": []
    }

    # 基础字段 (email, phone)
    all_text = " ".join(paragraphs)
    structured["email"] = extract_email(all_text)
    structured["phone"] = extract_phone(all_text)

    logger.debug("[PARSE DEBUG] 提取到 email=%s, phone=%s", structured["email"], structured["phone"])

    # NER + 分类
    for idx, para in enumerate(paragraphs):
        if not para.strip():
            continue

        logger.debug("[PARSE DEBUG] 段落[%d]: %s", idx, para[:80])
        ner_results = ner_pipeline(para)
        logger.debug("[NER RAW] idx=%d -> %s", idx, ner_results)

        # 简单分类逻辑
        category = None
        if any(ent["entity_group"] == "ORG" for ent in ner_results):
            category = "work_experience"
        elif "project" in para.lower():
            category = "projects"
        elif "university" in para.lower() or "bachelor" in para.lower() or "master" in para.lower():
            category = "education"
        elif "python" in para.lower() or "sql" in para.lower() or "tensorflow" in para.lower():
            category = "skills"

        if category:
            entry = ResumeEntry(description=para)
            structured[category].append(asdict(entry))
            logger.debug("[CATEGORY FINAL] idx=%d -> %s", idx, category)
        else:
            structured["other"].append(para)
            logger.debug("[CATEGORY FINAL] idx=%d -> other", idx)

    return structured

# ----------------- FAISS -----------------
def build_faiss(structured_resume: Dict[str, Any]):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    entries = []
    labels = []

    for cat in ["work_experience", "projects", "education", "skills"]:
        for item in structured_resume.get(cat, []):
            if isinstance(item, dict):
                text = item.get("description") or item.get("title") or ""
            else:
                text = str(item)
            if text:
                entries.append(text)
                labels.append(cat)

    if not entries:
        logger.warning("⚠️ 没有找到任何内容用于构建 FAISS")
        return None

    embeddings = model.encode(entries, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    logger.info("Built FAISS index with %d vectors", len(entries))
    return {"index": index, "entries": entries, "labels": labels}

# ----------------- Main -----------------
if __name__ == "__main__":
    file_name = "Resume(AI).docx"
    file_path = os.path.join("./downloads", file_name)

    # 强制重新解析时可直接置为 None
    structured_resume = load_json(file_name.replace(".docx", ""))
    if structured_resume is None:
        paragraphs = read_docx_paragraphs(file_path)
        structured_resume = parse_resume_to_structured(paragraphs)
        save_json(file_name.replace(".docx", ""), structured_resume)
    else:
        logger.info("直接使用缓存 JSON, 跳过解析")

    # 调试打印分类统计
    logger.info("分类结果统计: work=%d, edu=%d, projects=%d, skills=%d, other=%d",
        len(structured_resume.get("work_experience", [])),
        len(structured_resume.get("education", [])),
        len(structured_resume.get("projects", [])),
        len(structured_resume.get("skills", [])),
        len(structured_resume.get("other", []))
    )

    # 构建 FAISS
    faiss_db = build_faiss(structured_resume)

    logger.info("Processing completed successfully.")
