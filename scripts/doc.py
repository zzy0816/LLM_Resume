import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import re
from dotenv import load_dotenv
from docx import Document as DocxDocument
from difflib import SequenceMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 400

# -------------------------
# 文档读取
# -------------------------
def read_docx_paragraphs(docx_path: str):
    doc = DocxDocument(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    logger.info("Document split into %d paragraphs", len(paragraphs))
    return paragraphs

# -------------------------
# FAISS 分段（增强教育段落保留短文本）
# -------------------------
def semantic_split(text: str, max_size=MAX_CHUNK_SIZE):
    """
    将文本按句拆分为子块，用于 FAISS 插入
    - 对短段落（< max_size）保留
    - 支持中文句号和英文逗号分句
    """
    sentences = re.split(r"[。,.]", text.replace("\n"," "))
    sentences = [s.strip() for s in sentences if s.strip()]
    sub_chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_size:
            current += s + "。"
        else:
            if current.strip(): 
                sub_chunks.append(current.strip())
            current = s + "。"
    if current.strip(): 
        sub_chunks.append(current.strip())
    # 保留短段落，避免教育信息丢失
    sub_chunks = [sc for sc in sub_chunks if len(sc) > 5]
    return sub_chunks

# -------------------------
# 文本归一化与相似度判断（用来去重）
# -------------------------
def normalize_text_for_compare(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    # 移除常见中英文标点并折叠空白
    for ch in "。。，,、：:；;.-–—()[]{}\"'`··\u3000":
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def similar_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def is_duplicate_para(new_para: str, existing_list: list, threshold: float = 0.85) -> bool:
    norm_new = normalize_text_for_compare(new_para)
    if not norm_new:
        return True
    for e in existing_list:
        # e 可能是 dict 或 str
        if isinstance(e, dict):
            text = e.get("description") or " ".join([str(v) for v in e.values()])
        else:
            text = str(e)
        norm_e = normalize_text_for_compare(text)
        if not norm_e:
            continue
        if norm_new in norm_e or norm_e in norm_new:
            return True
        if similar_ratio(norm_new, norm_e) >= threshold:
            return True
    return False
