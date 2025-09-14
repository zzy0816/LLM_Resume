import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import re
import pdfplumber
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
def read_document_paragraphs(file_path: str):
    """
    统一读取 DOCX / PDF 文档为段落列表
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return read_docx_paragraphs(file_path)
    elif ext == ".pdf":
        return read_pdf_paragraphs(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def read_docx_paragraphs(docx_path: str):
    doc = DocxDocument(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    logger.info("DOCX split into %d paragraphs", len(paragraphs))
    return paragraphs

def read_pdf_paragraphs(pdf_path: str):
    paragraphs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # 按换行符分段
                page_paras = [p.strip() for p in text.split("\n") if p.strip()]
                paragraphs.extend(page_paras)
    logger.info("PDF split into %d paragraphs", len(paragraphs))
    return paragraphs

# -------------------------
# FAISS 分段（增强教育段落保留短文本）
# -------------------------
ACTION_VERBS = ("built", "created", "used", "collected", "led", "fine-tuned", "developed", "designed", "implemented")

def semantic_split(text: str, max_size=MAX_CHUNK_SIZE):
    """
    改造版 semantic_split，专门处理项目标题 + 描述格式：
    - 项目标题：不以动作动词开头，长度<100，或者包含 'project'
    - 描述行：以 '-' 开头或动词开头
    - 遇到新标题就拆分
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks = []
    curr_chunk = []

    for line in lines:
        line_lower = line.lower()
        # 判断是否为新项目标题
        is_title = (len(line) <= 100 and not line_lower.startswith(ACTION_VERBS)) or "project" in line_lower

        if is_title and curr_chunk:
            chunks.append("\n".join(curr_chunk))
            curr_chunk = [line]
        else:
            curr_chunk.append(line)

    if curr_chunk:
        chunks.append("\n".join(curr_chunk))

    # 对大段落再按 max_size 拆
    final_chunks = []
    for c in chunks:
        if len(c) <= max_size:
            final_chunks.append(c)
        else:
            # 大段落继续按句拆
            sentences = re.split(r"[。,.]", c)
            current = ""
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(current) + len(s) + 1 <= max_size:
                    current += s + "。"
                else:
                    if current.strip():
                        final_chunks.append(current.strip())
                    current = s + "。"
            if current.strip():
                final_chunks.append(current.strip())

    return final_chunks

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
