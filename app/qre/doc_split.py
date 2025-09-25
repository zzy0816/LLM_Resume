import logging
import os
import sys
import re
import textwrap
from difflib import SequenceMatcher

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from app.utils.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 400


# -------------------------
# 文本渲染（供 Donut 使用）
# -------------------------
def render_paragraphs_to_image(
    paragraphs: list[str], img_width: int = 1200, font_path: str | None = None
):
    if Image is None or ImageDraw is None or ImageFont is None:
        return None

    max_chars_per_line = 80
    lines = []
    for p in paragraphs:
        if not p:
            continue
        wrapped = textwrap.wrap(p, width=max_chars_per_line)
        lines.extend(wrapped if wrapped else [""])

    line_height = 20
    margin = 12
    img_height = margin * 2 + max(100, len(lines) * line_height)
    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = (
            ImageFont.truetype(font_path, 14)
            if font_path
            else ImageFont.load_default()
        )
    except Exception:
        font = ImageFont.load_default()

    y = margin
    for ln in lines:
        draw.text((margin, y), ln, fill=(0, 0, 0), font=font)
        y += line_height

    return img


# -------------------------
# 合并段落
# -------------------------
def merge_semantic_paragraphs(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    paragraphs, curr = [], []
    section_keywords = [
        "education",
        "experience",
        "project",
        "skills",
        "work",
        "certificate",
    ]

    for line in lines:
        if any(kw in line.lower() for kw in section_keywords):
            if curr:
                paragraphs.append(" ".join(curr))
            curr = [line]
        else:
            curr.append(line)
    if curr:
        paragraphs.append(" ".join(curr))

    # 按 MAX_CHUNK_SIZE 拆分
    final_paragraphs = []
    for para in paragraphs:
        if len(para) <= MAX_CHUNK_SIZE:
            final_paragraphs.append(para)
        else:
            sentences = re.split(r"[。,.]", para)
            chunk = ""
            for s in sentences:
                s = s.strip()
                if len(chunk) + len(s) + 1 <= MAX_CHUNK_SIZE:
                    chunk += s + "。"
                else:
                    final_paragraphs.append(chunk.strip())
                    chunk = s + "。"
            if chunk.strip():
                final_paragraphs.append(chunk.strip())
    return final_paragraphs


# -------------------------
# FAISS 分段逻辑
# -------------------------
ACTION_VERBS = (
    "built",
    "created",
    "used",
    "collected",
    "led",
    "fine-tuned",
    "developed",
    "designed",
    "implemented",
)


def semantic_split(text: str, max_size=MAX_CHUNK_SIZE):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks, curr_chunk = [], []

    for line in lines:
        is_title = (
            len(line) <= 100 and not line.lower().startswith(ACTION_VERBS)
        ) or "project" in line.lower()
        if is_title and curr_chunk:
            chunks.append("\n".join(curr_chunk))
            curr_chunk = [line]
        else:
            curr_chunk.append(line)
    if curr_chunk:
        chunks.append("\n".join(curr_chunk))

    final_chunks = []
    for c in chunks:
        if len(c) <= max_size:
            final_chunks.append(c)
        else:
            sentences, current = re.split(r"[。,.]", c), ""
            for s in sentences:
                s = s.strip()
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
# 文本归一化和去重
# -------------------------
def normalize_text_for_compare(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    for ch in "。。，,、：:；;.-–—()[]{}\"'`··\u3000":
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similar_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def is_duplicate_para(
    new_para: str, existing_list: list, threshold: float = 0.85
) -> bool:
    norm_new = normalize_text_for_compare(new_para)
    if not norm_new:
        return True
    for e in existing_list:
        text = e.get("description") if isinstance(e, dict) else str(e)
        norm_e = normalize_text_for_compare(text)
        if not norm_e:
            continue
        if norm_new in norm_e or norm_e in norm_new:
            return True
        if similar_ratio(norm_new, norm_e) >= threshold:
            return True
    return False
