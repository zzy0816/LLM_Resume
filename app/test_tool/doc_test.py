import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging
import re
import pdfplumber
from docx import Document as DocxDocument
from difflib import SequenceMatcher
import textwrap

# optional libs for layout / donut
try:
    import fitz  # type: ignore # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

try:
    import torch
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    from transformers import DonutProcessor, VisionEncoderDecoderModel
except Exception:
    torch = None
    LayoutLMv3Processor = None
    LayoutLMv3ForTokenClassification = None
    DonutProcessor = None
    VisionEncoderDecoderModel = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 400

# -------------------------
# 模型加载（可选，有则使用，无则 fallback）
# -------------------------
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

USE_LAYOUTLM = False
USE_DONUT = False

if LayoutLMv3Processor is not None and LayoutLMv3ForTokenClassification is not None and fitz is not None:
    try:
        layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base").to(DEVICE)
        USE_LAYOUTLM = True
        logger.info("LayoutLMv3 loaded (will be used for PDF).")
    except Exception as e:
        logger.warning(f"LayoutLMv3 init failed, will fallback to pdfplumber. Error: {e}")
        USE_LAYOUTLM = False

if DonutProcessor is not None and VisionEncoderDecoderModel is not None and Image is not None:
    try:
        donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base").to(DEVICE)
        USE_DONUT = True
        logger.info("Donut loaded (will be used for DOCX).")
    except Exception as e:
        logger.warning(f"Donut init failed, will fallback to python-docx text. Error: {e}")
        USE_DONUT = False

# -------------------------
# 文档读取（保持接口不变）
# -------------------------
def read_document_paragraphs(file_path: str):
    """
    统一读取 DOCX / PDF 文档为段落列表（接口保持不变）
    内部：PDF 优先走 LayoutLMv3（若可用），DOCX 优先走 Donut（若可用）
    最终返回：list[str]（段落文本），保持和你原脚本完全兼容
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return read_docx_paragraphs(file_path)
    elif ext == ".pdf":
        return read_pdf_paragraphs(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# -------------------------
# DOCX 解析：优先 Donut（若可用），否则 fallback 到 python-docx 文本
# 不修改函数签名，仍返回 list[str]
# -------------------------
def read_docx_paragraphs(docx_path: str):
    doc = DocxDocument(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    if USE_DONUT:
        try:
            img = render_paragraphs_to_image(paragraphs)
            if img is not None:
                pixel_values = donut_processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
                outputs = donut_model.generate(pixel_values, max_length=1024)
                decoded = donut_processor.batch_decode(outputs, skip_special_tokens=True)[0]

                donut_paras = [line.strip() for line in decoded.split("\n") if line.strip()]
                if donut_paras:  # Donut 成功
                    paragraphs = donut_paras
                    logger.info("DOCX parsed by Donut, paragraphs=%d", len(paragraphs))
                else:
                    logger.warning("Donut parsed empty text, fallback to python-docx paragraphs")
        except Exception as e:
            logger.warning("Donut parsing failed, fallback to docx text. Error: %s", e)

    if not paragraphs:
        logger.warning("No paragraphs extracted from DOCX.")
    else:
        logger.info("DOCX split into %d paragraphs", len(paragraphs))
    return paragraphs

def merge_semantic_paragraphs(text: str):
    """
    将 Donut 输出的文本按语义合并为段落
    规则：
    1. 遇到 Education / Project / Experience / Skills 等关键字开始新段
    2. 或者长度超过 MAX_CHUNK_SIZE 切分
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    paragraphs = []
    curr = []

    section_keywords = ["education", "experience", "project", "skills", "work", "certificate"]

    for line in lines:
        line_lower = line.lower()
        # 遇到标题，先保存上一个段落
        if any(kw in line_lower for kw in section_keywords):
            if curr:
                paragraphs.append(" ".join(curr))
            curr = [line]
        else:
            curr.append(line)

    if curr:
        paragraphs.append(" ".join(curr))

    # 再按 MAX_CHUNK_SIZE 拆分过长段落
    final_paragraphs = []
    for para in paragraphs:
        if len(para) <= MAX_CHUNK_SIZE:
            final_paragraphs.append(para)
        else:
            # 按句号/逗号拆
            sentences = re.split(r"[。,.]", para)
            chunk = ""
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(chunk) + len(s) + 1 <= MAX_CHUNK_SIZE:
                    chunk += s + "。"
                else:
                    final_paragraphs.append(chunk.strip())
                    chunk = s + "。"
            if chunk.strip():
                final_paragraphs.append(chunk.strip())

    return final_paragraphs

def render_paragraphs_to_image(paragraphs: list[str], img_width: int = 1200, font_path: str | None = None):
    """
    将纯文本段落渲染到一张长图（内存），以供 Donut 视觉模型使用。
    说明：这不是将 docx 转成 pdf，而是“内存渲染文本为图片”，以满足 Donut 的 image input。
    如果 PIL 不可用，则返回 None（Donut step 会被跳过）。
    """
    if Image is None or ImageDraw is None or ImageFont is None:
        return None

    # 简单换行包装
    max_chars_per_line = 80
    lines = []
    for p in paragraphs:
        if not p:
            continue
        wrapped = textwrap.wrap(p, width=max_chars_per_line)
        if not wrapped:
            lines.append("")
        else:
            lines.extend(wrapped)

    line_height = 20
    margin = 12
    img_height = margin * 2 + max(100, len(lines) * line_height)
    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)

    # 选默认字体（可传入 font_path）
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 14)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    y = margin
    for ln in lines:
        draw.text((margin, y), ln, fill=(0, 0, 0), font=font)
        y += line_height

    return img

# -------------------------
# PDF 解析：优先 LayoutLMv3（若可用），否则 fallback 到 pdfplumber
# 返回 list[str] 段落（不改变外部接口）
# -------------------------
def read_pdf_paragraphs(pdf_path: str):
    if not USE_LAYOUTLM or fitz is None:
        # fallback 原逻辑
        paragraphs = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    page_paras = [p.strip() for p in text.split("\n") if p.strip()]
                    paragraphs.extend(page_paras)
        logger.info("PDF fallback split into %d paragraphs", len(paragraphs))
        return paragraphs

    paragraphs = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            # 渲染页面为图片
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # 获取每个 word bbox
            words = page.get_text("words")
            if not words:
                continue
            words_text = [w[4] for w in words]
            boxes = [normalize_box(w[0], w[1], w[2], w[3], page.rect.width, page.rect.height) for w in words]

            encoding = layout_processor(images=img, words=words_text, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
            if DEVICE == "cuda":
                encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

            outputs = layout_model(**encoding)

            # decode logits → 获取 token text
            token_preds = outputs.logits.argmax(-1)[0].cpu().tolist()
            words_out = [words_text[i] for i in range(len(words_text)) if i < len(token_preds) and token_preds[i] > 0]
            page_paras = [" ".join(words_out)]
            paragraphs.extend(page_paras)

        doc.close()
        logger.info("PDF parsed by LayoutLMv3, paragraphs=%d", len(paragraphs))
        return paragraphs

    except Exception as e:
        logger.warning("LayoutLMv3 failed, fallback to pdfplumber text. Error: %s", e)
        paragraphs = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    page_paras = [p.strip() for p in text.split("\n") if p.strip()]
                    paragraphs.extend(page_paras)
        return paragraphs

def normalize_box(x0, y0, x1, y1, page_width, page_height):
    """
    将坐标归一化为 LayoutLM 期望的 0-1000 整数 bbox 格式
    """
    # fitz coords: 0..page_width / page_height
    try:
        return [
            int(1000 * (x0 / page_width)),
            int(1000 * (y0 / page_height)),
            int(1000 * (x1 / page_width)),
            int(1000 * (y1 / page_height)),
        ]
    except Exception:
        return [0, 0, 1000, 1000]

# -------------------------
# FAISS 分段（增强教育段落保留短文本）
# （未改动你原先逻辑）
# -------------------------
ACTION_VERBS = ("built", "created", "used", "collected", "led", "fine-tuned", "developed", "designed", "implemented")

def semantic_split(text: str, max_size=MAX_CHUNK_SIZE):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks = []
    curr_chunk = []

    for line in lines:
        line_lower = line.lower()
        is_title = (len(line) <= 100 and not line_lower.startswith(ACTION_VERBS)) or "project" in line_lower

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
# （未改动）
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

def is_duplicate_para(new_para: str, existing_list: list, threshold: float = 0.85) -> bool:
    norm_new = normalize_text_for_compare(new_para)
    if not norm_new:
        return True
    for e in existing_list:
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
