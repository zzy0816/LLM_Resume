import sys, os
import logging
import pdfplumber
from docx import Document as DocxDocument

# optional libs for layout / donut
try:
    import fitz
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 400
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

USE_LAYOUTLM = False
USE_DONUT = False

# -------------------------
# 模型加载
# -------------------------
if LayoutLMv3Processor is not None and LayoutLMv3ForTokenClassification is not None and fitz is not None:
    try:
        layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base").to(DEVICE)
        USE_LAYOUTLM = True
        logger.info("LayoutLMv3 loaded (will be used for PDF).")
    except Exception as e:
        logger.warning(f"LayoutLMv3 init failed, fallback to pdfplumber. Error: {e}")

if DonutProcessor is not None and VisionEncoderDecoderModel is not None and Image is not None:
    try:
        donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base").to(DEVICE)
        USE_DONUT = True
        logger.info("Donut loaded (will be used for DOCX).")
    except Exception as e:
        logger.warning(f"Donut init failed, fallback to python-docx text. Error: {e}")

# -------------------------
# DOCX 读取
# -------------------------
def read_docx_paragraphs(docx_path: str):
    from doc_split import render_paragraphs_to_image  # 避免循环引用
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
                if donut_paras:
                    paragraphs = donut_paras
                    logger.info("DOCX parsed by Donut, paragraphs=%d", len(paragraphs))
        except Exception as e:
            logger.warning("Donut parsing failed, fallback to docx text. Error: %s", e)

    logger.info("DOCX split into %d paragraphs", len(paragraphs))
    return paragraphs

# -------------------------
# PDF 读取
# -------------------------
def normalize_box(x0, y0, x1, y1, page_width, page_height):
    try:
        return [
            int(1000 * (x0 / page_width)),
            int(1000 * (y0 / page_height)),
            int(1000 * (x1 / page_width)),
            int(1000 * (y1 / page_height)),
        ]
    except Exception:
        return [0, 0, 1000, 1000]

def read_pdf_paragraphs(pdf_path: str):
    paragraphs = []

    if USE_LAYOUTLM and fitz is not None:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                words = page.get_text("words")
                if not words:
                    continue
                words_text = [w[4] for w in words]
                boxes = [normalize_box(w[0], w[1], w[2], w[3], page.rect.width, page.rect.height) for w in words]
                encoding = layout_processor(images=img, words=words_text, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
                if DEVICE == "cuda":
                    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
                outputs = layout_model(**encoding)
                token_preds = outputs.logits.argmax(-1)[0].cpu().tolist()
                words_out = [words_text[i] for i in range(len(words_text)) if i < len(token_preds) and token_preds[i] > 0]
                paragraphs.extend([" ".join(words_out)])
            doc.close()
            logger.info("PDF parsed by LayoutLMv3, paragraphs=%d", len(paragraphs))
            if paragraphs:
                return paragraphs
        except Exception as e:
            logger.warning("LayoutLMv3 failed, fallback to pdfplumber. Error: %s", e)

    # fallback pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    paragraphs.extend([p.strip() for p in text.split("\n") if p.strip()])
        logger.info("PDF fallback split into %d paragraphs", len(paragraphs))
    except Exception as e:
        logger.error("pdfplumber failed. Error: %s", e)

    return paragraphs

# -------------------------
# 统一接口
# -------------------------
def read_document_paragraphs(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return read_docx_paragraphs(file_path)
    elif ext == ".pdf":
        return read_pdf_paragraphs(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
