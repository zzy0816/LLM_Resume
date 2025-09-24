import json
import logging
import os
import random
import sys

import fitz # PyMuPDF
import pdfplumber
import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from docx import Document as DocxDocument
from PIL import Image
from transformers import (
    DonutProcessor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    VisionEncoderDecoderModel,
)

from app.qre.doc_split import render_paragraphs_to_image  # 避免循环引用


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "ner_service",
            "message": record.getMessage(),
            "request_id": str(random.randint(1000, 9999)),
        }
        return json.dumps(log)


# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)

# 设置日志 handler
handler = logging.FileHandler("logs/app.log")
handler.setFormatter(JsonFormatter())

logger = logging.getLogger()  # root logger
logger.addHandler(handler)
logger.setLevel(logging.INFO)

MAX_CHUNK_SIZE = 400
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

# -------------------------
# singleton + lazy load
# -------------------------
_layout_processor = None
_layout_model = None
_USE_LAYOUTLM = None

_donut_processor = None
_donut_model = None
_USE_DONUT = None


def _load_layoutlm():
    global _layout_processor, _layout_model, _USE_LAYOUTLM
    if _layout_processor is None or _layout_model is None:
        if (
            LayoutLMv3Processor is None
            or LayoutLMv3ForTokenClassification is None
            or fitz is None
        ):
            _USE_LAYOUTLM = False
            return None, None, _USE_LAYOUTLM
        try:
            _layout_processor = LayoutLMv3Processor.from_pretrained(
                "microsoft/layoutlmv3-base"
            )
            _layout_model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base"
            ).to(DEVICE)
            _USE_LAYOUTLM = True
            logger.info("LayoutLMv3 loaded (lazy load).")
        except Exception as e:
            logger.warning(f"LayoutLMv3 init failed: {e}")
            _USE_LAYOUTLM = False
    return _layout_processor, _layout_model, _USE_LAYOUTLM


def _load_donut():
    global _donut_processor, _donut_model, _USE_DONUT
    if _donut_processor is None or _donut_model is None:
        if (
            DonutProcessor is None
            or VisionEncoderDecoderModel is None
            or Image is None
        ):
            _USE_DONUT = False
            return None, None, _USE_DONUT
        try:
            _donut_processor = DonutProcessor.from_pretrained(
                "naver-clova-ix/donut-base"
            )
            _donut_model = VisionEncoderDecoderModel.from_pretrained(
                "naver-clova-ix/donut-base"
            ).to(DEVICE)
            _USE_DONUT = True
            logger.info("Donut loaded (lazy load).")
        except Exception as e:
            logger.warning(f"Donut init failed: {e}")
            _USE_DONUT = False
    return _donut_processor, _donut_model, _USE_DONUT


# -------------------------
# DOCX 读取
# -------------------------
def read_docx_paragraphs(docx_path: str):

    paragraphs = [
        p.text.strip()
        for p in DocxDocument(docx_path).paragraphs
        if p.text.strip()
    ]

    donut_processor, donut_model, USE_DONUT = _load_donut()
    if USE_DONUT:
        try:
            img = render_paragraphs_to_image(paragraphs)
            if img is not None:
                pixel_values = donut_processor(
                    images=img, return_tensors="pt"
                ).pixel_values.to(DEVICE)
                outputs = donut_model.generate(pixel_values, max_length=1024)
                decoded = donut_processor.batch_decode(
                    outputs, skip_special_tokens=True
                )[0]
                donut_paras = [
                    line.strip()
                    for line in decoded.split("\n")
                    if line.strip()
                ]
                if donut_paras:
                    paragraphs = donut_paras
                    logger.info(
                        "DOCX parsed by Donut, paragraphs=%d", len(paragraphs)
                    )
        except Exception as e:
            logger.warning(
                "Donut parsing failed, fallback to docx text. Error: %s", e
            )

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
    layout_processor, layout_model, USE_LAYOUTLM = _load_layoutlm()

    if USE_LAYOUTLM and fitz is not None:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples
                )
                words = page.get_text("words")
                if not words:
                    continue
                words_text = [w[4] for w in words]
                boxes = [
                    normalize_box(
                        w[0],
                        w[1],
                        w[2],
                        w[3],
                        page.rect.width,
                        page.rect.height,
                    )
                    for w in words
                ]
                encoding = layout_processor(
                    images=img,
                    words=words_text,
                    boxes=boxes,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                )
                if DEVICE == "cuda":
                    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
                outputs = layout_model(**encoding)
                token_preds = outputs.logits.argmax(-1)[0].cpu().tolist()
                words_out = [
                    words_text[i]
                    for i in range(len(words_text))
                    if i < len(token_preds) and token_preds[i] > 0
                ]
                paragraphs.extend([" ".join(words_out)])
            doc.close()
            logger.info(
                "PDF parsed by LayoutLMv3, paragraphs=%d", len(paragraphs)
            )
            if paragraphs:
                return paragraphs
        except Exception as e:
            logger.warning(
                "LayoutLMv3 failed, fallback to pdfplumber. Error: %s", e
            )

    # fallback pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    paragraphs.extend(
                        [p.strip() for p in text.split("\n") if p.strip()]
                    )
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
