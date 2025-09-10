import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def load_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("yashpwr/resume-ner-bert-v2")
    model = AutoModelForTokenClassification.from_pretrained("yashpwr/resume-ner-bert-v2")
    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipe

ner_pipeline = load_ner_pipeline()

# -------------------------
# 批量 NER
# -------------------------
def run_ner_batch(paragraphs: list[str]) -> list[list[dict]]:
    """对段落列表做批量 NER，返回每段的实体结果"""
    results = ner_pipeline(paragraphs, batch_size=8, truncation=True)
    # transformers pipeline 返回的可能是 flat list，需要按段落分组
    if isinstance(results[0], dict):
        return [results]  # 单段落
    # 已经是列表嵌套
    return results
