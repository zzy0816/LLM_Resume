import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging, json, random, time, os

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "ner_service",
            "message": record.getMessage(),
            "request_id": str(random.randint(1000, 9999))
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

# -------------------------
# Singleton / Lazy load NER
# -------------------------
_ner_pipeline = None

def get_ner_pipeline():
    """懒加载 NER pipeline，singleton，保证只初始化一次"""
    global _ner_pipeline
    if _ner_pipeline is None:
        logger.info("Loading NER model for the first time...")
        tokenizer = AutoTokenizer.from_pretrained("yashpwr/resume-ner-bert-v2")
        model = AutoModelForTokenClassification.from_pretrained("yashpwr/resume-ner-bert-v2")
        _ner_pipeline = pipeline(
            "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
        )
        logger.info("NER model loaded successfully.")
    return _ner_pipeline

# -------------------------
# 批量 NER
# -------------------------
def run_ner_batch(paragraphs: list[str]) -> list[list[dict]]:
    """对段落列表做批量 NER，返回每段的实体结果"""
    ner_pipe = get_ner_pipeline()
    results = ner_pipe(paragraphs, batch_size=8)

    # transformers pipeline 返回的可能是 flat list，需要按段落分组
    if isinstance(results[0], dict):
        return [results]  # 单段落
    return results
