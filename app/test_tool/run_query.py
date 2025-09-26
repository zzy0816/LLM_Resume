import os
import json
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from app.qre.query import query_dynamic_category
from app.storage.db import load_resume  # 你的 load_resume 函数
from app.utils.files import load_faiss, CLASSIFIED_DIR

# -----------------------------
# 配置路径
# -----------------------------
FAISS_DIR = "./data/faiss"

# 简历文件名（与 pipeline 保存一致）
resume_file = "Resume_AI_.pdf"

# -----------------------------
# 加载 JSON
# -----------------------------
json_path = os.path.join(CLASSIFIED_DIR, f"{resume_file}.json")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"找不到 JSON 文件: {json_path}")

with open(json_path, "r", encoding="utf-8") as f:
    structured_resume = json.load(f)

print(f"[INFO] 已加载 JSON: {json_path}")

# -----------------------------
# 加载 FAISS
# -----------------------------
db = load_faiss(resume_file)
if db is None:
    raise FileNotFoundError(f"找不到 FAISS 文件: {resume_file}")

print(f"[INFO] 已加载 FAISS: {resume_file}")

# -----------------------------
# 测试查询
# -----------------------------
query = "工作经历"
result = query_dynamic_category(db, structured_resume, query, top_k=5, use_category_filter=True)

print(f"查询: {query}")
print("结果:")
print(json.dumps(result, ensure_ascii=False, indent=2))
