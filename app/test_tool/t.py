import logging
import os
import pprint
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from app.utils.files import load_faiss, load_json
from app.utils.utils import sanitize_filename

file_name = "Resume(AI).docx"
safe_name = sanitize_filename(file_name)

json_data = load_json(safe_name)
faiss_db = load_faiss(safe_name)

print("JSON:", "找到" if json_data else "未找到")
print("FAISS:", "找到" if faiss_db else "未找到")
