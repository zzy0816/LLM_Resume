from docx import Document
from utils import extract_basic_info

def docx_to_text(path):
    doc = Document(path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    return "\n".join(full_text)

text = docx_to_text(r"D:\project\LLM_Resume\downloads\Resume(AI).docx")

print(text[:500])  # 打印前几百字符确认内容

info = extract_basic_info(text)
print("Extracted info:", info)

