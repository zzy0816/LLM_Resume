import logging
from scripts.upload_llm import (
    read_docx_paragraphs,
    classify_paragraphs,
    build_faiss_with_category,  
    summarize_full_category,     
    query_dynamic_category,       # 确保导入
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    resume_path = r"D:\project\LLM_Resume\data\Resume(AI).docx"

    # 1) 读取 + 分类
    paragraphs = read_docx_paragraphs(resume_path)
    classified = classify_paragraphs(paragraphs)

    # 2) 构建向量库
    faiss_db = build_faiss_with_category(classified)

    # 3) 全类别安全总结
    categories = ["WorkExperience", "Project", "Education", "Skills", "Other"]

    print("\n====== 简历自动分类分析报告 ======\n")
    for cat in categories:
        print(f"\n=== {cat} ===")
        result = summarize_full_category(classified, cat)
        print(result)

    # 4) 控制台动态问答
    print("\n====== 向量库问答（输入 q 退出） ======\n")
    while True:
        user_query = input("请输入查询内容: ").strip()
        if user_query.lower() in ["q", "quit", "exit"]:
            print("退出问答。")
            break
        if not user_query:
            continue

        # 可选类别过滤：True 表示按 query 自动识别类别
        result_text = query_dynamic_category(faiss_db, user_query, top_k=5, use_category_filter=True)
        print("\n" + result_text + "\n")
