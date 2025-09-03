# streamlit_app.py
import streamlit as st
import requests
import os
from scripts.storage_client import StorageClient

# 创建 downloads 目录
downloads_dir = "./downloads"
os.makedirs(downloads_dir, exist_ok=True)

st.title("简历分析系统")

# 初始化 StorageClient
client = StorageClient()

# =========================
# 选择使用 MinIO 现有文件 或 上传文件
# =========================
mode = st.radio("选择文件来源", ("MinIO 文件", "上传本地文件"))

file_name = ""
local_path = None

if mode == "MinIO 文件":
    # 获取 MinIO bucket 文件列表
    try:
        objects = [obj['Key'] for obj in client.s3.list_objects_v2(Bucket=client.bucket).get('Contents', [])]
    except Exception:
        objects = []
    if objects:
        file_name = st.selectbox("选择已有文件", objects)
    else:
        st.warning("MinIO 中没有文件，请先上传。")

elif mode == "上传本地文件":
    uploaded_file = st.file_uploader("上传简历 (.docx)", type=["docx"])
    if uploaded_file is not None:
        local_path = os.path.join(downloads_dir, uploaded_file.name)
        with open(local_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"{uploaded_file.name} 已保存到本地。")

        # 上传到 MinIO
        if st.button("上传到 MinIO"):
            client.upload_file(local_path)
            st.success(f"{uploaded_file.name} 上传到 MinIO 成功！")
            file_name = uploaded_file.name

# =========================
# 分析简历
# =========================
if st.button("分析简历"):
    if mode == "MinIO 文件" and not file_name:
        st.error("请先选择 MinIO 文件")
    elif mode == "上传本地文件" and not local_path and not file_name:
        st.error("请先上传文件")
    else:
        # 如果是 MinIO 文件，需要下载到本地
        if mode == "MinIO 文件":
            local_path = os.path.join(downloads_dir, file_name)
            client.read_file(file_name, local_path)

        # 调用 FastAPI 分析
        response = requests.post(
            "http://127.0.0.1:8000/analyze_resume",
            json={"file_name": file_name}
        )
        if response.ok:
            report = response.json()["report"]
            st.success("简历分析完成 ✅")
            for cat, text in report.items():
                st.subheader(cat)
                st.text(text)
        else:
            st.error(f"分析失败: {response.text}")

# =========================
# 问答功能
# =========================
query_text = st.text_input("问答查询")
if st.button("提交问题") and query_text:
    response = requests.post(
        "http://127.0.0.1:8000/query_resume",
        json={"query": query_text}
    )
    if response.ok:
        answer = response.json().get("answer", "")
        st.subheader("答案")
        st.text(answer)
    else:
        st.error(f"查询失败: {response.text}")
