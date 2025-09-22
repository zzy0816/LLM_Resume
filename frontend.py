# frontend.py  (aka streamlit_app.py)
import os
import requests
import streamlit as st
from app.storage.storage_client import StorageClient

# -----------------------------
# 基础初始化
# -----------------------------
st.set_page_config(page_title="简历分析系统", page_icon="📝", layout="centered")

# 本地下载目录
DOWNLOADS_DIR = "./downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

st.title("简历分析系统")

# -----------------------------
# 会话状态：避免脚本重跑丢失选择
# -----------------------------
for k, v in {
    "file_name": "",
    "local_path": None,
    "mode": "MinIO 文件"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# 初始化 StorageClient
# -----------------------------
try:
    client = StorageClient()
    s3 = client.s3
    bucket = client.bucket
except Exception as e:
    st.error(f"初始化存储客户端失败：{e}")
    st.stop()

# -----------------------------
# 文件来源选择
# -----------------------------
st.session_state.mode = st.radio("选择文件来源", ("MinIO 文件", "上传本地文件"), index=0)

# ====== MinIO 文件模式 ======
if st.session_state.mode == "MinIO 文件":
    with st.spinner("加载 MinIO 文件列表中..."):
        try:
            resp = s3.list_objects_v2(Bucket=bucket)
            objects = [obj["Key"] for obj in resp.get("Contents", [])] if resp.get("KeyCount", 0) > 0 else []
        except Exception as e:
            objects = []
            st.error(f"读取 MinIO 文件列表失败：{e}")

    if objects:
        # 只展示 .docx 文件（可按需放开）
        docx_objects = [k for k in objects if k.lower().endswith(".docx")] or objects
        selected = st.selectbox("选择已有文件", docx_objects, index=0 if st.session_state.file_name == "" else
                                max(0, docx_objects.index(st.session_state.file_name)) if st.session_state.file_name in docx_objects else 0)
        st.session_state.file_name = selected
        st.session_state.local_path = None   # 还未下载
    else:
        st.warning("MinIO 中没有文件，请先上传（或切换到“上传本地文件”）。")

# ====== 上传本地文件模式 ======
else:
    uploaded_file = st.file_uploader("上传简历（.docx）", type=["docx"])
    if uploaded_file is not None:
        # 先写到本地
        local_path = os.path.join(DOWNLOADS_DIR, uploaded_file.name)
        try:
            with open(local_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} 已保存到本地：{local_path}")
            st.session_state.local_path = local_path
            st.session_state.file_name = uploaded_file.name
        except Exception as e:
            st.error(f"保存本地文件失败：{e}")
            st.session_state.local_path = None

        # 上传到 MinIO 按钮
        if st.button("上传到 MinIO"):
            if st.session_state.local_path is None:
                st.error("本地文件不存在，无法上传。")
            else:
                with st.spinner("正在上传到 MinIO..."):
                    try:
                        # 可能 StorageClient 内部吞异常，这里上传后主动校验
                        client.upload_file(st.session_state.local_path, st.session_state.file_name)
                        try:
                            # 上传后校验对象是否存在
                            s3.head_object(Bucket=bucket, Key=st.session_state.file_name)
                            st.success(f"✅ 已上传到 MinIO：{st.session_state.file_name}")
                        except Exception as he:
                            st.error(f"上传后校验失败，MinIO 中未找到对象：{he}")
                    except Exception as e:
                        st.error(f"上传到 MinIO 失败：{e}")

# -----------------------------
# 分析简历
# -----------------------------
st.markdown("---")
st.caption(f"当前选中文件：**{st.session_state.file_name or '（未选择）'}**")

def _download_from_minio_to_local(key: str) -> str | None:
    """将 MinIO 中对象下载到本地 downloads 目录，返回本地路径或 None"""
    local_path = os.path.join(DOWNLOADS_DIR, os.path.basename(key))
    try:
        client.read_file(key, local_path)
        if os.path.exists(local_path):
            return local_path
        return None
    except Exception as e:
        st.error(f"下载 {key} 失败：{e}")
        return None

if st.button("分析简历"):
    # 前置校验
    if st.session_state.mode == "MinIO 文件":
        if not st.session_state.file_name:
            st.error("请先在 MinIO 中选择文件。")
            st.stop()
        # 下载到本地（供后续模块需要）
        with st.spinner("从 MinIO 下载到本地..."):
            lp = _download_from_minio_to_local(st.session_state.file_name)
            if lp is None:
                st.error("文件下载失败，请检查 MinIO 配置或对象名称。")
                st.stop()
            st.session_state.local_path = lp
            st.info(f"已下载到：{lp}")
    else:
        # 上传模式：已在本地
        if not st.session_state.local_path or not os.path.exists(st.session_state.local_path):
            st.error("请先选择并上传本地文件。")
            st.stop()

    # 调用后端 FastAPI
    with st.spinner("正在分析简历..."):
        try:
            resp = requests.post(
                "http://127.0.0.1:8000/analyze_resume",
                json={"file_name": st.session_state.file_name},
                timeout=60
            )
        except requests.exceptions.ConnectionError as e:
            st.error(f"后端连接失败：{e}")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("请求超时（60s）。请检查后端性能或网络。")
            st.stop()
        except Exception as e:
            st.error(f"请求异常：{e}")
            st.stop()

        if not resp.ok:
            st.error(f"分析失败：HTTP {resp.status_code} - {resp.text}")
        else:
            try:
                data = resp.json()
                report = data.get("report", {})
            except Exception as e:
                st.error(f"返回内容非 JSON 或解析失败：{e}\n原始内容：{resp.text[:500]}")
                st.stop()

            st.success("✅ 简历分析完成")
            if not report:
                st.warning("后端未返回 report 字段或为空。")
            else:
                for cat, text in report.items():
                    st.subheader(cat)
                    st.json(text)

# -----------------------------
# 问答功能
# -----------------------------
st.markdown("---")
query_text = st.text_input("问答查询（针对简历）")

if st.button("提交问题") and query_text:
    if not st.session_state.file_name:
        st.error("请先选择或上传简历文件，并完成分析。")
    else:
        with st.spinner("查询中..."):
            try:
                resp = requests.post(
                    "http://127.0.0.1:8000/query_resume",
                    json={
                        "file_name": st.session_state.file_name,
                        "query": query_text
                    },
                    timeout=60
                )
            except requests.exceptions.ConnectionError as e:
                st.error(f"后端连接失败：{e}")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("请求超时（60s）。")
                st.stop()
            except Exception as e:
                st.error(f"请求异常：{e}")
                st.stop()

            if not resp.ok:
                st.error(f"查询失败：HTTP {resp.status_code} - {resp.text}")
            else:
                try:
                    answer = resp.json().get("answer", "")
                except Exception as e:
                    st.error(f"返回内容非 JSON 或解析失败：{e}\n原始内容：{resp.text[:500]}")
                    st.stop()

                st.subheader(f"答案（文件：{st.session_state.file_name}）")
                if isinstance(answer, (dict, list)):
                    st.json(answer)  # ✅ JSON 结构化展示
                else:
                    st.text(answer if answer else "(空)")
