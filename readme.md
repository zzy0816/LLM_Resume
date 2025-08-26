# LLM Resume Analysis Project

基于 FastAPI 的简历上传与分析系统，使用本地 MinIO 存储简历文件，可扩展到 AWS S3。  

---

## 目录

- [项目简介](#项目简介) 
- [项目目录结构](#项目目录结构)  
- [环境要求](#环境要求)  
- [安装与配置](#安装与配置)  
- [运行 MinIO](#运行-minio)  
- [上传文件](#上传文件)  
- [读取文件 & 分析](#读取文件--分析)  
- [切换 AWS S3](#切换-aws-s3)  
- [示例运行](#示例运行)  
- [注意事项](#注意事项)  
- [每周更新](#每周更新)  

---

## 项目简介

- 功能：上传本地简历文件到 MinIO，读取文件并调用 LLM/LongChain 进行分析  
- 支持多文件并发上传  
- 配置通过 `.env` 文件管理，可轻松切换本地 MinIO 或 AWS S3  

[用户上传简历文件] 
          │
          ▼
      [FastAPI 接口]
          │
          ▼
[并发上传到 MinIO/S3 桶] <─ .env 配置可切换本地/云
          │
          ▼
   [读取文件内容]
          │
          ▼
   [LongChain/LLM 分析]
          │
          ▼
      [返回结果给用户]

---

## 项目目录结构
LLM_Resume_Project/
│
├── README.md
├── .env
├── requirements.txt
├── main.py                 # FastAPI 脚本
├── minio_data/             # 本地 MinIO 数据挂载目录
├── data/                   # 本地测试简历文件存放目录
│   ├── Resume(AI).docx
│   └── Resume(DS)v0.1.docx
└── scripts/                # 可选辅助脚本目录
    └── upload_files.py     # 单独的上传脚本（可用 boto3）

---

## 环境要求
- Python >= 3.10  
- 依赖库：`boto3`、`fastapi`、`python-dotenv`、`uvicorn` 等  
- Docker Desktop（用于运行 MinIO 容器）  

---

## 安装与配置
1. 克隆仓库：
git clone <your-repo-url>
cd <your-repo-folder>

2. 创建 .env 文件，示例内容：
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=admin123
MINIO_BUCKET=resume-bucket

3. 安装 Python 依赖：
pip install -r requirements.txt

## 运行 MinIO
在 Windows PowerShell 中运行：

& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" run -p 9000:9000 -p 9001:9001 `
  -e "MINIO_ROOT_USER=admin" `
  -e "MINIO_ROOT_PASSWORD=admin123" `
  -v D:\project\LLM_Resume\minio_data:/data `
  quay.io/minio/minio server /data --console-address ":9001"

Web 控制台访问：http://localhost:9001
默认用户名/密码：admin / admin123

---

## 上传文件
1. 文件并发上传示例
from concurrent.futures import ThreadPoolExecutor
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
)

bucket_name = os.getenv("MINIO_BUCKET")
files_to_upload = [
    r"D:\project\LLM_Resume\data\Resume(AI).docx",
    r"D:\project\LLM_Resume\data\Resume(DS)v0.1.docx"
]

def upload_file(file_path):
    object_name = os.path.basename(file_path)
    s3.upload_file(Filename=file_path, Bucket=bucket_name, Key=object_name)
    print(f"Uploaded {object_name}")

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(upload_file, files_to_upload)

print("All files uploaded successfully.")

---

## 读取文件 & 分析
obj = s3.get_object(Bucket=bucket_name, Key="resume.pdf")
content = obj['Body'].read()  # bytes 类型
text = content.decode("utf-8")  # 如果是文本文件
print(text)

对接 LongChain / LLM：将 content 或 text 传入分析模块即可

---

## 切换 AWS S3
只需修改 .env：

MINIO_ENDPOINT=https://s3.amazonaws.com
MINIO_ACCESS_KEY=<YOUR_AWS_ACCESS_KEY>
MINIO_SECRET_KEY=<YOUR_AWS_SECRET_KEY>
MINIO_BUCKET=resume-bucket

脚本逻辑 无需改动，上传、读取完全通用

---

## 示例运行
1. 启动 FastAPI：
uvicorn main:app --reload

2. 上传文件接口示例：
POST /upload_resume
Form-data: file=<resume.pdf>

3. 读取分析接口示例：
GET /analyze_resume?filename=resume.pdf

---

## 注意事项
确保 MinIO 容器已运行
Windows 路径建议使用原始字符串 r"path" 避免转义错误
并发上传可提高效率
.env 管理密钥信息，不要硬编码在脚本中

## 每周更新
