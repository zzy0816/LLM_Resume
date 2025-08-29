# LLM Resume Analysis Project

基于 FastAPI 的简历上传与分析系统，使用本地 MinIO 存储简历文件，可扩展到 AWS S3。  

---

## 目录

- [项目简介](#项目简介) 
- [项目目录结构](#项目目录结构)  
- [环境要求](#环境要求)  
- [安装与配置](#安装与配置)  
- [运行 MinIO](#运行-minio)  
- [上传&下载文件](#上传&下载文件)  
- [文件分段分类抓取](#文件分段分类抓取)  
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
├── README.md
├── .env
├── requirements.txt
├── main.py                 # 使用LLM查询简历的所有分类
├── minio_data/             # 本地 MinIO 数据挂载目录
├── data/                   # 本地测试简历文件存放目录
│   ├── Resume(AI).docx
│   └── Resume(DS)v0.1.docx
└── scripts/                # 可选辅助脚本目录
    ├── upload_langchain_nollm.py # 不使用LLM,只用sentence_transform模型分段
    ├── upload_llm.py # 使用LLM分类,单独查询一个分类
    └── storage_client.py     # 上传和下载简历（可用 boto3）

---

## 环境要求
- Python >= 3.10  
- 依赖库：`boto3`、`fastapi`、`python-dotenv`、`uvicorn` 等  
- Docker Desktop（用于运行 MinIO 容器）  

---

## 安装与配置
1. 创建 .env 文件，示例内容：
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=admin123
MINIO_BUCKET=resume-bucket

2. 安装 Python 依赖：
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

## 上传&下载文件
1. 文件上传支持单独上传和并发
2. 可以下载文件到本地

---

## 文件分段分类抓取

1. 用docx分段落
2. 用LLM为每段分类
3. 用LangChain为每段生成faiss向量库
4. 根据要求输出对应分类的段落

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
main.py

""""""""""""""""""""""""""""""""
====== 简历自动分类分析报告 ======


=== WorkExperience ===
=== WorkExperience 原文内容 ===

1. Professional Experience

2. Yangtze River Consulting Service LLC | Ethical Consultant | Piscataway Township, NJ | Sep 2022 – Jul 2023

3. New York Technology & Management LLC | Programming Manager | New York, NY | May 2025 – Present



=== Project ===
=== Project 原文内容 ===

1. Built a QA system with Transformer, improving automated question answering accuracy.

2. Built and deployed app for NGO to auto-publish messages via AWS, boosting engagement 20%.

3. Built React agent with LangChain tool-calling to automate internal tool, reducing manual work.

4. Created a token counter in Ollama to efficiently track and manage token usage.

5. Built multi-agent system for sales and after-sales, streamlining communication and task flow.

6. Built a LangChain agent to automate Gmail email tracking and improve communication.

7. Projects

8. Built flower AI assistant to answer inquiries, boosting online sales and reducing workload.

9. Built Ollama-based chatbot with memory, vector DB(FAISS), with LangChain and Flask.

10. Crypto Predicted and Analysis with Vester AI

11. Built a Crypto analysis tool with transformer model and LLM for Realtime market prediction.

12. Collected Bitfinex API data, trained transformer model with historical data, added vector database.

13. Using metal model to stack transformer model and sentiment model to get 84% accuracy.



=== Education ===
=== Education 原文内容 ===

1. Northeastern University | Master of Professional Study in Applied Machine Intelligence | Boston, MA | April 2025 | GPA: 3.9/4.0

2. University of Connecticut | Bachelor of Art | Storrs, CT | May 2022Skills



=== Skills ===
=== Skills 原文内容 ===

1. Languages & Tools: Python, SQL, Tableau

2. Frameworks & Libraries: TensorFlow, Pytorch, Scikit-learn, Pandas, NumPy, Seaborn, Langchain

3. Platforms & Tech: Ollama, Hugging Face

4. Led COVID-19 time series and survey data analysis, focusing on cleaning, EDA, and modeling.

5. Used Ollama to analyze files and generate PowerPoint reports, improving reporting workflows.

6. Increased online orders by 10% and reduced manual customer service workload by 40%.



=== Other ===
=== Other 原文内容 ===

1. Zhenyu Zhang

2. Phone: +1 (860) 234-7101 | Email: Zhang.zhenyu6@northeastern.edu | Linkedin Profile | GithubCareer Goal

3. Applied Machine Intelligence with solid experience in Machine learning, Data Analysis, software development, and LLM. Seeking roles in those and others AI-related fields.Education

4. Industry Experience

5. Fine-tuned and quantized Hugging Face model on custom data for optimized, faster inference.

6. Flower Market Assistant

7. Sourced Code available on GitHub: Flower-Market-Assistant



====== 向量库问答（输入 q 退出） ======

请输入查询内容: machine learning experience

1. New York Technology & Management LLC | Programming Manager | New York, NY | May 2025 – Present。



请输入查询内容: q
退出问答。

""""""""""""""""""""""

---

## 注意事项
确保 MinIO 容器已运行
并发上传可提高效率
.env 管理密钥信息，不要硬编码在脚本中

## 更新目标
1. 目前的简历分析抓取中设置了分类, 未来也许可以考虑完全去除手动设置,全自动分析(可选)
2. FASTAPI:用户可以在前端(PostMan)选择文件,查看文件分析结果,适当追问,等等

## 每周更新
1. 项目创建: 虚拟环境, 项目基本架构(数据,脚本,环境文件,requirement,readme...)GitHub上传
2. 因为AWS无法创建免费账号,使用用为S3和boto的MINIO代替
3. storage_client.py 管理文件上传(可以并发)和下载(到本地) 到MINIOserver
4. upload_langchain_nollm.py 仅用docx格式如换行,句号等分类, 想输出所有相关内容,会分段过大,想分段合适会无法精确输出
5. upload_llm.py 使用LLM为分好的段落分类,同时添加目标类别过滤,大多数情况可以精确分类(模型能力有限,不是每次运行都是完美,偶尔在分类中会出现不是特别相关的段落), 同时可以利用向量库进行追问回答
