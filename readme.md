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
├── frontend.py             # streamlit 了一个简单的前端
├── minio_data/             # 本地 MinIO 数据挂载目录
├── data/                   # 本地测试简历文件存放目录
├── downloads/              # 储存从MINIO下载到本地的文件
│   ├── Resume(AI).docx
│   └── Resume(DS)v0.1.docx
└── scripts/                # 可选辅助脚本目录
    ├── upload_langchain_nollm.py # 不使用LLM,只用sentence_transform模型分段
    ├── upload_llm.py # 使用LLM分类,单独查询一个分类
    ├── upload_llm_v0.py # upload_llm.py 的初始版,保留防止意外
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

简历处理主流程：
1. 读取 Word 段落
2. 段落语义拆分
3. 正则兜底邮箱,电话等等
4. 自定义结构化JSON
5. 用NER解析成结构化字典
6. 自动补全缺失字段
7. 如果提供 FAISS，则进行语义补全
8. 返回最终结构化字典

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
main.py 或 strealit run frontend.py

{
    "message": "简历分析完成",
    "report": {
        "WorkExperience": [
            {
                "company": "New York Technology & Management LLC",
                "title": "Programming Manager",
                "start_date": "2025",
                "end_date": "Present",
                "description": "New York Technology & Management LLC | Programming Manager | New York, NY | May 2025 – Present。"
            },
            {
                "company": "Yangtze River Consulting Service LLC",
                "title": "Ethical Consultant",
                "start_date": "Unknown",
                "end_date": "Unknown",
                "description": "Yangtze River Consulting Service LLC | Ethical Consultant | Piscataway Township, NJ | Sep 2022 – Jul 2023。"
            }
        ],
        "Project": [
            {
                "description": "Created a token counter in Ollama to efficiently track and manage token usage.。"
            },
            {
                "description": "Fine-tuned and quantized Hugging Face model on custom data for optimized, faster inference.。"
            },
            {
                "description": "Built Ollama-based chatbot with memory, vector DB(FAISS), with LangChain and Flask.。"
            },
            {
                "description": "Used Ollama to analyze files and generate PowerPoint reports, improving reporting workflows.。"
            },
            {
                "description": "Built and deployed app for NGO to auto-publish messages via AWS, boosting engagement 20%.。"
            },
            {
                "description": "Built flower AI assistant to answer inquiries, boosting online sales and reducing workload.。"
            },
            {
                "description": "Led COVID-19 time series and survey data analysis, focusing on cleaning, EDA, and modeling.。"
            },
            {
                "description": "Built a QA system with Transformer, improving automated question answering accuracy.。"
            },
            {
                "description": "Collected Bitfinex API data, trained transformer model with historical data, added vector database.。"
            },
            {
                "description": "Using metal model to stack transformer model and sentiment model to get 84% accuracy.。"
            }
        ],
        "Education": [
            {
                "school": "Northeastern University",
                "degree": "Master of Professional Study in Applied Machine Intelligence",
                "grad_date": "2025",
                "description": "Northeastern University | Master of Professional Study in Applied Machine Intelligence | Boston, MA | April 2025 | GPA: 3.9/4.0。"
            },
            {
                "school": "University of Connecticut",
                "degree": "Bachelor of Art",
                "grad_date": "N/A",
                "description": "University of Connecticut | Bachelor of Art | Storrs, CT | May 2022Skills。"
            }
        ],
        "Skills": [
            "Hugging face",
            "Langchain",
            "Numpy",
            "Ollama",
            "Pandas",
            "Python",
            "Pytorch",
            "SQL",
            "Scikit-learn",
            "Seaborn",
            "Tableau",
            "Tensorflow"
        ]
    }
}
---

## 注意事项
确保 MinIO 容器已运行
并发上传可提高效率
.env 管理密钥信息，不要硬编码在脚本中

## 更新目标
1. 目前的简历分析抓取中设置了分类, 未来也许可以考虑完全去除手动设置,全自动分析(已实现)
2. FASTAPI:用户可以在前端(PostMan)选择文件,查看文件分析结果,适当追问,等等 (已实现)
3. 继续优化LLM的分析能力 (已实现: 数据结构化+NER模型抽取+正则兜底)
4. 继续强化这三步: 更多结构化内容+抽取更完善(不再从中截断)+正则兜底

## 每周更新

# week 1: 
1. 项目创建: 虚拟环境, 项目基本架构(数据,脚本,环境文件,requirement,readme...)GitHub上传
2. 因为AWS无法创建免费账号,使用用为S3和boto的MINIO代替
3. storage_client.py 管理文件上传(可以并发)和下载(到本地) 到MINIOserver
4. upload_langchain_nollm.py 仅用docx格式如换行,句号等分类, 想输出所有相关内容,会分段过大,想分段合适会无法精确输出
5. upload_llm.py 使用LLM为分好的段落分类,同时添加目标类别过滤,大多数情况可以精确分类(模型能力有限,不是每次运行都是完美,偶尔在分类中会出现不是特别相关的段落), 同时可以利用向量库进行追问回答

# week 2
1. 添加FASTAPI, 可以通过postman进行简历分析 和 追问
2. 用streamlit做了一个简单的前端,用户可以通过前端上传本地文件到MINIO,以及提供分析和追问功能


3. 优化: 
    - 1. fallback 增加关键词+全局语义模型
    - 2. 上传/下载时加上异常处理: 重试和缓存 
    - 3. faiss和classified 存储到本地/data文件夹下
4. 优化分类能力: 
    - 1. 数据结构化: 读取 Word 段落, 段落语义拆分, 文本和分类归一化去重, 技能提取与规范化, 自定义JSON结构包括大模块和模块包括的内容, 正则兜底邮箱,电话等等
    - 2. NER模型抽取: 使用 yashpwr/resume-ner-bert-v2等做parsing, 解析成结构化字典, 使用语义回退 fallback, 去空+分类+对技能段落规范化+去重, 自动补全缺失字段,  返回最终结构化字典, 根据段落列表构建 FAISS 向量数据库并保存, 根据查询语义动态匹配分类, 根据 query 精准返回相关字段内容
    - 3. 总流程: 1. 加载或解析简历(看data是否有classified), 2. 构建或加载 FAISS(看是否有faiss), 3. 查询 FAISS 并生成 query_results (NER模型解析抽取), 4. 使用 query 结果填充结构化 JSON (完成数据格式化), # 5. 保存最终 JSON 到 faiss文件夹

