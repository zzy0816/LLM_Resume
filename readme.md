# LLM Resume Analysis Project

一个基于 **NER 模型** 和 **FastAPI** 的简历分析系统，用于自动解析上传的简历文件，提取结构化信息并存储到 **MongoDB**。  
项目支持本地 **MinIO** 文件存储，并可轻松扩展到 **AWS S3**。系统特点包括高准确率的实体识别、灵活的 API 接口以及可扩展的数据存储方案。  

---

## 目录

- [项目简介](#项目简介) 
- [项目目录结构](#项目目录结构)  
- [环境要求](#环境要求)  
- [安装与配置](#安装与配置)
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
   [NER 模型 及 规则化 分析]
          │
          ▼
      [返回结果给用户]

---

## 项目目录结构
LLM_Resume_Project/
├── readme.md
├── .env
├── requirements.txt
├── frontend.py             # streamlit 了一个简单的前端
├── minio_data/             # 本地 MinIO 数据挂载目录
├── .github/                # GITHUB ACTION 的 CI/CD 运行脚本 
├── logs/                   # 日志
└── elk/                    # elastic,logstash, Kibana, 
    ├── filebeat.yml 设置filebeat 可以Kibana查看日志
    ├── logstash.conf 设置logstash 可以Kibana查看日志
    └── init.ps1 启动elk 可以自动记录日志
├── auto_format.py/         # 代码风格正式化
├── fronted.py/             # streanlit 前端代码
├── docker-compose.yml/     # docker下载启动MINIO(S3数据储存), ELK(elastksearch,filebeat,logstash,kibana)
└── data/                   # 本地测试简历文件存放目录
    ├── classified 各个文件的json缓存
    ├── faiss 各个文件的faiss缓存
    ├── Resume(AI).docx
    ├── Resume(AI).pdf
    └── Resume(DS)v0.1.docx
└── downloads/              # 储存从MINIO下载到本地的文件
    ├── Resume(AI).docx
    └── Resume(DS)v0.1.docx
└── app/
    ├── main.py             # 使用fastAPI后台脚本
    └── pipline/              
        └── pipline.py          # 总流程 
    └── qre/                    # 解析合集
        ├── doc_read.py         # 读取 doc/pdf 文件
        ├── doc_split.py        # faiss分段, 去重/归一化
        ├── ner.py              # 加载ner模型
        ├── parser.py           # 分类和结构化
        ├── query.py            # 查询分类段落和覆盖结构化json
        └── semantic.py         # build_faiss 生成faiss的
    └── storage/              
        ├── db.py               # mongodb上传
        └── storage_client.py   # 上传和下载简历（可用 boto3）
    └── test_tool/              # 各个函数的单独测试
        ├── doc_read_test.py        
        ├── doc_split_test.py        
        ├── ner_test.py              
        ├── parser_test.py           
        ├── query_test.py
        ├── pipline_test.py
        ├── run_parser.py       #测试总流程中到解析为止的部分
        ├── run_faiss.py        #测试总流程中解析后, faiss为主的部分    
        └── semantic_test.py         
    └── utils/              # 储存从MINIO下载到本地的文件
        ├── utils.py            # 各种不涉及其他模块的辅助工具
        └── utils_parser.py     # 因为semantic_fallback涉及其他模块,放utils会造成循环调用,故隔离开                  

---

## 环境要求
- Python >= 3.10  
- 依赖库：`boto3`、`fastapi`、`python-dotenv`、`uvicorn` 等  
- Docker Desktop（用于运行 MinIO 以及 ELK 容器）  

---

## 安装与配置
1. 创建 .env 文件，示例内容：
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=admin123
MINIO_BUCKET=resume-bucket

2. 安装 Python 依赖：
pip install -r requirements.txt

---

## 上传&下载文件
1. 文件上传支持单独上传和并发
2. 可以下载文件到本地

---

## 文件分段分类抓取

简历处理主流程：
1. 读取 Word 段落
2. 段落语义拆分
3. 自定义结构化JSON
4. 正则兜底邮箱,电话等;设定规则解析
5. 用NER解析补充缺失部分
6. 存储为FAISS
7. 返回最终结构化JSON

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

===== FINAL STRUCTURED RESUME JSON for Zhang.zhenyu6@northeastern.edu =====
2025-09-22 10:19:13,553 [INFO] {
  "name": "Zhenyu Zhang",
  "email": "Zhang.zhenyu6@northeastern.edu",
  "phone": "+1 (860) 234-7101 ",
  "education": [
    {
      "school": "Northeastern University",
      "degree": "Master of Professional Study in Applied Machine Intelligence",
      "grad_date": "2025",
      "description": "Northeastern University | Master of Professional Study in Applied Machine Intelligence | Boston, MA | April 2025 | GPA: 3.9/4.0"
    },
    {
      "school": "University of Connecticut",
      "degree": "Bachelor of Art",
      "grad_date": "May 2022",
      "description": "University of Connecticut | Bachelor of Art | Storrs, CT | May 2022 Skills"
    }
  ],
  "work_experience": [
    {
      "company": "Yangtze River Consulting Service LLC",
      "position": "Ethical Consultant",
      "location": "Piscataway Township, NJ",
      "start_date": "Sep 2022",
      "end_date": "Jul 2023",
      "description": "Yangtze River Consulting Service LLC | Ethical Consultant | Piscataway Township, NJ | Sep 2022 – Jul 2023",
      "highlights": [
        "Built a QA system with Transformer, improving automated question answering accuracy.",
        "Built and deployed app for NGO to auto-publish messages via AWS, boosting engagement 20%.",
        "Led COVID-19 time series and survey data analysis, focusing on cleaning, EDA, and modeling."
      ]
    },
    {
      "company": "New York Technology & Management LLC",
      "position": "Programming Manager",
      "location": "New York, NY",
      "start_date": "May 2025",
      "end_date": "Present",
      "description": "New York Technology & Management LLC | Programming Manager | New York, NY | May 2025 – Present",       
      "highlights": [
        "Built React agent with LangChain tool-calling to automate internal tool, reducing manual work.",
        "Created a token counter in Ollama to efficiently track and manage token usage.",
        "Built multi-agent system for sales and after-sales, streamlining communication and task flow.",
        "Used Ollama to analyze files and generate PowerPoint reports, improving reporting workflows.",
        "Built a LangChain agent to automate Gmail email tracking and improve communication.",
        "Fine-tuned and quantized Hugging Face model on custom data for optimized, faster inference."
      ]
    }
  ],
  "projects": [
    {
      "title": "Flower Market Assistant",
      "highlights": [
        "Built flower AI assistant to answer inquiries, boosting online sales and reducing workload.",
        "Built Ollama-based chatbot with memory, vector DB(FAISS), with LangChain and Flask.",
        "Increased online orders by 10% and reduced manual customer service workload by 40%."
      ]
    },
    {
      "title": "Crypto Predicted and Analysis with Vester AI",
      "highlights": [
        "Built a Crypto analysis tool with transformer model and LLM for Realtime market prediction.",
        "Collected Bitfinex API data, trained transformer model with historical data, added vector database.",
        "Using metal model to stack transformer model and sentiment model to get 84% accuracy."
      ]
    }
  ],
  "skills": [
    "Python",
    "Tableau",
    "SQL",
    "TensorFlow",
    "Pytorch",
    "Scikit-learn",
    "Pandas",
    "NumPy",
    "Seaborn",
    "Langchain"
  ],
  "other": []
}
---

## 注意事项
确保 MinIO 容器已运行
并发上传可提高效率
.env 管理密钥信息，不要硬编码在脚本中

## 更新目标
1. 升级: PDF兼容OCR, NER模型更换LLM.CPP或OPENAI (暂时不用)

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

# week 3
1. 拆分原upload_llm.py为doc, files, ner, parser, semantic, query, pipline, utils, 和 utils_parser多个模块. 拆分后代码不会太长, 同时按功能划分, 便于维护. 就是问题涉及多个模块时,难以确定问题所在. 
2. 优化:
    - 1. 在files添加save/load_embed, 可以缓存embeddings, 批量推理实现
    - 2. 在 doc 添加 read_pdf_paragraphs, 可以使用pdfplumber 支持PDF输入 直接处理PDF
    - 2. 1. 目前直接解析PDF文件效果不好,分类都是乱的, 同时, ner中的truncation也有问题, 即便是docx文件,删除data中的classified和faiss缓存, 运行pipline会报错不支持truncation为true, 删除truncation后,解析的分类就会变乱和PDF一样
    - 3. 添加db, 引入MONGODB数据库, 储存解析好的结构化JSON, pipline修改支持批量上传
    - 4. JSON字段命名不一致是main里写前台report是标题写错了
    - 5. eductio和workexperience部分的日期问题已修复, 在query和parser添加更多正则,可以正确输出月+年, 
    - 6. project已添加更多结构化
    - 6. 1. 目前分段过细, 按行分段, 标题之类太短被舍弃, 所以无法识别出 project的模块和project下的两个项目小模块. 尝试修改doc的semantic split, semantic的build faiss, 以及parser和query里的分类和结构化部分, 都是无效, 还是无法识别出项目标题    
    - 7. 在 utils增加SKILL_NORMALIZATION技能词典, 用于统一大小写和拼写格式
    - 8. 名字null问题 经测试发现utils的extract_basic_info可以抓取名字,但是修改parser和query的主要函数的基本信息抓取部分都无效,最终结构化json的名字还是null

# week 4
1. 优化解决了上周的所有错误:
  - 1. PDF问题: 现在删除了truncation参数, 在doc文件修改了文件读取逻辑,为docx添加donut, 为PDF添加LayoutLMv3(fallback为pdfplumber)
  - 2. project问题: 在utils文件添加了is_project_title,is_workline,parse_work_line...等等函数, 修改 parser解析文件, 可以识别并添加各工作经历的描述部分和各个项目的标题并添加描述
  - 3. 名字问题: 修改pipline文件, 可以在存储faiss后用解析后恢复原解析结构
  - 4. 其他优化: 解决以上问题之余的其他问题,如各个重要文件单独测试, 技能清理优化, 过滤条条件优化, 日期处理fallback问题等
  - 5. PDF的分割不太好,经常从中间分段,比如月+年,分成月一行,年一行,也许PDF2DOCX会更好

# week 5
1. 代码优化:
  - 1. 项目结构, 代码清理, 项目分割, 代码冗余... 
  - 2. lazy loading 和singleton,懒启动和单例模式,全局缓存,NER模型,语义模型,DonutDOCX分段模型都已添加
  - 3. pipline/main,可以一次处理多个文件,每个文件的处理过程都有显示
  - 4. 统一logging 日志, ELK:elastksearch,filebeat,logstash,kibana, docker部署容器,elk文件用filebeat和logstash设置kibana, 用init.ps1启动, 本地logs文件夹和kibana都有日志储存
  - 5. CI/CD pipline, GITHUB action: ci-cd.yml, CI部分: 忽略了代码风格问题(E402,501问题),忽略了外部数据问题(MongoDB问题),在test-tool的pipline_test添加test函数做测试, 成功返回 pytest-report CD部分: 测试了docker-compose, 成返回docker-build-log
