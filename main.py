from fastapi import FastAPI, UploadFile, File
from concurrent.futures import ThreadPoolExecutor
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# MinIO/S3 配置
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name=os.getenv("MINIO_REGION", None)
)
bucket_name = os.getenv("MINIO_BUCKET")

# 确保桶存在
existing_buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]
if bucket_name not in existing_buckets:
    s3.create_bucket(Bucket=bucket_name)

# 上传单个文件
def upload_to_s3(file: UploadFile):
    s3.upload_fileobj(file.file, bucket_name, file.filename)
    print(f"Uploaded {file.filename}")

# 上传接口
@app.post("/upload_resumes/")
async def upload_resumes(files: list[UploadFile] = File(...)):
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(upload_to_s3, files)
    return {"status": "uploaded", "files": [f.filename for f in files]}

# 读取 + 分析接口
@app.get("/analyze_resume/")
async def analyze_resume(filename: str):
    obj = s3.get_object(Bucket=bucket_name, Key=filename)
    content = obj['Body'].read()
    text = content.decode("utf-8", errors="ignore")  # 文本解析示例

    # 模拟 LLM 分析
    analysis_result = f"分析结果（示例）：文件 {filename} 字符长度 {len(text)}"
    return {"filename": filename, "analysis": analysis_result}
