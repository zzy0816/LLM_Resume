from concurrent.futures import ThreadPoolExecutor
import boto3
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从环境变量读取配置
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

# 初始化 S3 客户端
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

# 创建桶（如果不存在）
existing_buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]
if MINIO_BUCKET not in existing_buckets:
    s3.create_bucket(Bucket=MINIO_BUCKET)

# 文件列表
files_to_upload = [
    r"D:\project\LLM_Resume\data\Resume(AI).docx",
    r"D:\project\LLM_Resume\data\Resume(DS)v0.1.docx"
]

def upload_file(file_path):
    object_name = os.path.basename(file_path)
    s3.upload_file(Filename=file_path, Bucket=MINIO_BUCKET, Key=object_name)
    print(f"Uploaded {object_name}")

# 并发上传
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(upload_file, files_to_upload)

print("All files uploaded successfully.")
