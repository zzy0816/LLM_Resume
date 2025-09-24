import boto3
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, EndpointConnectionError
import logging, json, random, time, os

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "ner_service",
            "message": record.getMessage(),
            "request_id": str(random.randint(1000, 9999))
        }
        return json.dumps(log)

# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)

# 设置日志 handler
handler = logging.FileHandler("logs/app.log")
handler.setFormatter(JsonFormatter())

logger = logging.getLogger()  # root logger
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class StorageClient:
    def __init__(self):
        load_dotenv()
        self.endpoint = os.getenv("STORAGE_ENDPOINT")
        self.access_key = os.getenv("STORAGE_ACCESS_KEY")
        self.secret_key = os.getenv("STORAGE_SECRET_KEY")
        self.bucket = os.getenv("STORAGE_BUCKET")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint if self.endpoint else None,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        self.ensure_bucket()

    def ensure_bucket(self):
        """检查并创建存储桶"""
        try:
            existing_buckets = [b['Name'] for b in self.s3.list_buckets().get('Buckets', [])]
            if self.bucket not in existing_buckets:
                self.s3.create_bucket(Bucket=self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
        except Exception as e:
            logger.error(f"检查/创建 bucket 出错: {e}")

    def upload_file(self, file_path: str, object_name: str = None, retries: int = 3, delay: int = 2):
        """上传单个文件，带重试和分块支持"""
        object_name = object_name or os.path.basename(file_path)
        config = TransferConfig(multipart_threshold=50*1024*1024)  # 50MB 起用分块上传

        for attempt in range(1, retries + 1):
            try:
                self.s3.upload_file(Filename=file_path, Bucket=self.bucket, Key=object_name, Config=config)
                logger.info(f"Uploaded {file_path} as {object_name}")
                return True
            except (EndpointConnectionError, ClientError) as e:
                logger.warning(f"Attempt {attempt} failed for {file_path}: {e}")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Unhandled error uploading {file_path}: {e}")
                break
        logger.error(f"All {retries} attempts failed for {file_path}")
        return False

    def upload_files(self, file_paths: list, max_workers: int = 4):
        """并发上传多个文件，单文件失败不影响其他文件"""
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.upload_file, f): f for f in file_paths}
            for fut in futures:
                file = futures[fut]
                try:
                    results[file] = fut.result()
                except Exception as e:
                    logger.error(f"Unhandled error uploading {file}: {e}")
                    results[file] = False
        logger.info(f"Upload summary: {results}")
        return results

    def read_file(self, object_name: str, local_path: str = None, retries: int = 3, delay: int = 2) -> str:
        """读取文件并保存到本地，带重试和缓存"""
        local_path = local_path or object_name
        if os.path.exists(local_path):
            logger.info(f"Using cached file: {local_path}")
            return local_path

        for attempt in range(1, retries + 1):
            try:
                self.s3.download_file(Bucket=self.bucket, Key=object_name, Filename=local_path)
                logger.info(f"Downloaded {object_name} to {local_path}")
                return local_path
            except (EndpointConnectionError, ClientError) as e:
                logger.warning(f"Attempt {attempt} failed for {object_name}: {e}")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Unhandled error downloading {object_name}: {e}")
                break
        logger.error(f"All {retries} attempts failed for {object_name}")
        return None


# =============================
# ✅ 测试入口
# =============================
if __name__ == "__main__":
    client = StorageClient()

    files_to_upload = [
        r"D:\project\LLM_Resume\data\Resume(AI).docx",
        r"D:\project\LLM_Resume\data\Resume(AI).pdf"
    ]

    # 上传多个文件
    client.upload_files(files_to_upload)

    # 读取其中一个文件
    client.read_file("Resume(AI).docx", r"D:\project\LLM_Resume\downloads\Resume(AI).pdf")
