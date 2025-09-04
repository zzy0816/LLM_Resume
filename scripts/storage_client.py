import boto3
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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

    def upload_file(self, file_path: str, object_name: str = None):
        """上传单个文件"""
        try:
            if object_name is None:
                object_name = os.path.basename(file_path)
            self.s3.upload_file(Filename=file_path, Bucket=self.bucket, Key=object_name)
            logger.info(f"Uploaded {file_path} as {object_name}")
        except Exception as e:
            logger.error(f"上传文件失败 {file_path}: {e}")

    def upload_files(self, file_paths: list, max_workers: int = 4):
        """并发上传多个文件"""
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(self.upload_file, file_paths)
            logger.info("All files uploaded successfully")
        except Exception as e:
            logger.error(f"批量上传失败: {e}")

    def read_file(self, object_name: str, local_path: str = None) -> str:
        """读取文件并保存到本地"""
        try:
            if local_path is None:
                local_path = object_name
            self.s3.download_file(Bucket=self.bucket, Key=object_name, Filename=local_path)
            logger.info(f"Downloaded {object_name} to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"下载文件失败 {object_name}: {e}")
            return None

# =============================
# ✅ 测试入口
# =============================
if __name__ == "__main__":
    client = StorageClient()

    files_to_upload = [
        r"D:\project\LLM_Resume\data\Resume(AI).docx",
        r"D:\project\LLM_Resume\data\Resume(DS)v0.1.docx"
    ]

    # 上传多个文件
    client.upload_files(files_to_upload)

    # 读取其中一个文件
    client.read_file("Resume(AI).docx", r"D:\project\LLM_Resume\downloads\Resume(AI).docx")
