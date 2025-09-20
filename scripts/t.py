import logging
import json
from files import save_json
from doc import read_document_paragraphs
from parser_test import parse_resume_to_structured 
from utils import auto_fill_fields, extract_basic_info, validate_and_clean
from db import save_resume

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def sanitize_filename(file_name: str) -> str:
    return file_name.replace("(", "_").replace(")", "_").replace(" ", "_")

def make_safe_for_mongo(obj):
    """
    将对象转换为 MongoDB/JSON 可插入/序列化格式，同时避免循环引用。
    - dict/list/tuple/set 递归处理；
    - 基本类型( str/int/float/bool/None ) 保留；
    - 其他类型转为 str；
    - 遇到已访问的容器（循环引用）返回 "[Circular]"。
    """
    seen = set()

    def _sanitize(o):
        # 不对不可哈希的类型做 seen 判断前先检测类型（只有容器类型需要循环检测）
        if isinstance(o, dict):
            oid = id(o)
            if oid in seen:
                return "[Circular]"
            seen.add(oid)
            new = {}
            for k, v in o.items():
                # JSON keys 必须是字符串
                new_key = str(k)
                new[new_key] = _sanitize(v)
            return new

        if isinstance(o, (list, tuple, set)):
            oid = id(o)
            if oid in seen:
                return "[Circular]"
            seen.add(oid)
            return [_sanitize(v) for v in o]

        # 基本可序列化类型
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o

        # 其他对象（例如自定义类实例、langchain 文档对象等）转换为字符串
        try:
            return str(o)
        except Exception:
            # 最后保底
            try:
                return repr(o)
            except Exception:
                return "[Unserializable]"

    return _sanitize(obj)

def main_pipeline(files_to_process: list[str]) -> dict[str, dict]:
    results = {}

    for file_name in files_to_process:
        file_path = f"./downloads/{file_name}"
        logger.info(f"Processing file {file_name}")
        safe_name = sanitize_filename(file_name)

        # 1️⃣ 读取文档并解析
        paragraphs = read_document_paragraphs(file_path)
        full_text = "\n".join(paragraphs)
        basic_info = extract_basic_info(full_text)

        parsed_resume = parse_resume_to_structured(paragraphs) or {}
        # 打印 parsed_resume（可能已经序列化友好），若过大可注释
        logger.info("Parsed resume:")
        try:
            logger.info(json.dumps(parsed_resume, ensure_ascii=False, indent=2))
        except Exception:
            logger.info(json.dumps(make_safe_for_mongo(parsed_resume), ensure_ascii=False, indent=2))

        # 2️⃣ 自动填充和合并基本信息
        structured_resume = auto_fill_fields(parsed_resume) or {}
        structured_resume["name"] = structured_resume.get("name") or basic_info.get("name")
        structured_resume["email"] = structured_resume.get("email") or basic_info.get("email")
        structured_resume["phone"] = structured_resume.get("phone") or basic_info.get("phone")

        # 3️⃣ 清理和验证
        logger.info(f"Other after auto_fill_fields: {structured_resume.get('other')}")
        structured_resume = validate_and_clean(structured_resume) or {}
        logger.info(f"Other after validate_and_clean: {structured_resume.get('other')}")


        # 4️⃣ 转换为 MongoDB 可插入格式，避免循环引用
        safe_resume_for_db = make_safe_for_mongo(structured_resume)

        # 5️⃣ 保存到数据库（传入安全化的对象）
        user_email = structured_resume.get("email") or safe_name
        try:
            save_resume(user_id=user_email, file_name=safe_name + "_parsed", data=safe_resume_for_db)
            logger.info(f"Saved resume to MongoDB: {safe_name}_parsed")
        except Exception as e:
            logger.warning(f"Saving to MongoDB failed: {e}")

        # 6️⃣ 保存 JSON，也使用安全化对象，避免循环引用
        try:
            save_json(safe_name + "_parsed", safe_resume_for_db)
            logger.info(f"Saved JSON: {safe_name}_parsed.json")
        except Exception as e:
            logger.warning(f"Saving JSON failed: {e}")
            # 备用：本地直接写文件
            try:
                fallback_path = f"./data/{safe_name}_parsed_fallback.json"
                with open(fallback_path, "w", encoding="utf-8") as f:
                    json.dump(safe_resume_for_db, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved fallback JSON to {fallback_path}")
            except Exception as e2:
                logger.error(f"Fallback JSON save also failed: {e2}")

        # 7️⃣ 返回值里放安全化后的对象，避免后续打印再出错
        results[user_email] = safe_resume_for_db

    return results

if __name__ == "__main__":
    files_to_process = ["Resume(AI).docx"]
    all_results = main_pipeline(files_to_process)
    for user_email, safe_structured_resume in all_results.items():
        logger.info(f"\n===== FINAL STRUCTURED RESUME JSON for {user_email} =====")
        # 这里直接打印已经安全化的结果，避免循环引用
        logger.info(json.dumps(safe_structured_resume, ensure_ascii=False, indent=2))
