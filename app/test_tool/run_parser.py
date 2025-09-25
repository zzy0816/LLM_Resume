import json
import logging
import os
import random
import re
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from app.qre.doc_read import read_document_paragraphs
from app.storage.db import save_resume
from app.test_tool.parser_test import parse_resume_to_structured
from app.utils.files import save_json
from app.utils.utils import (
    auto_fill_fields,
    extract_basic_info,
    validate_and_clean,
    setup_logging
)

setup_logging()
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


def fix_resume_dates(structured_resume: dict) -> dict:
    """
    修复教育和工作经历日期：
    1. 教育经历：优先使用 description 中 Month Year，否则用年份。
    2. 工作经历：优先使用 description 中 Month Year，end_date 会根据 highlights 推断，保证 end_date >= start_date。
    3. highlights 中纯年份或 "Present" 会被清理。
    """
    if not structured_resume:
        return {}

    # ---- 教育经历 ----
    month_year_pattern = (
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(19|20)\d{2}"
    )
    year_pattern = r"(19|20)\d{2}"

    for edu in structured_resume.get("education", []):
        desc = edu.get("description", "")
        desc_clean = re.sub(r"[\r\n]+", " ", desc)
        desc_clean = re.sub(r"\s+", " ", desc_clean).strip()

        match = re.search(month_year_pattern, desc_clean)
        if match:
            edu["grad_date"] = match.group(0)
        else:
            match_year = re.search(year_pattern, desc_clean)
            if match_year:
                edu["grad_date"] = match_year.group(0)

    # ---- 工作经历 ----
    work_exp = structured_resume.get("work_experience", [])
    structured_resume["work_experience"] = fix_work_dates(work_exp)

    return structured_resume


def fix_work_dates(work_experience: list) -> list:
    month_pattern = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    year_pattern = r"(19|20)\d{2}"

    for job in work_experience:
        desc = job.get("description", "")
        highlights = job.get("highlights", [])

        # 合并换行并压缩空格
        desc_clean = re.sub(r"[\r\n]+", " ", desc)
        desc_clean = re.sub(r"\s+", " ", desc_clean).strip()

        # 收集 highlights 年份
        highlight_years = []
        cleaned_highlights = []
        for h in highlights:
            h_strip = h.strip()
            if re.fullmatch(r"\d{4}", h_strip):
                highlight_years.append(int(h_strip))
            else:
                cleaned_highlights.append(h)
                highlight_years.extend(
                    [int(y) for y in re.findall(year_pattern, h_strip)]
                )
        job["highlights"] = cleaned_highlights

        # 提取 description 中所有 Month Year
        month_year_matches = re.findall(
            rf"{month_pattern}\s+{year_pattern}", desc_clean
        )
        # 提取 description 中只有 Month 的部分
        months_only_matches = re.findall(
            rf"{month_pattern}(?=\s*(–|$))", desc_clean
        )

        # --- start_date ---
        start_date = job.get("start_date")
        if not start_date or str(start_date).lower() in ["null", "n/a", ""]:
            start_date = (
                month_year_matches[0]
                if month_year_matches
                else f"{months_only_matches[0] if months_only_matches else 'Jan'} 2020"
            )

        # --- end_date ---
        end_date = job.get("end_date")
        if not end_date or str(end_date).lower() in ["null", "n/a", ""]:
            if len(month_year_matches) >= 2:
                end_date = month_year_matches[1]
            elif months_only_matches:
                # Month-only，年份 = start_date 年 +1
                start_m, start_y = start_date.split()
                start_y = int(start_y)
                # 找到 end 月份，如果只有一个 month-only，用 start month +1 年
                end_m = (
                    months_only_matches[1]
                    if len(months_only_matches) > 1
                    else months_only_matches[0]
                )
                end_date = f"{end_m} {start_y + 1}"
            elif highlight_years:
                end_date = f"{start_date.split()[0]} {max(highlight_years)}"
            else:
                end_date = "Present"

        # --- 确保 end_date >= start_date ---
        try:
            start_m, start_y = start_date.split()
            start_y = int(start_y)
            if end_date.lower() != "present":
                end_m, end_y = end_date.split()
                end_y = int(end_y)
                if end_y < start_y:
                    end_y = start_y + 1
                    end_date = f"{end_m} {end_y}"
        except Exception:
            pass

        job["start_date"] = start_date
        job["end_date"] = end_date

    return work_experience


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
            logger.info(
                json.dumps(parsed_resume, ensure_ascii=False, indent=2)
            )
        except Exception:
            logger.info(
                json.dumps(
                    make_safe_for_mongo(parsed_resume),
                    ensure_ascii=False,
                    indent=2,
                )
            )

        # 2️⃣ 自动填充和合并基本信息
        structured_resume = auto_fill_fields(parsed_resume) or {}
        structured_resume["name"] = structured_resume.get(
            "name"
        ) or basic_info.get("name")
        structured_resume["email"] = structured_resume.get(
            "email"
        ) or basic_info.get("email")
        structured_resume["phone"] = structured_resume.get(
            "phone"
        ) or basic_info.get("phone")

        # 3️⃣ 清理和验证
        logger.info(
            f"Other after auto_fill_fields: {structured_resume.get('other')}"
        )
        structured_resume = validate_and_clean(structured_resume) or {}
        logger.info(
            f"Other after validate_and_clean: {structured_resume.get('other')}"
        )

        structured_resume = fix_resume_dates(structured_resume)

        # 4️⃣ 转换为 MongoDB 可插入格式，避免循环引用
        safe_resume_for_db = make_safe_for_mongo(structured_resume)

        # 5️⃣ 保存到数据库（传入安全化的对象）
        user_email = structured_resume.get("email") or safe_name
        try:
            save_resume(
                user_id=user_email,
                file_name=safe_name + "_parsed",
                data=safe_resume_for_db,
            )
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
                    json.dump(
                        safe_resume_for_db, f, ensure_ascii=False, indent=2
                    )
                logger.info(f"Saved fallback JSON to {fallback_path}")
            except Exception as e2:
                logger.error(f"Fallback JSON save also failed: {e2}")

        # 7️⃣ 返回值里放安全化后的对象，避免后续打印再出错
        results[user_email] = safe_resume_for_db

    return results


if __name__ == "__main__":
    files_to_process = ["Resume(AI).pdf"]
    all_results = main_pipeline(files_to_process)
    for user_email, safe_structured_resume in all_results.items():
        logger.info(
            f"\n===== FINAL STRUCTURED RESUME JSON for {user_email} ====="
        )
        # 这里直接打印已经安全化的结果，避免循环引用
        logger.info(
            json.dumps(safe_structured_resume, ensure_ascii=False, indent=2)
        )
