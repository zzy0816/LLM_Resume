import os
import re
import subprocess
import sys
import textwrap

# 排除文件夹
EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", ".github"}
MAX_LINE_LENGTH = 79


def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARNING] Command finished with non-zero exit code: {cmd}")


def find_py_files():
    py_files = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.normpath(os.path.join(root, f)))
    return py_files


def check_test_dir():
    if not os.path.exists("test_tool") or not os.path.isdir("test_tool"):
        print(
            "[WARNING] test_tool directory not found! Please create or push it."
        )


def wrap_long_strings_and_literals(file_path):
    """折行长字符串、注释以及极长字面量"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    for line in lines:
        # 去掉尾部换行方便处理
        clean_line = line.rstrip("\n")
        if len(clean_line) > MAX_LINE_LENGTH:
            # 尝试折行字符串或注释
            if clean_line.strip().startswith("#") or (
                '"' in clean_line or "'" in clean_line
            ):
                wrapped = textwrap.fill(
                    clean_line,
                    width=MAX_LINE_LENGTH,
                    break_long_words=True,
                    replace_whitespace=False,
                )
                new_lines.extend([l + "\n" for l in wrapped.split("\n")])
                modified = True
            else:
                # 对复杂字面量或极长表达式，简单切分
                chunks = [
                    clean_line[i : i + MAX_LINE_LENGTH]
                    for i in range(0, len(clean_line), MAX_LINE_LENGTH)
                ]
                new_lines.extend([c + "\n" for c in chunks])
                modified = True
        else:
            new_lines.append(line)

    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"[INFO] Wrapped long lines in {file_path}")


def main():
    check_test_dir()
    py_files = find_py_files()
    if not py_files:
        print("No Python files found.")
        return

    # 1️⃣ isort 排序 import
    print("Running isort on Python files...")
    run_command(f"{sys.executable} -m isort {' '.join(py_files)}")

    # 2️⃣ black 格式化代码
    print("Running black to format code...")
    run_command(
        f"{sys.executable} -m black --line-length {MAX_LINE_LENGTH} {' '.join(py_files)}"
    )

    # 3️⃣ 折行剩余极长字符串、注释、字面量
    print("Wrapping remaining long lines...")
    for file in py_files:
        wrap_long_strings_and_literals(file)

    # 4️⃣ flake8 检查风格
    print("Running flake8 to check code style...")
    for file in py_files:
        run_command(f"{sys.executable} -m flake8 {file}")

    print("All Python files have been auto-formatted and checked!")


if __name__ == "__main__":
    main()
