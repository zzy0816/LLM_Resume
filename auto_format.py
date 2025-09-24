import os
import subprocess

# 排除文件夹
EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", ".github"}


def run_command(cmd):
    """运行 shell 命令"""
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")


def main():
    py_files = []

    # 遍历所有文件
    for root, dirs, files in os.walk("."):
        # 排除指定文件夹
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    if not py_files:
        print("No Python files found.")
        return

    # 1️⃣ 用 isort 排序 import
    print("Running isort on Python files...")
    run_command(f"isort {' '.join(py_files)}")

    # 2️⃣ 用 black 格式化代码
    print("Running black to format code...")
    run_command(f"black {' '.join(py_files)}")

    print("All Python files have been formatted!")


if __name__ == "__main__":
    main()
