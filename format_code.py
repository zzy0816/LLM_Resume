import os
import subprocess

# 定义需要排除的文件夹
EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", ".github"}


def run_command(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")


def main():
    for root, dirs, files in os.walk("."):
        # 排除特定文件夹
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        py_files = [os.path.join(root, f) for f in files if f.endswith(".py")]
        if py_files:
            # 用 isort 排序 import
            run_command(f'isort {" ".join(py_files)}')
            # 用 black 格式化代码
            run_command(f'black {" ".join(py_files)}')


if __name__ == "__main__":
    main()
