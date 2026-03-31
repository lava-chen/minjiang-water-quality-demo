"""启动器：解决 PowerShell 中文路径编码问题"""
import os, sys, runpy

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
os.chdir(root)

# 设置环境变量，让子脚本知道项目根目录
os.environ["PROJECT_ROOT"] = root

runpy.run_path(
    os.path.join(root, "src", "model", "train.py"),
    run_name="__main__",
)
