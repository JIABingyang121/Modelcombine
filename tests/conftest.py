import sys
from pathlib import Path

# 让测试能 import src.*（仓库根加入 sys.path）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
