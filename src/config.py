"""配置管理模块 — 通过 Streamlit secrets 和环境变量管理 API Key 等配置."""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录（ChromaDB 持久化路径）
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ChromaDB 集合名称
COLLECTION_NAME = "knowledge_base"

# LLM 配置 — 从环境变量读取，Streamlit Cloud 用 st.secrets
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 嵌入模型配置
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 分块配置
CHUNK_SIZE = 500       # 每块字符数
CHUNK_OVERLAP = 50     # 块间重叠字符数

# 检索配置
TOP_K = 4              # 每次检索返回的文档片段数

# 文件大小上限 (字节)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# 支持的文件后缀
SUPPORTED_SUFFIXES = [".pdf", ".docx", ".txt", ".md"]
