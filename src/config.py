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

# 嵌入模型配置 — 优先使用本地下载的模型，避免连接 HuggingFace（国内超时）
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
_LOCAL_MODEL_DIR = Path.home() / ".cache" / "modelscope" / "bge-m3-tmp" / "BAAI" / "bge-m3"
LOCAL_EMBEDDING_PATH = str(_LOCAL_MODEL_DIR) if _LOCAL_MODEL_DIR.exists() else None

# 分块配置
CHUNK_SIZE = 1000      # 每块字符数
CHUNK_OVERLAP = 100    # 块间重叠字符数

# 检索配置
TOP_K = 4              # 每次检索返回的文档片段数

# 文件大小上限 (字节)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# 支持的文件后缀
SUPPORTED_SUFFIXES = [".pdf", ".docx", ".txt", ".md"]

# === API Key 本地持久化 ===
_API_KEY_FILE = PROJECT_ROOT / ".streamlit" / "saved_key"


def save_api_key(key: str) -> None:
    """将 API Key 保存到本地文件."""
    _API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _API_KEY_FILE.write_text(key.strip())


def load_api_key() -> str:
    """从本地文件加载已保存的 API Key，不存在则返回空字符串."""
    if _API_KEY_FILE.exists():
        return _API_KEY_FILE.read_text().strip()
    return ""


# === 知识库管理 ===
DEFAULT_KB_NAME = "默认知识库"
_KB_NAMES_FILE = DATA_DIR / "kb_names.json"


def kb_to_collection(display_name: str) -> str:
    """将中文 KB 名转为 ChromaDB 合法的集合名（仅 [a-zA-Z0-9._-]）."""
    import hashlib
    # 默认知识库 → 兼容旧的 knowledge_base 集合
    if display_name == DEFAULT_KB_NAME:
        return COLLECTION_NAME
    # 其它名称：确定性 hash，保证同名 KB 始终映射到同一集合
    h = hashlib.sha256(display_name.encode("utf-8")).hexdigest()[:12]
    return f"kb_{h}"


def load_kb_names() -> list:
    """加载所有知识库名称列表."""
    import json
    if _KB_NAMES_FILE.exists():
        try:
            return json.loads(_KB_NAMES_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return [DEFAULT_KB_NAME]
    return [DEFAULT_KB_NAME]


def save_kb_names(names: list) -> None:
    """保存知识库名称列表."""
    import json
    _KB_NAMES_FILE.write_text(
        json.dumps(names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# === 对话记录持久化（按知识库分文件） ===
import re


def _safe_filename(name: str) -> str:
    """将知识库名称转为安全的文件名."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def _chat_history_path(kb_name: str) -> Path:
    return DATA_DIR / f"chat_history_{_safe_filename(kb_name)}.json"


def save_chat_history(messages: list, kb_name: str = DEFAULT_KB_NAME) -> None:
    """将指定知识库的聊天记录保存到本地 JSON 文件."""
    import json
    _chat_history_path(kb_name).write_text(
        json.dumps(messages, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_chat_history(kb_name: str = DEFAULT_KB_NAME) -> list:
    """从本地文件加载指定知识库的聊天记录."""
    import json
    path = _chat_history_path(kb_name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []
