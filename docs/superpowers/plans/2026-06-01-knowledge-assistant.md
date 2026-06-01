# Knowledge Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG-powered knowledge base Q&A system where users upload documents, ask questions in natural language, and get AI-generated answers with source citations.

**Architecture:** Streamlit monolith with modular `src/` package. File parsing → text chunking → BGE-M3 embeddings → ChromaDB storage. Query flow: question → embed → ChromaDB search → DeepSeek LLM with retrieved context → answer + sources. Single-page Streamlit app with sidebar for file management and main area for chat.

**Tech Stack:** Python 3.11+, Streamlit, DeepSeek API (OpenAI-compatible), BGE-M3 via sentence-transformers, ChromaDB, PyMuPDF, python-docx

---

## File Structure (Final)

```
knowledge-assistant/
├── app.py                 # Streamlit 主入口
├── requirements.txt       # 依赖清单
├── src/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── file_parser.py     # 多格式文件解析
│   ├── text_splitter.py   # 文本智能分块
│   ├── embedder.py        # BGE-M3 嵌入
│   ├── vector_store.py    # ChromaDB 操作
│   ├── llm_client.py      # LLM API 客户端
│   ├── rag_engine.py      # RAG 核心引擎
│   └── ui_components.py   # Streamlit UI 组件
├── tests/
│   ├── __init__.py
│   ├── test_file_parser.py
│   ├── test_text_splitter.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_llm_client.py
│   └── test_rag_engine.py
└── docs/
    ├── design.md
    └── superpowers/
        └── plans/
            └── 2026-06-01-knowledge-assistant.md
```

---

### Task 1: 项目初始化

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`

- [ ] **Step 1: 编写 requirements.txt**

```txt
streamlit==1.37.1
chromadb==0.5.23
sentence-transformers==3.2.1
pymupdf==1.25.2
python-docx==1.1.2
openai==1.57.4
```

- [ ] **Step 2: 安装依赖**

Run: `pip install -r requirements.txt`
Expected: 全部安装成功，无报错

- [ ] **Step 3: 编写 src/__init__.py**

```python
"""Knowledge Assistant - RAG-powered document Q&A system."""
__version__ = "0.1.0"
```

- [ ] **Step 4: 编写 src/config.py**

```python
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
```

- [ ] **Step 5: 编写 tests/__init__.py**

```python
"""Tests for Knowledge Assistant."""
```

- [ ] **Step 6: 提交**

```bash
git add requirements.txt src/__init__.py src/config.py tests/__init__.py
git commit -m "chore: project init — deps, config, structure"
```

---

### Task 2: 文件解析模块

**Files:**
- Create: `src/file_parser.py`
- Create: `tests/test_file_parser.py`

- [ ] **Step 1: 编写测试**

```python
"""Tests for file_parser.py"""
import pytest
from pathlib import Path
from src.file_parser import FileParser, FileParseError, UnsupportedFormatError

FIXTURES = Path(__file__).parent / "fixtures"

class TestFileParser:
    def test_parse_txt_returns_text(self, tmp_path):
        """解析 TXT 文件应返回文本内容"""
        f = tmp_path / "test.txt"
        f.write_text("你好，这是一个测试文档。\n第二行内容。", encoding="utf-8")
        result = FileParser.parse(f)
        assert "测试文档" in result
        assert "第二行" in result

    def test_parse_unsupported_extension_raises(self, tmp_path):
        """不支持的扩展名应抛出 UnsupportedFormatError"""
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(UnsupportedFormatError) as exc:
            FileParser.parse(f)
        assert "xyz" in str(exc.value)

    def test_file_too_large_raises(self, tmp_path):
        """超过 50MB 文件应抛出 FileParseError"""
        f = tmp_path / "large.txt"
        # 创建一个假的超大文件路径（不写实际内容，通过 mock size）
        f.write_text("small")
        from unittest.mock import patch
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 51 * 1024 * 1024
            with pytest.raises(FileParseError) as exc:
                FileParser.parse(f)
            assert "50MB" in str(exc.value)

    def test_txt_empty_file_returns_empty_string(self, tmp_path):
        """空文件应返回空字符串"""
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = FileParser.parse(f)
        assert result == ""

    def test_parse_pdf_with_text(self):
        """解析含文字的 PDF 应提取出文本"""
        # Create a simple PDF with text for testing
        import fitz
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "Hello PDF 测试内容")
        pdf_path = FIXTURES / "test_text.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(pdf_path))
        doc.close()
        result = FileParser.parse(pdf_path)
        assert "测试内容" in result

    def test_pdf_no_text_detected_raises(self, tmp_path):
        """扫描版 PDF（无文字层）应抛出 FileParseError"""
        import fitz
        doc = fitz.open()
        # Add a page with only image, no text
        page = doc.new_page()
        doc.save(str(tmp_path / "scan.pdf"))
        doc.close()
        with pytest.raises(FileParseError) as exc:
            FileParser.parse(tmp_path / "scan.pdf")
        assert "OCR" in str(exc.value) or "文字" in str(exc.value)

    def test_parse_docx(self, tmp_path):
        """解析 DOCX 文件"""
        from docx import Document
        doc = Document()
        doc.add_paragraph("这是Word文档内容。")
        doc.add_paragraph("第二个段落。")
        path = tmp_path / "test.docx"
        doc.save(str(path))
        result = FileParser.parse(path)
        assert "Word文档内容" in result
        assert "第二个段落" in result

    def test_compute_md5(self, tmp_path):
        """计算文件 MD5 指纹"""
        f = tmp_path / "md5test.txt"
        f.write_text("hello world", encoding="utf-8")
        md5 = FileParser.compute_md5(f)
        import hashlib
        expected = hashlib.md5(b"hello world").hexdigest()
        assert md5 == expected
        assert len(md5) == 32  # MD5 总是 32 位十六进制
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_file_parser.py -v`
Expected: 全部 FAIL — FileParser 尚未实现

- [ ] **Step 3: 编写实现**

```python
"""文件解析模块 — 支持 PDF/Word/TXT/Markdown 的文本提取."""
import hashlib
from pathlib import Path
from src.config import MAX_FILE_SIZE, SUPPORTED_SUFFIXES


class UnsupportedFormatError(Exception):
    """不支持的文���格式."""

class FileParseError(Exception):
    """文件解析失败."""


class FileParser:
    """多格式文件解析器."""

    # 文件类型 -> 后缀列表映射
    SUFFIX_MAP = {
        "pdf": [".pdf"],
        "docx": [".docx"],
        "text": [".txt", ".md"],
    }

    @classmethod
    def parse(cls, file_path: Path) -> str:
        """解析文件，统一返回文本字符串."""
        file_path = Path(file_path)

        # 检查文件大小
        if file_path.stat().st_size > MAX_FILE_SIZE:
            raise FileParseError(f"文件过大：{file_path.name} 超过 50MB 上限，请拆分后上传")

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise UnsupportedFormatError(
                f"不支持的文件格式：{suffix}。"
                f"支持：{', '.join(SUPPORTED_SUFFIXES)}"
            )

        if suffix == ".pdf":
            return cls._parse_pdf(file_path)
        elif suffix == ".docx":
            return cls._parse_docx(file_path)
        else:
            return cls._parse_text(file_path)

    @classmethod
    def _parse_pdf(cls, file_path: Path) -> str:
        """解析 PDF 文件."""
        import fitz  # PyMuPDF
        doc = fitz.open(str(file_path))
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[第{page_num}页]\n{text}")
        doc.close()

        if not text_parts:
            raise FileParseError(
                f"该 PDF ({file_path.name}) 为扫描图片，无文字层。请先用 OCR 工具（如 WPS）转换后上传。"
            )
        return "\n\n".join(text_parts)

    @classmethod
    def _parse_docx(cls, file_path: Path) -> str:
        """解析 DOCX 文件."""
        from docx import Document
        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            raise FileParseError(
                f"Word 文档 ({file_path.name}) 中未找到文字内容。"
            )
        return "\n\n".join(paragraphs)

    @classmethod
    def _parse_text(cls, file_path: Path) -> str:
        """解析纯文本文件."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    @staticmethod
    def compute_md5(file_path: Path) -> str:
        """计算文件 MD5 指纹（用于去重）."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_file_parser.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/file_parser.py tests/test_file_parser.py
git commit -m "feat: file parser — PDF/Word/TXT/MD text extraction with MD5 dedup"
```

---

### Task 3: 文本分块模块

**Files:**
- Create: `src/text_splitter.py`
- Create: `tests/test_text_splitter.py`

- [ ] **Step 1: 编写测试**

```python
"""Tests for text_splitter.py"""
from src.text_splitter import TextSplitter


class TestTextSplitter:
    def test_split_chunks_under_max_size(self):
        """每块字符数不超过 chunk_size"""
        text = "测试内容。" * 100
        chunks = TextSplitter.split(text, chunk_size=500, overlap=50)
        for c in chunks:
            assert len(c) <= 500

    def test_overlap_between_chunks(self):
        """相邻块之间应有重叠"""
        text = "0123456789" * 50  # 500 chars
        chunks = TextSplitter.split(text, chunk_size=200, overlap=50)
        # Check that chunk N's tail overlaps with chunk N+1's head
        for i in range(len(chunks) - 1):
            tail = chunks[i][-50:]
            head = chunks[i + 1][:50]
            assert tail == head, f"Chunk {i} tail != Chunk {i+1} head"

    def test_short_text_returns_single_chunk(self):
        """短于 chunk_size 的文本返回单块"""
        text = "你好世界"
        chunks = TextSplitter.split(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "你好世界"

    def test_empty_text_returns_empty_list(self):
        """空文本返回空列表"""
        chunks = TextSplitter.split("", chunk_size=500, overlap=50)
        assert chunks == []

    def test_paragraph_boundary_split(self):
        """在段落边界处分块（用 \\n\\n 分割）"""
        paragraphs = ["段落一。" * 80, "段落二。" * 80, "段落三。" * 80]
        text = "\n\n".join(paragraphs)
        chunks = TextSplitter.split(text, chunk_size=300, overlap=50)
        assert len(chunks) >= 3  # 至少按三个段落分开
        for i, chunk in enumerate(chunks):
            assert chunk, f"Chunk {i} should not be empty"

    def test_chunks_with_metadata(self):
        """带元数据的分块应保留文件名和页码信息"""
        text = "第一页内容。" * 100
        chunks = TextSplitter.split_with_metadata(
            text, "test.pdf", 1, chunk_size=500, overlap=50
        )
        for c in chunks:
            assert c["file"] == "test.pdf"
            assert c["page"] == 1
            assert "content" in c
            assert len(c["content"]) <= 500
```

- [ ] **Step 2: 确认测试失败**

Run: `pytest tests/test_text_splitter.py -v`
Expected: 全部 FAIL

- [ ] **Step 3: 编写实现**

```python
"""文本智能分块模块 — 段落级分段 + 滑动窗口重叠."""
from typing import List, Dict, Any
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextSplitter:
    """文本分块器，优先在段落边界切分."""

    @classmethod
    def split(
        cls,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[str]:
        """将文本切分为重叠的块."""
        if not text or not text.strip():
            return []

        # 先按段落分割
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # 单个段落太长则按 chunk_size 切分
            if len(para) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                para_chunks = cls._split_long_text(para, chunk_size, overlap)
                chunks.extend(para_chunks)
            elif len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks

    @classmethod
    def _split_long_text(cls, text: str, chunk_size: int, overlap: int) -> List[str]:
        """对超长文本进行滑动窗口切割."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += (chunk_size - overlap)
        return chunks

    @classmethod
    def split_with_metadata(
        cls,
        text: str,
        file_name: str,
        page_num: int,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> List[Dict[str, Any]]:
        """切分文本并附加文件来源元数据."""
        text_chunks = cls.split(text, chunk_size, overlap)
        return [
            {
                "content": chunk,
                "file": file_name,
                "page": page_num,
                "chunk_id": f"{file_name}_p{page_num}_{i}",
            }
            for i, chunk in enumerate(text_chunks)
        ]
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_text_splitter.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/text_splitter.py tests/test_text_splitter.py
git commit -m "feat: text splitter — paragraph-aware chunking with overlap"
```

---

### Task 4: 嵌入模型模块

**Files:**
- Create: `src/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: 编写测试**

```python
"""Tests for embedder.py"""
import pytest
import numpy as np
from src.embedder import Embedder


class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        """全测试共用 embedder，避免重复加载模型."""
        return Embedder()

    def test_embed_single_text_returns_ndarray(self, embedder):
        """嵌入单条文本应返回 numpy 数组"""
        vec = embedder.embed("你好世界")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1  # 一维向量
        assert vec.shape[0] == 1024  # BGE-M3 输出 1024 维

    def test_embed_batch_returns_2d_array(self, embedder):
        """批量嵌入应返回二维数组"""
        texts = ["第一段", "第二段", "第三段"]
        vecs = embedder.embed_batch(texts)
        assert isinstance(vecs, np.ndarray)
        assert vecs.ndim == 2
        assert vecs.shape == (3, 1024)

    def test_semantic_similarity(self, embedder):
        """语义相近的句子向量余弦相似度应较高"""
        v1 = embedder.embed("今天天气真好")
        v2 = embedder.embed("天气很不错呢")
        v3 = embedder.embed("Python 编程语言的学习")

        sim_12 = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sim_13 = float(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)))

        assert sim_12 > sim_13, (
            f"相似语义应该有更高相似度: sim12={sim_12:.3f}, sim13={sim_13:.3f}"
        )

    def test_empty_text_raises(self, embedder):
        """空字符串应抛出 ValueError"""
        with pytest.raises(ValueError):
            embedder.embed("")

    def test_embed_query_returns_ndarray(self, embedder):
        """查询嵌入（embed_query）应返回一维向量"""
        vec = embedder.embed_query("如何学习深度学习？")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] == 1024
```

- [ ] **Step 2: 确认测试失败**

Run: `pytest tests/test_embedder.py -v`
Expected: 全部 FAIL

- [ ] **Step 3: 编写实现**

```python
"""嵌入模型模块 — BGE-M3 本地嵌入，文本转向量."""
import numpy as np
from typing import List
from src.config import EMBEDDING_MODEL_NAME


class Embedder:
    """文本嵌入器，使用 BGE-M3 模型."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """嵌入向量维度."""
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """将单条文本转为向量."""
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入，返回 shape=(N, dim) 的数组."""
        if not texts:
            raise ValueError("文本列表不能为空")
        return self._model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """嵌入查询文本（BGE-M3 对查询使用相同的编码方式）."""
        return self.embed(query)
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_embedder.py -v`
Expected: 全部 PASS（首次运行会自动下载 BGE-M3 模型，约 2GB）

- [ ] **Step 5: 提交**

```bash
git add src/embedder.py tests/test_embedder.py
git commit -m "feat: embedder — BGE-M3 local embeddings with batch support"
```

---

### Task 5: 向量存储模块

**Files:**
- Create: `src/vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: 编写测试**

```python
"""Tests for vector_store.py"""
import pytest
import numpy as np
from src.vector_store import VectorStore


class TestVectorStore:
    @pytest.fixture
    def store(self):
        """每个测试使用独立的内存存储."""
        store = VectorStore(collection_name="test_kb", persist_directory=None)
        yield store
        # 清理
        store.clear()

    def test_add_and_count(self, store):
        """添加文档后 count 应增加"""
        docs = [
            {"content": "文档一：Python 编程", "file": "a.pdf", "page": 1},
            {"content": "文档二：机器学习", "file": "a.pdf", "page": 2},
        ]
        embeddings = [np.random.rand(1024).tolist() for _ in docs]
        store.add(docs, embeddings)
        assert store.count() == 2

    def test_search_returns_top_k(self, store):
        """搜索应返回 top_k 条结果"""
        docs = [
            {"content": f"内容{i}", "file": "a.pdf", "page": i}
            for i in range(1, 11)
        ]
        embeddings = [np.random.rand(1024).tolist() for _ in docs]
        store.add(docs, embeddings)

        query_vec = np.random.rand(1024).tolist()
        results = store.search(query_vec, top_k=3)
        assert len(results) == 3
        # 每条结果应包含元数据
        for r in results:
            assert "content" in r
            assert "file" in r
            assert "page" in r

    def test_search_empty_store_returns_empty(self, store):
        """空知识库搜索返回空列表"""
        results = store.search(np.random.rand(1024).tolist(), top_k=4)
        assert results == []

    def test_clear_removes_all(self, store):
        """清空后 count 应为 0"""
        docs = [{"content": "测试", "file": "a.pdf", "page": 1}]
        store.add(docs, [np.random.rand(1024).tolist()])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0

    def test_has_content(self, store):
        """has_content 检查文本内容是否存在"""
        docs = [{"content": "独一无二的内容", "file": "a.pdf", "page": 1}]
        store.add(docs, [np.random.rand(1024).tolist()])
        assert store.has_content("独一无二的内容") is True
        assert store.has_content("不存在的内容") is False

    def test_get_sources(self, store):
        """get_sources 返回所有不重复的来源文件列表"""
        docs = [
            {"content": "a", "file": "file1.pdf", "page": 1},
            {"content": "b", "file": "file1.pdf", "page": 2},
            {"content": "c", "file": "file2.pdf", "page": 1},
        ]
        store.add(docs, [np.random.rand(1024).tolist() for _ in docs])
        sources = store.get_sources()
        assert "file1.pdf" in sources
        assert "file2.pdf" in sources
        assert len(sources) == 2
```

- [ ] **Step 2: 确认测试失败**

Run: `pytest tests/test_vector_store.py -v`
Expected: 全部 FAIL

- [ ] **Step 3: 编写实现**

```python
"""向量存储模块 — ChromaDB 封装，文档增删查."""
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.config import COLLECTION_NAME, DATA_DIR


class VectorStore:
    """ChromaDB 向量存储封装."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: Optional[str] = None,
    ):
        if persist_directory is None:
            persist_directory = str(DATA_DIR / "chromadb")

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

    def add(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """批量添加文档块到向量库."""
        if not chunks:
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "file": c.get("file", ""),
                "page": c.get("page", 0),
                "chunk_id": c.get("chunk_id", ""),
            }
            for c in chunks
        ]
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """向量相似度搜索，返回 top_k 条最相关结果."""
        if self.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "content": results["documents"][0][i],
                "file": results["metadatas"][0][i]["file"],
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i],
            })
        return items

    def count(self) -> int:
        """返回当前文档块总数."""
        return self._collection.count()

    def clear(self) -> None:
        """清空知识库."""
        if self.count() > 0:
            ids = self._collection.get()["ids"]
            self._collection.delete(ids=ids)

    def has_content(self, text: str) -> bool:
        """检查内容是否已存在（精确匹配去重）."""
        if self.count() == 0:
            return False
        results = self._collection.get(
            where_document={"$contains": text[:100]}  # 仅用前 100 字符匹配
        )
        return len(results["ids"]) > 0

    def get_sources(self) -> List[str]:
        """返回所有不重复的源文件名列表."""
        if self.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])
        files = {m["file"] for m in all_meta["metadatas"] if m.get("file")}
        return sorted(files)
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_vector_store.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/vector_store.py tests/test_vector_store.py
git commit -m "feat: vector store — ChromaDB CRUD with cosine similarity search"
```

---

### Task 6: LLM 客户端模块

**Files:**
- Create: `src/llm_client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: 编写测试**

```python
"""Tests for llm_client.py"""
import pytest
from unittest.mock import patch, MagicMock
from src.llm_client import LLMClient, LLMCallError


class TestLLMClient:
    @pytest.fixture
    def client(self):
        return LLMClient(api_key="test-key", base_url="https://test.api")

    def test_build_prompt_includes_context(self, client):
        """构建提示词应包含检索上下文和用户问题"""
        question = "什么是SLAM？"
        context_chunks = [
            {"content": "SLAM即同步定位与地图构建。", "file": "a.pdf", "page": 1},
            {"content": "VINS-Mono是一种视觉惯性SLAM方案。", "file": "b.pdf", "page": 3},
        ]
        prompt = client._build_prompt(question, context_chunks)
        assert "SLAM" in prompt
        assert "同步定位" in prompt
        assert "VINS-Mono" in prompt
        assert question in prompt

    def test_build_prompt_includes_source_citations(self, client):
        """构建提示词应指示 LLM 标注来源"""
        question = "test"
        context_chunks = [{"content": "答案内容", "file": "test.pdf", "page": 2}]
        prompt = client._build_prompt(question, context_chunks)
        assert "来源" in prompt or "source" in prompt.lower()
        assert "test.pdf" in prompt

    def test_answer_with_empty_context(self, client):
        """空上下文时应返回无相关内容提示"""
        with patch.object(client, "_call_api") as mock_call:
            mock_call.return_value = "无关回答"
            result = client.answer("随机问题", [])
            assert mock_call.called
            assert len(result) > 0

    @patch("openai.OpenAI")
    def test_api_call_retry_on_failure(self, mock_openai_cls, client):
        """API 调用失败重试 2 次后应抛出 LLMCallError"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            Exception("网络错误"),
            Exception("网络错误"),
            Exception("网络错误"),
        ]
        mock_openai_cls.return_value = mock_client

        with pytest.raises(LLMCallError):
            client._call_api([{"role": "user", "content": "test"}])
        assert mock_client.chat.completions.create.call_count == 3  # 1 次原始 + 2 次重试

    @patch("openai.OpenAI")
    def test_api_call_success(self, mock_openai_cls, client):
        """成功调用应返回消息文本"""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "这是回答"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        mock_openai_cls.return_value = mock_client

        result = client._call_api([{"role": "user", "content": "你好"}])
        assert result == "这是回答"

    def test_extract_citations(self, client):
        """解析 LLM 输出中的引用标记 [N]"""
        response = """回答内容...
参考来源：
[1] test.pdf 第1页 — 「原文片段一」
[2] test2.pdf 第3页 — 「原文片段二」"""
        citations = client._extract_citations(response)
        assert len(citations) >= 2
```

- [ ] **Step 2: 确认测试失败**

Run: `pytest tests/test_llm_client.py -v`
Expected: 全部 FAIL

- [ ] **Step 3: 编写实现**

```python
"""LLM 客户端模块 — DeepSeek API 封装，带重试和 prompt 模板."""
import re
from typing import List, Dict, Any
from openai import OpenAI
from src.config import LLM_BASE_URL, LLM_MODEL


class LLMCallError(Exception):
    """LLM API 调用失败."""


class LLMClient:
    """DeepSeek LLM 客户端（兼容 OpenAI SDK）."""

    MAX_RETRIES = 2

    def __init__(self, api_key: str, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """基于知识库上下文回答问题."""
        prompt = self._build_prompt(question, context_chunks)
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages)

    def _build_prompt(
        self, question: str, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """构建 RAG 提示词模板."""
        if not context_chunks:
            return (
                f"用户问题：{question}\n\n"
                "知识库中暂无相关内容，请告知用户先上传相关文档再提问。"
            )

        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += (
                f"[{i}] 来源：{chunk['file']} 第{chunk['page']}页\n"
                f"内容：{chunk['content']}\n\n"
            )

        return f"""你是一个知识库问答助手。请基于以下文档内容回答用户的问题。

## 规则
1. 只用提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确说"知识库中暂无相关内容"
3. 回答末尾标注引用的来源，格式：📎 参考来源：[N] 文件名 第X页 — 「原文片段」
4. 回答使用中文

## 相关文档
{context_text}

## 用户问题
{question}

## 回答"""
        return prompt

    def _call_api(self, messages: List[Dict], retries: int = 0) -> str:
        """调用 LLM API，带自动重试."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.3,  # 低温度保证回答一致性
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            if retries < self.MAX_RETRIES:
                return self._call_api(messages, retries + 1)
            raise LLMCallError(
                f"LLM API 调用失败（已重试 {self.MAX_RETRIES} 次）：{e}"
            )

    def _extract_citations(self, response: str) -> List[Dict[str, str]]:
        """从 LLM 回答中提取引用信息."""
        pattern = re.compile(
            r'\[(\d+)\]\s+(\S+?)\s+第(\d+)页\s*[—\-—]\s*「(.+?)」'
        )
        citations = []
        for m in pattern.finditer(response):
            citations.append({
                "num": m.group(1),
                "file": m.group(2),
                "page": m.group(3),
                "snippet": m.group(4),
            })
        return citations
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_llm_client.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/llm_client.py tests/test_llm_client.py
git commit -m "feat: LLM client — DeepSeek API with retry and RAG prompt template"
```

---

### Task 7: RAG 引擎（核心编排）

**Files:**
- Create: `src/rag_engine.py`
- Create: `tests/test_rag_engine.py`

- [ ] **Step 1: 编写测试**

```python
"""Tests for rag_engine.py"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.rag_engine import RAGEngine


class TestRAGEngine:
    @pytest.fixture
    def engine(self):
        """创建使用内存存储的 RAG 引擎."""
        with patch("src.rag_engine.Embedder") as mock_embedder_cls, \
             patch("src.rag_engine.LLMClient") as mock_llm_cls:

            mock_embedder = MagicMock()
            mock_embedder.dimension = 1024
            mock_embedder.embed_batch.return_value = np.random.rand(3, 1024)
            mock_embedder.embed_query.return_value = np.random.rand(1024)
            mock_embedder_cls.return_value = mock_embedder

            mock_llm = MagicMock()
            mock_llm.answer.return_value = "这是基于文档的回答。[1] test.pdf 第1页"
            mock_llm_cls.return_value = mock_llm

            engine = RAGEngine(
                llm_api_key="test-key",
                persist_directory=None,  # 内存模式
            )
            yield engine

    def test_ingest_file_increases_count(self, engine):
        """摄入文件后文档计数增加"""
        initial = engine.get_document_count()
        engine.ingest_text("测试文档内容。", "test.pdf", 1)
        assert engine.get_document_count() == initial + 1

    def test_ingest_duplicate_is_skipped(self, engine):
        """重复内容应跳过"""
        text = "独一无二的内容用于去重测试。"
        engine.ingest_text(text, "test.pdf", 1)
        count = engine.get_document_count()
        engine.ingest_text(text, "test.pdf", 1)
        assert engine.get_document_count() == count  # 不变

    def test_ingest_file_computes_md5(self, engine, tmp_path):
        """摄入文件应计算 MD5 去重"""
        f = tmp_path / "test.txt"
        f.write_text("测试文件内容", encoding="utf-8")
        engine.ingest_file(f)
        assert engine.is_file_ingested(f) is True

    def test_query_returns_answer(self, engine):
        """查询应返回回答"""
        engine.ingest_text("SLAM 是同步定位与地图构建。", "a.pdf", 1)
        result = engine.query("什么是 SLAM？")
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert "sources" in result

    def test_query_empty_store_returns_not_found(self, engine):
        """空知识库查询应返回提示"""
        result = engine.query("任意问题")
        assert "暂无" in result["answer"] or "上传" in result["answer"]

    def test_get_sources_list(self, engine):
        """获取已上传来源列表"""
        engine.ingest_text("内容A", "file_a.pdf", 1)
        engine.ingest_text("内容B", "file_b.pdf", 1)
        sources = engine.get_sources()
        assert "file_a.pdf" in sources
        assert "file_b.pdf" in sources

    def test_clear_knowledge_base(self, engine):
        """清空知识库后计数归零"""
        engine.ingest_text("测试", "a.pdf", 1)
        engine.clear()
        assert engine.get_document_count() == 0
```

- [ ] **Step 2: 确认测试失败**

Run: `pytest tests/test_rag_engine.py -v`
Expected: 全部 FAIL

- [ ] **Step 3: 编写实现**

```python
"""RAG 引擎 — 核心编排层，连接嵌入、向量库、LLM."""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import TOP_K
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.file_parser import FileParser
from src.text_splitter import TextSplitter
from src.llm_client import LLMClient


class RAGEngine:
    """RAG 核心引擎，协调文档摄入和问答流程."""

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        self._embedder = Embedder()
        self._vector_store = VectorStore(persist_directory=persist_directory)
        self._llm = LLMClient(
            api_key=llm_api_key,
            base_url=llm_base_url or "https://api.deepseek.com",
        )
        self._ingested_files: Dict[str, str] = {}  # 路径 -> MD5 映射

    def ingest_file(self, file_path: Path) -> int:
        """摄入单个文件：解析 → 分块 → 嵌入 → 存储。返回新增块数."""
        md5 = FileParser.compute_md5(file_path)
        key = str(file_path.absolute())

        # 去重检查
        if key in self._ingested_files and self._ingested_files[key] == md5:
            return 0  # 已摄入，跳过

        text = FileParser.parse(file_path)
        if not text.strip():
            return 0

        chunks = self._ingest_text(text, file_path.name)
        self._ingested_files[key] = md5
        return len(chunks)

    def ingest_text(self, text: str, file_name: str, page_num: int = 1) -> int:
        """摄入纯文本（测试用），返回块数."""
        chunks = self._ingest_text(text, file_name, page_num)
        return len(chunks)

    def _ingest_text(
        self, text: str, file_name: str, page_num: int = 1
    ) -> List[Dict[str, Any]]:
        """文本摄入内部实现."""
        parts = text.split("\n\n[第")  # 按 PDF 页码标记分割
        all_chunks = []

        if len(parts) > 1:
            # 有页码标记的 PDF
            for part in parts:
                if not part.strip():
                    continue
                try:
                    # 格式: "X页]\n内容..."
                    page_str, content = part.split("页]\n", 1)
                    page = int(page_str)
                except (ValueError, IndexError):
                    content = part
                    page = page_num
                chunks = TextSplitter.split_with_metadata(
                    content, file_name, page
                )
                all_chunks.extend(chunks)
        else:
            # 无页码标记（纯文本/Word）
            chunks = TextSplitter.split_with_metadata(
                text, file_name, page_num
            )
            all_chunks.extend(chunks)

        # 去重 + 嵌入 + 存储
        new_chunks = []
        for chunk in all_chunks:
            if not self._vector_store.has_content(chunk["content"]):
                new_chunks.append(chunk)

        if new_chunks:
            embeddings = self._embedder.embed_batch(
                [c["content"] for c in new_chunks]
            )
            self._vector_store.add(new_chunks, embeddings.tolist())

        return new_chunks

    def query(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """用户提问，返回回答 + 来源."""
        # 空知识库检查
        if self._vector_store.count() == 0:
            return {
                "answer": "📭 知识库为空，请先在侧边栏上传文档再提问哦～",
                "sources": [],
            }

        # 查询向量化
        query_vec = self._embedder.embed_query(question)

        # 检索相关文档
        retrieved = self._vector_store.search(query_vec.tolist(), top_k=top_k)

        # LLM 生成回答
        answer = self._llm.answer(question, retrieved)

        return {
            "answer": answer,
            "sources": [
                {
                    "file": r["file"],
                    "page": r["page"],
                    "snippet": r["content"][:100],
                }
                for r in retrieved
            ],
        }

    def is_file_ingested(self, file_path: Path) -> bool:
        """检查文件是否已摄入."""
        return str(file_path.absolute()) in self._ingested_files

    def get_document_count(self) -> int:
        """返回知识库文档块总数."""
        return self._vector_store.count()

    def get_sources(self) -> List[str]:
        """返回所有已摄入的源文件名."""
        return self._vector_store.get_sources()

    def clear(self) -> None:
        """清空知识库."""
        self._vector_store.clear()
        self._ingested_files.clear()
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_rag_engine.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/rag_engine.py tests/test_rag_engine.py
git commit -m "feat: RAG engine — core pipeline: ingest, embed, search, answer"
```

---

### Task 8: Streamlit UI

**Files:**
- Create: `src/ui_components.py`
- Create: `app.py`

- [ ] **Step 1: 编写 UI 组件**

```python
"""Streamlit UI 组件 — 侧边栏、聊天界面、文件上传."""
import streamlit as st
from pathlib import Path
from src.config import SUPPORTED_SUFFIXES
from src.file_parser import FileParser, FileParseError, UnsupportedFormatError


def init_page():
    """初始化页面配置."""
    st.set_page_config(
        page_title="知识小助手",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar(engine) -> list:
    """渲染侧边栏 — 文件上传管理。返回是否需要重建 session."""
    with st.sidebar:
        st.title("📚 知识小助手")
        st.caption("上传你的文档，我来帮你回答问题～")

        # API Key 配置
        api_key = st.text_input(
            "🔑 DeepSeek API Key",
            type="password",
            placeholder="sk-...",
            help="在 https://platform.deepseek.com 获取",
        )

        st.divider()

        # 文件上传
        uploaded_files = st.file_uploader(
            "📄 上传文档",
            type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
            accept_multiple_files=True,
            help=f"支持：{', '.join(SUPPORTED_SUFFIXES)}，单文件 ≤ 50MB",
        )

        if uploaded_files and api_key:
            for uf in uploaded_files:
                # 保存临时文件
                tmp_path = Path("data/tmp") / uf.name
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(uf.getvalue())

                try:
                    new_count = engine.ingest_file(tmp_path)
                    if new_count > 0:
                        st.toast(f"✅ {uf.name} — 已添加 {new_count} 个片段", icon="✅")
                    else:
                        st.toast(f"⏭️ {uf.name} — 已存在，跳过", icon="⏭️")
                except UnsupportedFormatError as e:
                    st.error(str(e))
                except FileParseError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"❌ 处理 {uf.name} 失败：{e}")

                # 清理临时文件
                tmp_path.unlink(missing_ok=True)

        st.divider()

        # 已上传文档列表
        sources = engine.get_sources()
        if sources:
            st.subheader("📋 已加载文档")
            for s in sources:
                st.text(f"  • {s}")

            st.caption(f"共 {engine.get_document_count()} 个文档片段")

            if st.button("🗑️ 清空知识库", use_container_width=True):
                engine.clear()
                st.rerun()

        return api_key


def render_chat(engine, api_key: str):
    """渲染主聊天界面."""
    st.title("💬 知识库问答")

    if engine.get_document_count() == 0:
        st.info("👈 请先在侧边栏上传文档并填写 API Key")
        return

    # 对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 参考来源"):
                    for src in msg["sources"]:
                        st.caption(
                            f"[{src.get('index', '?')}] **{src['file']}** "
                            f"第{src['page']}页 — 「{src['snippet']}...」"
                        )

    # 输入区
    if prompt := st.chat_input("输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在检索知识库..."):
                result = engine.query(prompt)
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📎 参考来源"):
                        for i, src in enumerate(result["sources"], 1):
                            st.caption(
                                f"[{i}] **{src['file']}** "
                                f"第{src['page']}页 — 「{src['snippet']}...」"
                            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": [
                {**src, "index": i}
                for i, src in enumerate(result["sources"], 1)
            ],
        })
```

- [ ] **Step 2: 编写 app.py 主入口**

```python
"""知识小助手 — RAG 智能知识库问答系统."""
import streamlit as st
from src.rag_engine import RAGEngine
from src.ui_components import init_page, render_sidebar, render_chat


def main():
    init_page()

    # 初始化 RAG 引擎（session 级别，保持状态）
    if "engine" not in st.session_state:
        st.session_state.engine = RAGEngine(
            llm_api_key="",  # 从 sidebar 动态获取
        )

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    # 侧边栏
    api_key = render_sidebar(st.session_state.engine)

    # API Key 变更时重建引擎
    if api_key and api_key != st.session_state.api_key:
        from src.llm_client import LLMClient
        st.session_state.engine._llm = LLMClient(api_key=api_key)
        st.session_state.api_key = api_key

    # 错误提示
    if not api_key:
        st.warning("⚠️ 请在左侧填写 DeepSeek API Key 后开始使用")

    # 主聊天区
    if api_key:
        render_chat(st.session_state.engine, api_key)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 本地启动测试**

Run: `streamlit run app.py`
Expected: 浏览器打开，能看到侧边栏和聊天界面

- [ ] **Step 4: 手动集成测试**

1. 填写 DeepSeek API Key
2. 上传一个 PDF 文件
3. 在聊天框输入问题
4. 验证：有回答、有来源标注

- [ ] **Step 5: 提交**

```bash
git add app.py src/ui_components.py
git commit -m "feat: Streamlit UI — sidebar file manager + chat interface"
```

---

### Task 9: README 完善 & 项目收尾

**Files:**
- Modify: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: 编写 .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
.venv/
venv/

# Data
data/
*.db

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 2: 完善 README.md**

```markdown
# 📚 知识小助手 (Knowledge Assistant)

> RAG 智能知识库问答系统 — 上传文档，AI 精准回答并标注来源

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector--DB-green)](https://trychroma.com)

## ✨ 功能

- 📄 **多格式支持**：PDF / Word / TXT / Markdown
- 🔍 **语义搜索**：BGE-M3 中文嵌入模型，精准检索
- 🤖 **AI 回答**：DeepSeek 大模型，基于文档内容生成回答
- 📎 **来源追溯**：每条回答标注参考文件、页码、原文片段
- ⚡ **2 周快速开发**：Streamlit 极简界面，从零到 Demo

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/3480413499/knowledge-assistant.git
cd knowledge-assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 获取 API Key

在 [DeepSeek 开放平台](https://platform.deepseek.com) 注册并获取 API Key

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

## 🏗️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 界面 | Streamlit | Python 原生 Web UI |
| LLM | DeepSeek API | 中文问答能力强，性价比高 |
| 嵌入 | BGE-M3 (BAAI) | C-MTEB 前列，本地免费部署 |
| 向量库 | ChromaDB | 轻量级，零部署成本 |
| 文档解析 | PyMuPDF + python-docx | PDF/Word 全覆盖 |

## 📁 项目结构

```
knowledge-assistant/
├── app.py                  # Streamlit 主入口
├── src/
│   ├── config.py           # 配置管理
│   ├── file_parser.py      # 多格式文件解析
│   ├── text_splitter.py    # 文本智能分块
│   ├── embedder.py         # BGE-M3 嵌入模型
│   ├── vector_store.py     # ChromaDB 向量存储
│   ├── llm_client.py       # LLM API 客户端
│   ├── rag_engine.py       # RAG 核心引擎
│   └── ui_components.py    # Streamlit UI 组件
├── tests/                  # 单元测试
├── docs/
│   ├── design.md           # 设计文档
│   └── superpowers/plans/  # 实现计划
└── requirements.txt
```

## 🧪 运行测试

```bash
pytest tests/ -v
```

## 📝 开发计划

- [x] 项目初始化 & 设计文档
- [ ] 核心模块开发 (进行中)
- [ ] Streamlit UI
- [ ] 在线部署 (Streamlit Cloud)

## 📄 License

MIT
```

- [ ] **Step 3: 最终提交并推送**

```bash
git add .gitignore README.md
git commit -m "docs: complete README with setup guide and project structure"
git push origin main
```

---

## Self-Review

### 1. Spec Coverage
- ✅ 文档上传 → Task 8 (UI) + Task 7 (RAG Engine) + Task 2 (File Parser)
- ✅ 智能分块 → Task 3
- ✅ BGE-M3 嵌入 → Task 4
- ✅ ChromaDB 存储 → Task 5
- ✅ LLM 问答 → Task 6
- ✅ 来源标注 → Task 6 (prompt 模板) + Task 8 (UI 展示)
- ✅ 文件去重 → Task 2 (MD5) + Task 7 (ingest_file 去重检查)
- ✅ 7 种错误处理 → 分布在各自模块中
- ✅ 在线部署 → Task 9 README 提及（Streamlit Cloud 部署不在本次代码范围内）

### 2. Placeholder Scan
- ✅ 无 TBD/TODO
- ✅ 所有代码步骤有完整实现
- ✅ 所有命令有确切预期输出

### 3. Type Consistency
- ✅ FileParser.parse() 返回 str，RAGEngine 消费 str
- ✅ Embedder.embed_batch() 返回 np.ndarray，VectorStore 接收 List[List[float]]
- ✅ chunk 字典格式一致：{content, file, page, chunk_id}
