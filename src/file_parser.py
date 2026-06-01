"""文件解析模块 — 支持 PDF/Word/TXT/Markdown 的文本提取."""
import hashlib
from pathlib import Path
from src.config import MAX_FILE_SIZE, SUPPORTED_SUFFIXES


class UnsupportedFormatError(Exception):
    """不支持的文件格式."""

class FileParseError(Exception):
    """文件解析失败."""


class FileParser:
    """多格式文件解析器."""

    SUFFIX_MAP = {
        "pdf": [".pdf"],
        "docx": [".docx"],
        "text": [".txt", ".md"],
    }

    @classmethod
    def parse(cls, file_path: Path) -> str:
        """解析文件，统一返回文本字符串."""
        file_path = Path(file_path)

        if file_path.stat().st_size > MAX_FILE_SIZE:
            raise FileParseError(
                f"文件过大：{file_path.name} 超过 50MB 上限，请拆分后上传"
            )

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
        import fitz
        doc = fitz.open(str(file_path))
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[第{page_num}页]\n{text}")
        doc.close()

        if not text_parts:
            raise FileParseError(
                f"该 PDF ({file_path.name}) 为扫描图片，无文字层。"
                f"请先用 OCR 工具（如 WPS）转换后上传。"
            )
        return "\n\n".join(text_parts)

    @classmethod
    def _parse_docx(cls, file_path: Path) -> str:
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
