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

        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
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
