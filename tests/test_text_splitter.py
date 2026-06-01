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
        text = "0123456789" * 50
        chunks = TextSplitter.split(text, chunk_size=200, overlap=50)
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

    def test_whitespace_only_returns_empty_list(self):
        """纯空白文本返回空列表"""
        chunks = TextSplitter.split("   \n  \t  ", chunk_size=500, overlap=50)
        assert chunks == []

    def test_paragraph_boundary_split(self):
        """在段落边界处分块"""
        paragraphs = ["段落一。" * 80, "段落二。" * 80, "段落三。" * 80]
        text = "\n\n".join(paragraphs)
        chunks = TextSplitter.split(text, chunk_size=300, overlap=50)
        assert len(chunks) >= 3

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
