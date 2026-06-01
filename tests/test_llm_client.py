"""Tests for llm_client.py"""
import pytest
from unittest.mock import patch, MagicMock
from src.llm_client import LLMClient, LLMCallError


class TestLLMClient:
    @pytest.fixture
    def client(self):
        return LLMClient(api_key="test-key", base_url="https://test.api")

    def test_build_prompt_includes_context(self, client):
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
        question = "test"
        context_chunks = [{"content": "答案内容", "file": "test.pdf", "page": 2}]
        prompt = client._build_prompt(question, context_chunks)
        assert "test.pdf" in prompt

    def test_answer_with_empty_context(self, client):
        with patch.object(client, "_call_api") as mock_call:
            mock_call.return_value = "无关回答"
            result = client.answer("随机问题", [])
            assert mock_call.called
            assert len(result) > 0

    def test_api_call_retry_on_failure(self):
        mock_api_client = MagicMock()
        mock_api_client.chat.completions.create.side_effect = [
            Exception("网络错误"),
            Exception("网络错误"),
            Exception("网络错误"),
        ]
        with patch("src.llm_client.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = mock_api_client
            client = LLMClient(api_key="test-key", base_url="https://test.api")

            with pytest.raises(LLMCallError):
                client._call_api([{"role": "user", "content": "test"}])
            assert mock_api_client.chat.completions.create.call_count == 3

    def test_api_call_success(self):
        mock_api_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "这是回答"
        mock_api_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        with patch("src.llm_client.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = mock_api_client
            client = LLMClient(api_key="test-key", base_url="https://test.api")

            result = client._call_api([{"role": "user", "content": "你好"}])
            assert result == "这是回答"

    def test_extract_citations(self, client):
        response = """回答内容...
参考来源：
[1] test.pdf 第1页 — 「原文片段一」
[2] test2.pdf 第3页 — 「原文片段二」"""
        citations = client._extract_citations(response)
        assert len(citations) >= 2
