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

    def _call_api(self, messages: List[Dict], retries: int = 0) -> str:
        """调用 LLM API，带自动重试."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.3,
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
