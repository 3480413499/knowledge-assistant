"""知识小助手 — RAG 智能知识库问答系统."""
import streamlit as st
from src.rag_engine import RAGEngine
from src.llm_client import LLMClient
from src.ui_components import init_page, render_sidebar, render_chat


def main():
    init_page()

    if "engine" not in st.session_state:
        st.session_state.engine = RAGEngine(llm_api_key="")

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    api_key = render_sidebar(st.session_state.engine)

    if api_key and api_key != st.session_state.api_key:
        st.session_state.engine._llm = LLMClient(api_key=api_key)
        st.session_state.api_key = api_key

    if not api_key:
        st.warning("⚠️ 请在左侧填写 DeepSeek API Key 后开始使用")

    if api_key:
        render_chat(st.session_state.engine)


if __name__ == "__main__":
    main()
