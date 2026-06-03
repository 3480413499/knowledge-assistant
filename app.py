"""知识小助手 — RAG 智能知识库问答系统."""
import streamlit as st
from src.rag_engine import RAGEngine
from src.llm_client import LLMClient
from src.embedder import Embedder
from src.config import load_api_key, load_kb_names, DEFAULT_KB_NAME
from src.ui_components import init_page, render_sidebar, render_chat


def main():
    init_page()

    # 用 session_state 持久化 embedder，避免每次 rerun 都重建
    if "embedder" not in st.session_state:
        st.session_state.embedder = Embedder()

    if "engine" not in st.session_state:
        # 初始化引擎，使用已保存的知识库列表中的第一个
        kb_list = load_kb_names()
        initial_kb = kb_list[0] if kb_list else DEFAULT_KB_NAME
        st.session_state.engine = RAGEngine(
            llm_api_key="",
            embedder=st.session_state.embedder,
            kb_name=initial_kb,
        )

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    # 用已保存的 Key 预填充输入框
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = load_api_key()

    api_key = render_sidebar(st.session_state.engine)

    if api_key and api_key != st.session_state.api_key:
        st.session_state.engine._llm = LLMClient(api_key=api_key)
        st.session_state.api_key = api_key

    if not api_key:
        st.warning("请在左侧填写 DeepSeek API Key 后开始使用")

    if api_key:
        render_chat(st.session_state.engine)


if __name__ == "__main__":
    main()
