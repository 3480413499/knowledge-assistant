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


def render_sidebar(engine) -> str:
    """渲染侧边栏 — 文件上传管理。返回 api_key."""
    with st.sidebar:
        st.title("📚 知识小助手")
        st.caption("上传你的文档，我来帮你回答问题～")

        api_key = st.text_input(
            "🔑 DeepSeek API Key",
            type="password",
            placeholder="sk-...",
            help="在 https://platform.deepseek.com 获取",
        )

        st.divider()

        uploaded_files = st.file_uploader(
            "📄 上传文档",
            type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
            accept_multiple_files=True,
            help=f"支持：{', '.join(SUPPORTED_SUFFIXES)}，单文件 ≤ 50MB",
            key="file_uploader",
        )

        if uploaded_files and api_key:
            for uf in uploaded_files:
                tmp_dir = Path("data/tmp")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / uf.name
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

                tmp_path.unlink(missing_ok=True)

        st.divider()

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


def render_chat(engine):
    """渲染主聊天界面."""
    st.title("💬 知识库问答")

    if engine.get_document_count() == 0:
        st.info("👈 请先在侧边栏上传文档并填写 API Key")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

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

    if prompt := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

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
