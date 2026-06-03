"""Streamlit UI 组件 — 侧边栏、聊天界面、文件上传."""
import streamlit as st
from pathlib import Path
from src.config import (
    SUPPORTED_SUFFIXES, DEFAULT_KB_NAME,
    save_api_key, load_kb_names, save_kb_names,
    save_chat_history, load_chat_history,
)
from src.file_parser import FileParser, FileParseError, UnsupportedFormatError


def _inject_cn_translations():
    """注入 CSS，把 Streamlit 自带英文替换为中文，隐藏录制/截图，调整布局."""
    st.markdown(
        """
        <style>
        /* 文件上传区：拖拽提示 */
        [data-testid="stFileUploadDropzone"] [data-testid="stText"] {
            visibility: hidden;
        }
        [data-testid="stFileUploadDropzone"] [data-testid="stText"]::after {
            content: "拖拽文件到此处，或点击下方按钮选择文件";
            visibility: visible;
            display: block;
            white-space: nowrap;
            font-size: 14px;
        }
        /* 浏览文件按钮 */
        [data-testid="stFileUploader"] > section > button {
            font-size: 0 !important;
        }
        [data-testid="stFileUploader"] > section > button::before {
            content: "📂 选择文件";
            font-size: 16px !important;
        }
        /* 隐藏菜单中的录制/截图项（aria-label + nth 双保险） */
        button[aria-label*="Record"],
        button[aria-label*="screenshot"],
        button[aria-label*="Screenshot"],
        [data-testid="stMainMenu"] button:nth-of-type(3),
        [data-testid="stMainMenu"] button:nth-of-type(4) {
            display: none !important;
        }
        /* 标题靠左上角（留出顶栏空间） */
        .block-container {
            padding-top: 3.5rem !important;
        }
        .block-container h1 {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_THEMES = {
    "默认": "",
    "牛皮纸暖黄": """
        <style>
        /* === 核心底色：只改最外层容器 === */
        .stApp {
            background: #EDD9B5 !important;
        }
        header[data-testid="stHeader"] {
            background: #EDD9B5 !important;
        }
        [data-testid="stSidebar"] {
            background: #E8D5A8 !important;
        }
        .block-container {
            background: #EDD9B5 !important;
        }
        footer {
            background: #EDD9B5 !important;
        }
        /* === 侧边栏文字 === */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] li, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] small, [data-testid="stSidebar"] caption,
        [data-testid="stSidebar"] .stMarkdown {
            color: #4A3728 !important;
        }
        /* === 主区文字 === */
        .main h1, .main h2, .main h3, .main h4,
        .main p, .main span, .main li, .main label {
            color: #4A3728 !important;
        }
        /* === 聊天消息：微调背景，不碰内部结构 === */
        [data-testid="stChatMessage"] {
            background: #FBF3E4 !important;
        }
        /* === 聊天输入 === */
        [data-testid="stChatInput"] textarea {
            background: #FBF3E4 !important;
            color: #4A3728 !important;
            border-color: #C4A87C !important;
        }
        /* === 侧边栏输入框 === */
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
            background: #FBF3E4 !important;
            color: #4A3728 !important;
            border-color: #C4A87C !important;
        }
        /* === 红色保持（错误/警告不变） === */
        </style>
    """,
    "护眼柔绿": """
        <style>
        /* === 核心底色：只改最外层容器 === */
        .stApp {
            background: #DCE8D7 !important;
        }
        header[data-testid="stHeader"] {
            background: #DCE8D7 !important;
        }
        [data-testid="stSidebar"] {
            background: #D4E5CF !important;
        }
        .block-container {
            background: #DCE8D7 !important;
        }
        footer {
            background: #DCE8D7 !important;
        }
        /* === 侧边栏文字 === */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] li, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] small, [data-testid="stSidebar"] caption,
        [data-testid="stSidebar"] .stMarkdown {
            color: #2C3E2D !important;
        }
        /* === 主区文字 === */
        .main h1, .main h2, .main h3, .main h4,
        .main p, .main span, .main li, .main label {
            color: #2C3E2D !important;
        }
        /* === 聊天消息：微调背景，不碰内部结构 === */
        [data-testid="stChatMessage"] {
            background: #F2F7EF !important;
        }
        /* === 聊天输入 === */
        [data-testid="stChatInput"] textarea {
            background: #F2F7EF !important;
            color: #2C3E2D !important;
            border-color: #8DB48E !important;
        }
        /* === 侧边栏输入框 === */
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
            background: #F2F7EF !important;
            color: #2C3E2D !important;
            border-color: #8DB48E !important;
        }
        </style>
    """,
}


def _inject_theme():
    """根据 session_state 注入护眼主题 CSS."""
    theme = st.session_state.get("theme", "默认")
    css = _THEMES.get(theme, "")
    if css:
        st.markdown(css, unsafe_allow_html=True)


def init_page():
    """初始化页面配置."""
    st.set_page_config(
        page_title="知识小助手",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_cn_translations()
    _inject_theme()


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
            key="api_key_input",
        )

        # 自动保存 Key
        if api_key and api_key != st.session_state.get("_saved_key", ""):
            save_api_key(api_key)
            st.session_state._saved_key = api_key

        st.divider()

        # 护眼主题切换
        theme = st.selectbox(
            "🎨 显示风格",
            ["默认", "牛皮纸暖黄", "护眼柔绿"],
            key="theme",
            help="切换背景色，保护视力～",
        )

        st.divider()

        # ===== 知识库管理 =====
        kb_names = engine.list_kb()
        if not kb_names:
            kb_names = [DEFAULT_KB_NAME]

        # 确保当前 KB 在列表中
        current_idx = kb_names.index(engine.kb_name) if engine.kb_name in kb_names else 0

        selected_kb = st.selectbox(
            "📁 知识库",
            kb_names,
            index=current_idx,
            key="kb_selector",
            help="切换不同知识库，各自独立的文档和对话",
        )

        if selected_kb and selected_kb != engine.kb_name:
            old_kb = engine.kb_name
            save_chat_history(st.session_state.get("messages", []), old_kb)
            engine.switch_kb(selected_kb)
            st.session_state.messages = load_chat_history(selected_kb)
            st.rerun()

        # 新建 / 删除 KB
        col_new, col_del = st.columns([3, 1])
        with col_new:
            with st.popover("➕ 新建知识库"):
                new_name = st.text_input("名称", placeholder="例如：AI导论、信号处理", key="new_kb_name")
                if st.button("创建", key="create_kb") and new_name:
                    name = new_name.strip()
                    kb_names = engine.list_kb()
                    if name not in kb_names:
                        kb_names.append(name)
                        save_kb_names(kb_names)
                    engine.switch_kb(name)
                    st.session_state.messages = []
                    st.rerun()
        with col_del:
            if len(kb_names) > 1:
                with st.popover("🗑️"):
                    st.caption("删除后将无法恢复")
                    if st.button("确认删除", key="del_kb", type="secondary"):
                        kb_names = engine.list_kb()
                        if selected_kb in kb_names:
                            kb_names.remove(selected_kb)
                            save_kb_names(kb_names)
                        engine.delete_kb(selected_kb)
                        if kb_names:
                            engine.switch_kb(kb_names[0])
                        st.session_state.messages = []
                        st.rerun()

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
                with st.status(f"正在处理 {uf.name}...", expanded=True) as file_status:
                    tmp_dir = Path("data/tmp")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tmp_path = tmp_dir / uf.name
                    tmp_path.write_bytes(uf.getvalue())

                    def on_progress(step, total, msg):
                        pct = step / total * 100 if total else 0
                        file_status.update(
                            label=f"🔍 {uf.name}: {step}/{total} ({pct:.0f}%) — {msg}"
                        )

                    try:
                        new_count = engine.ingest_file(
                            tmp_path, progress_callback=on_progress
                        )
                        if new_count > 0:
                            file_status.update(
                                label=f"✅ {uf.name} — 已添加 {new_count} 个片段",
                                state="complete",
                            )
                            st.toast(f"✅ {uf.name} — 已添加 {new_count} 个片段", icon="✅")
                        else:
                            file_status.update(
                                label=f"⏭️ {uf.name} — 已存在，跳过",
                                state="complete",
                            )
                            st.toast(f"⏭️ {uf.name} — 已存在，跳过", icon="⏭️")
                    except UnsupportedFormatError as e:
                        file_status.update(
                            label=f"❌ {uf.name} 格式不支持", state="error"
                        )
                        st.error(str(e))
                    except FileParseError as e:
                        file_status.update(
                            label=f"❌ {uf.name} 解析失败", state="error"
                        )
                        st.error(str(e))
                    except Exception as e:
                        file_status.update(
                            label=f"❌ {uf.name} 处理失败", state="error"
                        )
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
        # 即使没有文档也展示历史对话
        if "messages" in st.session_state and st.session_state.messages:
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
            _render_clear_chat_btn()
        return

    # 加载或初始化对话记录
    if "messages" not in st.session_state:
        saved = load_chat_history(engine.kb_name)
        st.session_state.messages = saved if saved else []

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

    # 清空对话按钮
    _render_clear_chat_btn()

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
        # 持久化保存
        save_chat_history(st.session_state.messages, engine.kb_name)


def _render_clear_chat_btn():
    """渲染清空对话按钮."""
    if not st.session_state.get("messages"):
        return
    kb = st.session_state.get("engine")
    kb_name = kb.kb_name if kb else DEFAULT_KB_NAME
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            save_chat_history([], kb_name)
            st.rerun()

