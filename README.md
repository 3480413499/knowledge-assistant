# 📚 知识小助手 (Knowledge Assistant)

> RAG 智能知识库问答系统 — 上传文档，AI 精准回答并标注来源

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector--DB-green)](https://trychroma.com)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple)](https://platform.deepseek.com)

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

首次运行会自动下载 BGE-M3 嵌入模型（约 2GB），请耐心等待。

### 3. 获取 API Key

在 [DeepSeek 开放平台](https://platform.deepseek.com) 注册并获取 API Key（新用户有免费额度）。

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

### 5. 使用方式

1. 在左侧填写 DeepSeek API Key
2. 上传 PDF / Word / TXT 文档
3. 在聊天框用自然语言提问
4. AI 回答并标注参考来源

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
├── requirements.txt        # 依赖清单
├── src/
│   ├── config.py           # 配置管理
│   ├── file_parser.py      # 多格式文件解析
│   ├── text_splitter.py    # 文本智能分块
│   ├── embedder.py         # BGE-M3 嵌入模型
│   ├── vector_store.py     # ChromaDB 向量存储
│   ├── llm_client.py       # LLM API 客户端
│   ├── rag_engine.py       # RAG 核心引擎
│   └── ui_components.py    # Streamlit UI 组件
├── tests/                  # 单元测试（39 个）
├── docs/
│   ├── design.md           # 设计文档
│   └── superpowers/plans/  # 实现计划
└── .gitignore
```

## 🧪 运行测试

```bash
pytest tests/ -v
```

## 📝 开发计划

- [x] 项目初始化 & 设计文档
- [x] 核心模块开发（文件解析、分块、嵌入、向量库、LLM、RAG引擎）
- [x] Streamlit 界面
- [ ] 在线部署 (Streamlit Cloud / Hugging Face Spaces)
- [ ] 混合检索 (BM25 + 向量)
- [ ] Reranker 二次排序

## 📄 License

MIT
