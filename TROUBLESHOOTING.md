# 踩坑记录与优化日志

本文档记录了知识小助手项目开发过程中遇到的问题及解决方案，供后续维护参考。

---

## 1. ChromaDB 集合名不支持中文

**问题**：创建中文名知识库时报错 `InvalidArgumentError: Expected a name containing 3-512 characters from [a-zA-Z0-9._-]`。

**根因**：ChromaDB 集合名只允许 `[a-zA-Z0-9._-]`，不能直接使用中文。

**解决**：`src/config.py` 中新增 `kb_to_collection()` 函数，将中文显示名通过 SHA256 哈希映射为安全的集合名：
- 默认知识库 → `knowledge_base`（保持向后兼容）
- 其他名称 → `kb_<sha256前12位>`

---

## 2. 新建/删除知识库无效

**问题**：点击"创建"或"删除"知识库后页面刷新，但列表未更新。

**根因**：`save_kb_names(engine.list_kb())` 调用顺序错误。`list_kb()` 从文件读取旧列表，操作后新名字未加入（或删除后未移除）就保存，相当于保存了旧列表。

**解决**：**先更新内存列表 → 保存到文件 → 再操作 ChromaDB**。
```python
# 创建：先加入列表
kb_names = engine.list_kb()
if name not in kb_names:
    kb_names.append(name)
    save_kb_names(kb_names)
engine.switch_kb(name)

# 删除：先从列表移除
kb_names = engine.list_kb()
if selected_kb in kb_names:
    kb_names.remove(selected_kb)
    save_kb_names(kb_names)
engine.delete_kb(selected_kb)
```

---

## 3. 文件上传处理极慢

**问题**：上传文件处理非常慢，大文件（27MB TXT）耗时 14 分钟以上。

**根因分析**：

| 瓶颈 | 原因 | 影响 |
|------|------|------|
| 逐块去重 | 每个 chunk 调用 `has_content()` → ChromaDB `$contains` 全表扫描 | 100块=100次查询 |
| BGE-M3 CPU推理 | 5.67亿参数模型纯CPU运行 | 22k块≈40+分钟 |
| 一次性嵌入 | `embed_batch` 把22k文本全塞进GPU | 显存7GB濒临OOM |
| 分块太碎 | CHUNK_SIZE=500 | 块数过多 |
| 无进度反馈 | 用户盲等14分钟 | 体感极差 |

**解决方案**：

### 3.1 去重优化
- `vector_store.py`：`has_content()` 逐块查询 → `get_text_hashes()` 一次性拉取全部文本哈希，内存 set 去重
- N 次 ChromaDB 查询 → 1 次查询 + O(1) set 查找

### 3.2 GPU 加速
- 安装 CUDA 版 PyTorch：`pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128`
- `embedder.py`：加载模型时自动检测 CUDA，`SentenceTransformer(model_path, device="cuda")`
- RTX 4060 Laptop GPU 8GB，嵌入速度比 CPU 快 3-5 倍

### 3.3 分批嵌入
- `rag_engine.py`：不再一次性嵌入所有 chunks，每 500 条一批
- 每批嵌入完立即存入 ChromaDB，中间崩溃不丢数据
- GPU 显存峰值从 7GB 降至 2-3GB

### 3.4 分块参数调整
- `CHUNK_SIZE`：500 → 1000（块数减半）
- `CHUNK_OVERLAP`：50 → 100

### 3.5 进度条
- 引擎层新增 `progress_callback(step, total, message)` 参数
- UI 层通过 `st.status().update()` 实时显示进度百分比

---

## 4. Streamlit 文件上传 UI 英文残留

**问题**：文件上传区域的拖拽提示文字是英文。

**解决**：CSS 伪元素替换：
```css
[data-testid="stFileUploadDropzone"] [data-testid="stText"]::after {
    content: "拖拽文件到此处，或点击下方按钮选择文件";
}
```

---

## 5. Deploy 按钮和录制/截图菜单项

**问题**：Streamlit 右上角 Deploy 按钮和菜单中的 Record/Screenshot 项不需要。

**解决**：
- `.streamlit/config.toml` 设置 `toolbarMode = "viewer"`
- CSS `display: none` 隐藏录制/截图按钮

---

## 6. API Key 无法持久化

**问题**：每次刷新页面都要重新输入 API Key。

**解决**：`config.py` 中保存到 `.streamlit/saved_key`，启动时预填。

---

## 优化效果总结

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 27MB TXT 处理时间 | ~14 分钟（可能崩溃） | ~14 分钟（稳定+进度条） |
| GPU 显存占用 | 6-7 GB | 2-3 GB |
| 去重方式 | 逐块 ChromaDB 扫描 | 内存哈希 set |
| 崩溃恢复 | 全部丢失 | 已处理批次保留 |
| 进度可见性 | 无 | 实时百分比 |

---

## 环境依赖

- Python 3.14
- PyTorch 2.11+cu128（CUDA 版，必须 GPU）
- NVIDIA GPU + 驱动 >= 596
- BGE-M3 嵌入模型（本地 Modelscope 缓存）
- ChromaDB（持久化模式）
- Streamlit 1.55
