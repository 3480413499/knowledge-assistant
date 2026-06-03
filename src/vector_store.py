"""向量存储模块 — ChromaDB 封装，文档增删查."""
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.config import COLLECTION_NAME, DATA_DIR


class VectorStore:
    """ChromaDB 向量存储封装，支持多集合切换."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: Optional[str] = None,
    ):
        if persist_directory is None:
            persist_directory = str(DATA_DIR / "chromadb")

        self._persist_dir = persist_directory
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def switch_collection(self, name: str) -> None:
        """切换到指定集合，不存在则自动创建."""
        self._collection_name = name
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def list_collections(self) -> List[str]:
        """列出所有集合名称."""
        cols = self._client.list_collections()
        return sorted([c.name for c in cols])

    def delete_collection(self, name: str) -> None:
        """删除指定集合."""
        try:
            self._client.delete_collection(name)
        except Exception:
            pass

    def add(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """批量添加文档块到向量库."""
        if not chunks:
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "file": c.get("file", ""),
                "page": c.get("page", 0),
                "chunk_id": c.get("chunk_id", ""),
            }
            for c in chunks
        ]
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """向量相似度搜索，返回 top_k 条最相关结果."""
        if self.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "content": results["documents"][0][i],
                "file": results["metadatas"][0][i]["file"],
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i],
            })
        return items

    def count(self) -> int:
        """返回当前文档块总数."""
        return self._collection.count()

    def clear(self) -> None:
        """清空知识库."""
        if self.count() > 0:
            ids = self._collection.get()["ids"]
            self._collection.delete(ids=ids)

    def get_text_hashes(self) -> set:
        """返回已有文本的哈希集合（MD5前16位），内存友好的去重方案."""
        import hashlib
        if self.count() == 0:
            return set()
        data = self._collection.get(include=["documents"])
        if not data["documents"]:
            return set()
        return {hashlib.md5(d.encode()).hexdigest()[:16] for d in data["documents"]}

    def get_sources(self) -> List[str]:
        """返回所有不重复的源文件名列表."""
        if self.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])
        files = {m["file"] for m in all_meta["metadatas"] if m.get("file")}
        return sorted(files)
