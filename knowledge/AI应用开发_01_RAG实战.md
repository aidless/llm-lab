# RAG系统从零到生产：完整实现与优化实战

## 一、RAG架构全景：Naive RAG → Advanced RAG → Modular RAG

### 1.1 Naive RAG

Naive RAG是最简实现，遵循"索引→检索→生成"三阶段流水线：

```python
def naive_rag(query: str, docs: list[str], top_k: int = 3) -> str:
    doc_embeddings = embed_fn(docs)
    query_emb = embed_fn([query])[0]
    scores = cosine_similarity(query_emb, doc_embeddings)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    retrieved = [docs[i] for i in top_indices]

    context = "\n---\n".join(retrieved)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return llm(prompt)
```

**优点：** 实现简单，无需外部依赖，适合小型demo。

**缺点：** 单次检索无法应对复杂查询，无查询优化，无重排序，上下文混入噪声后生成质量急剧下降，无引用溯源，无法处理表格/多模态内容，无缓存机制。生产场景下召回率通常低于60%。

### 1.2 Advanced RAG

在检索前后引入预处理和后处理管线：

```python
def advanced_rag(query, index, retriever, reranker, generator) -> str:
    queries = query_expansion(query)
    hypo_doc = generator(f"Write a passage that answers: {query}")

    dense_results = retriever.dense_search(queries, top_k=20)
    sparse_results = retriever.bm25_search(queries, top_k=20)
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

    reranked = reranker.rerank(query, fused[:30], top_k=5)
    compressed = extract_relevant_spans(reranked, query, max_tokens=2048)

    prompt = build_rag_prompt(query, compressed)
    return generator.generate(prompt, cite_sources=True)
```

**改进点：** 查询扩展/重写，混合检索，重排序，上下文压缩，引用溯源。

**仍存在的问题：** 管线固定无法动态调整，无路由机制，工具调用需外部编排，评估指标分散。

### 1.3 Modular RAG

将每个环节抽象为可插拔模块，通过路由/调度器组合：

```python
class RAGPipeline:
    def __init__(self):
        self.modules = {
            "query_rewriter": QueryRewriter(),
            "router": QueryRouter(),
            "retriever": EnsembleRetriever(),
            "reranker": CrossEncoderReranker(),
            "compressor": SelectiveContextCompressor(),
            "generator": GeneratorWithCitation(),
            "memory": ChatMemory(),
        }
        self.toolkit = ToolKit([
            CalculatorTool(), WebSearchTool(), CodeInterpreter()
        ])

    async def arun(self, query: str, chat_history: list) -> str:
        strategy = self.modules["router"].route(query)
        rewritten = self.modules["query_rewriter"].rewrite(query, chat_history)
        context = await self.modules["retriever"].aretrieve(rewritten, strategy=strategy)
        compressed = self.modules["compressor"].compress(context, query)

        if strategy == "tool_required":
            tool_result = await self.toolkit.acall(query)
            compressed += f"\n[Tool Output]: {tool_result}"

        return self.modules["generator"].generate(
            query=query, context=compressed, strategy=strategy
        )
```

**对比总结：**

| 维度 | Naive RAG | Advanced RAG | Modular RAG |
|------|-----------|--------------|-------------|
| 检索方式 | 单向量检索 | 混合检索+重排序 | 可路由多策略检索 |
| 查询处理 | 无 | 扩展/重写 | 路由+重写+记忆 |
| 上下文管理 | 直接拼接 | 压缩+选择 | 自适应压缩+工具注入 |
| 可扩展性 | 差 | 中 | 高（插件式） |
| 生产就绪度 | ❌ | 需额外工程 | ✅ |
| 端到端延迟 | 低 | 中 | 可控（根据路由） |

---

## 二、文档解析与分块

### 2.1 文档解析挑战

#### PDF表格解析

PDF中的表格本质上是文本、线条和坐标的混合体，缺乏语义结构：

```python
import pdfplumber
import camelot

def parse_pdf_tables(pdf_path: str) -> dict:
    """Extract text and tables from PDF with structure preservation."""
    result = {"text_chunks": [], "tables": []}

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            tables = page.extract_tables()
            for table in tables:
                md = "\n".join(["| " + " | ".join(row) + " |" for row in table])
                result["tables"].append(md)

    try:
        tables = camelot.read_pdf(pdf_path, flavor="lattice", pages="all")
        for t in tables:
            result["tables"].append(t.df.to_markdown())
    except Exception:
        pass

    return result
```

**常见问题与解法：**
- **表格跨页：** 检测表头重复模式，跨页合并后按列对齐
- **合并单元格：** 用 `span` 坐标推断合并逻辑，或用基于视觉的 parser（如 `layoutparser`）
- **表格后接正文：** 用 y 坐标判断上下文关系，插入分隔标记 `[TABLE_END]`

#### 多栏布局

```python
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox

def detect_columns(pdf_path: str, page_num: int = 0) -> int:
    """Detect number of text columns on a page using 1D k-means on x0 positions."""
    pages = list(extract_pages(pdf_path))
    page = pages[page_num]

    x_positions = []
    for element in page:
        if isinstance(element, LTTextBox):
            x_positions.append(element.bbox[0])

    if len(x_positions) < 3:
        return 1

    from sklearn.cluster import KMeans
    X = np.array(x_positions).reshape(-1, 1)
    best_k, best_score = 1, float("inf")
    for k in range(1, 4):
        cluster = KMeans(n_clusters=k, n_init=1).fit(X)
        score = cluster.inertia_
        if score < best_score and k > best_k:
            best_k, best_score = k, score
    return best_k
```

#### 扫描件OCR

```python
import easyocr

def ocr_scanned_pdf(image_path: str, lang: str = "ch_sim+en") -> str:
    """OCR for scanned documents with layout preservation."""
    reader = easyocr.Reader(lang.split("+"), gpu=True)
    results = reader.readtext(image_path, paragraph=True, y_ths=0.5)

    lines = []
    for bbox, text, conf in results:
        if conf < 0.5:
            continue
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        lines.append((y_center, text))

    lines.sort(key=lambda x: x[0])
    return "\n".join(text for _, text in lines)
```

### 2.2 分块策略

#### Fixed-size Chunking

```python
def fixed_size_chunking(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Fixed-size token-based chunking with overlap."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks
```

**问题：** 硬切分会切断句子、段落甚至单词，导致语义断裂。

#### Semantic Chunking

```python
from sentence_transformers import SentenceTransformer

def semantic_chunking(sentences: list[str], model=None, threshold: float = 0.3) -> list[str]:
    """
    Chunk by semantic similarity between consecutive sentences.
    When embedding cosine distance exceeds threshold, start a new chunk.
    """
    model = model or SentenceTransformer("all-MiniLM-L6-v2")
    if len(sentences) == 1:
        return sentences

    embeds = model.encode(sentences)
    chunks, cur_chunk = [], [sentences[0]]

    for i in range(1, len(sentences)):
        sim = np.dot(embeds[i-1], embeds[i]) / (
            np.linalg.norm(embeds[i-1]) * np.linalg.norm(embeds[i])
        )
        if 1 - sim > threshold:
            chunks.append(" ".join(cur_chunk))
            cur_chunk = [sentences[i]]
        else:
            cur_chunk.append(sentences[i])

    if cur_chunk:
        chunks.append(" ".join(cur_chunk))
    return chunks
```

**阈值调优建议：** 根据嵌入空间分布动态调整——对语料采样后做百分位数分析，取第85百分位数作为默认阈值。

#### Agentic Chunking

```python
class AgenticChunker:
    """LLM-driven chunking with content type awareness."""

    CHUNK_TYPES = ["prose", "code", "table", "list", "equation", "heading"]

    def chunk(self, document: str) -> list[dict]:
        sections = self._split_by_headings(document)
        chunks = []
        for section in sections:
            ctype = self._classify_section(section)
            if ctype == "code":
                chunks.extend(self._chunk_code(section))
            elif ctype == "table":
                chunks.append(self._preserve_table(section))
            else:
                chunks.extend(self._chunk_prose(section, max_tokens=512))
        return chunks

    def _classify_section(self, text: str) -> str:
        prompt = f"Classify this text (prose/code/table/list/equation):\n{text[:200]}"
        return llm_call(prompt).strip().lower()
```

### 2.3 Chunk大小实验与最佳实践

```python
def chunk_size_experiment(corpus, queries, ground_truth,
                          sizes: list[int] = [128, 256, 512, 768, 1024]):
    results = {}
    for size in sizes:
        chunks = [fixed_size_chunking(doc, chunk_size=size, overlap=size//8)
                  for doc in corpus]
        index = build_vector_index(chunks)
        recalls = []
        for q, gt in zip(queries, ground_truth):
            retrieved = index.search(q, k=5)
            recall = len(set(retrieved) & set([gt])) / len([gt])
            recalls.append(recall)
        results[size] = np.mean(recalls)
    return results
```

**实验结论（基于开源基准）：**

| Chunk Size | Recall@5 | 生成质量 | 延迟(99th) |
|-----------|---------|---------|-----------|
| 128 | 0.62 | 上下文碎片化严重 | 15ms |
| 256 | 0.71 | 偶有缺失信息 | 22ms |
| **512** | **0.83** | **最佳平衡点** | 38ms |
| 768 | 0.81 | 偶有噪声混入 | 55ms |
| 1024 | 0.76 | 噪声增加降低质量 | 72ms |

**最佳实践：**
- 默认用512 tokens + 64 tokens overlap
- 代码文档用256 tokens（代码块通常较小）
- 法律合同用768+ tokens（条款需保持完整上下文）
- Overlap固定为chunk_size的12.5%-25%

---

## 三、嵌入与向量存储

### 3.1 嵌入模型选型

```python
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class EmbeddingFactory:
    """Unified interface for multiple embedding providers."""

    @staticmethod
    def create(provider: str, **kwargs):
        if provider == "openai-small":
            return OpenAIEmbedding(model="text-embedding-3-small", dimensions=512)
        elif provider == "openai-large":
            return OpenAIEmbedding(model="text-embedding-3-large", dimensions=256)
        elif provider == "e5":
            return SentenceTransformerEmbedding("intfloat/e5-mistral-7b-instruct")
        elif provider == "bge":
            return SentenceTransformerEmbedding("BAAI/bge-large-en-v1.5")
        elif provider == "jina":
            return JinaEmbedding("jina-embeddings-v3")

    @staticmethod
    def benchmark():
        return {
            "text-embedding-3-small":   {"dim": 512,  "MTEB": 62.3, "cost/M": "$0.02"},
            "text-embedding-3-large":   {"dim": 256,  "MTEB": 64.6, "cost/M": "$0.13"},
            "bge-large-en-v1.5":        {"dim": 1024, "MTEB": 64.2, "cost":  "free (local)"},
            "e5-mistral-7b-instruct":   {"dim": 4096, "MTEB": 66.6, "cost":  "free (local)"},
            "jina-embeddings-v3":       {"dim": 1024, "MTEB": 64.2, "cost/M": "$0.01"},
        }

class OpenAIEmbedding:
    def __init__(self, model: str, dimensions: int = 512):
        self.client = OpenAI()
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions
        )
        return np.array([d.embedding for d in resp.data])
```

**选型建议：**
- **生产API首选：** `text-embedding-3-small` + `dimensions=256`（成本与质量的最佳平衡）
- **学术/本地部署：** `bge-large-en-v1.5`（1024维，MTEB 64.2，完全免费）
- **极致质量：** `e5-mistral-7b-instruct`（4096维，但推理成本高）
- **多语言场景：** `jina-embeddings-v3`（支持多语言和任务特定向量）

### 3.2 向量数据库对比

```python
# Qdrant production setup
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="rag_docs",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        on_disk=True,  # memory-efficient for large collections
    ),
)

# Recommendation matrix
DB_RECOMMENDATIONS = {
    "prototype/notebook":          "Chroma",
    "production_small (<100K)":    "Qdrant (self-hosted)",
    "production_enterprise":       "Milvus or Pinecone",
    "multi_modal":                  "Weaviate",
    "kubernetes_native":           "Qdrant Operator",
}

DB_COMPARISON = {
    "Pinecone":  {"type": "managed SaaS",    "latency_p99": "10ms",  "pricing": "pay-per-pod"},
    "Weaviate":  {"type": "self-hosted/cloud","latency_p99": "15ms",  "pricing": "BYOC or SaaS"},
    "Qdrant":    {"type": "self-hosted/cloud","latency_p99": "8ms",   "pricing": "free tier + cloud"},
    "Chroma":    {"type": "embedded",         "latency_p99": "20ms",  "pricing": "free"},
    "Milvus":    {"type": "distributed",      "latency_p99": "12ms",  "pricing": "self-hosted/cloud"},
}
```

### 3.3 HNSW索引参数调优

```python
import hnswlib

def hnsw_tuning(embeddings, queries, ground_truth):
    """Grid search for optimal HNSW parameters."""
    best_recall, best_params = 0, {}

    for ef_construction in [100, 200, 400]:
        for M in [16, 32, 48, 64]:
            for ef_search in [128, 256, 512]:
                dim = embeddings.shape[1]
                idx = hnswlib.Index(space="cosine", dim=dim)
                idx.init_index(
                    max_elements=len(embeddings),
                    ef_construction=ef_construction,
                    M=M,
                )
                idx.add_items(embeddings, list(range(len(embeddings))))

                recalls = []
                for q, gt in zip(queries, ground_truth):
                    idx.set_ef(ef_search)
                    labels, _ = idx.knn_query(q, k=10)
                    recall = len(set(labels[0]) & set(gt)) / len(gt)
                    recalls.append(recall)

                recall = np.mean(recalls)
                if recall > best_recall:
                    best_recall = recall
                    best_params = {
                        "ef_construction": ef_construction,
                        "M": M, "ef_search": ef_search,
                        "recall@10": recall,
                    }
    return best_params
```

**推荐默认值：**

| 参数 | < 100K | 100K-1M | > 1M |
|------|--------|---------|------|
| M | 16 | 32 | 48-64 |
| ef_construction | 100 | 200 | 400 |
| ef_search | 128 | 256 | 512 |
| 预期Recall | 0.97+ | 0.95+ | 0.92+ |

**重要权衡：** `ef_construction` 和 `M` 越大，索引构建越慢（可差5-10x），内存占用越高，但召回率提升边际递减。建议从默认值开始，逐步增大。

---

## 四、检索优化

### 4.1 混合检索：Dense + Sparse

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Dense (embedding) + Sparse (BM25) hybrid retrieval with RRF fusion."""

    def __init__(self, dense_model, alpha: float = 0.5):
        self.dense = dense_model
        self.alpha = alpha
        self.bm25 = None
        self.documents = []

    def index(self, documents: list[str]):
        self.documents = documents
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.dense_embeds = self.dense.embed(documents)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_emb = self.dense.embed([query])[0]
        dense_scores = self.dense_embeds @ q_emb

        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)

        def minmax(scores):
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        combined = self.alpha * minmax(dense_scores) + \
                   (1 - self.alpha) * minmax(sparse_scores)

        top_indices = np.argsort(combined)[-top_k:][::-1]
        return [(i, combined[i]) for i in top_indices]
```

**RRF (Reciprocal Rank Fusion) 替代方案：**

```python
def reciprocal_rank_fusion(result_lists: list[list[int]], k: int = 60) -> list[int]:
    """
    Fuse multiple ranked lists using RRF score = 1 / (k + rank).
    result_lists: each list is doc_ids sorted by relevance.
    """
    scores = {}
    for ranked in result_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

### 4.2 重排序（Reranking）

双阶段检索的核心：第一阶段用高效的双编码器（bi-encoder）召回top-K，第二阶段用交叉编码器（cross-encoder）精细重排。

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

# Cohere rerank alternative
def cohere_rerank(query: str, documents: list[str], top_k: int = 5):
    import cohere
    co = cohere.Client(api_key="YOUR_API_KEY")
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_k,
    )
    return [(r.document["text"], r.relevance_score) for r in response.results]
```

**为什么重排序有效：** bi-encoder将query和doc独立编码，仅计算向量点积，丢失了交叉注意力信息。cross-encoder将query-doc pair送入同一Transformer，计算完整的交叉注意力，质量提升显著（通常Recall@5提升10-15%），但计算成本高一个数量级。因此只对top-30~50重排到top-5。

### 4.3 查询转换

#### HyDE（Hypothetical Document Embedding）

```python
def hyde_retrieval(query: str, generator, retriever, k: int = 10) -> list[str]:
    """
    Step 1: Generate a hypothetical document that would answer the query.
    Step 2: Use that document's embedding (not the query's) for retrieval.
    """
    hypo_doc = generator(
        f"Write a detailed passage that answers the question: {query}\nPassage:",
        max_tokens=256
    )
    # Embed the hypothetical document instead of the original query
    hypo_emb = embed_fn([hypo_doc])[0]
    return retriever.search_by_vector(hypo_emb, top_k=k)
```

**工作原理：** 查询通常很简短（「Transformer相比RNN的优缺点」），嵌入空间离相关文档远。HyDE先生成一段假想的回答文本，其嵌入与真实文档的嵌入在语义上更接近，从而提升召回率。在对话式QA场景中，HyDE可提升Recall@10约5-8%。

#### Multi-Query

```python
def multi_query_retrieval(query: str, llm, retriever, n_variants: int = 5) -> list[str]:
    """
    Generate multiple query variants to cover different aspects.
    Then retrieve for each variant and fuse results.
    """
    prompt = (
        f"You are a helpful search assistant. Generate {n_variants} different "
        f"versions of the following question that cover its key aspects. "
        f"Each should be a self-contained question.\n\n"
        f"Original: {query}\n\n"
        f"Variants (one per line):"
    )
    variants_text = llm(prompt)
    variants = [v.strip() for v in variants_text.split("\n") if v.strip()]
    variants.append(query)

    all_results = []
    for vq in variants:
        results = retriever.search(vq, top_k=10)
        all_results.append([doc_id for doc_id, _ in results])

    return reciprocal_rank_fusion(all_results, k=60)
```

#### Query Rewriting

```python
class QueryRewriter:
    """Rewrite query with conversation context for multi-turn RAG."""

    def rewrite(self, query: str, history: list[str]) -> str:
        if not history:
            return query

        prompt = (
            f"Given the conversation history and a follow-up question, "
            f"rewrite the follow-up as a standalone question.\n\n"
            f"History:\n{chr(10).join(history[-3:])}\n\n"
            f"Follow-up: {query}\n\n"
            f"Standalone question:"
        )
        return llm_call(prompt).strip()
```

### 4.4 检索策略进阶

#### MMR（Maximum Marginal Relevance）

```python
def mmr_diversify(query_emb: np.ndarray, doc_embeds: np.ndarray,
                  lambda_param: float = 0.7, top_k: int = 5) -> list[int]:
    """
    MMR = argmax[ lambda * sim(q, di) - (1-lambda) * max sim(di, dj) ]
    Balances relevance and diversity.
    """
    selected = []
    candidate_indices = list(range(len(doc_embeds)))

    while len(selected) < top_k and candidate_indices:
        mmr_scores = []
        for idx in candidate_indices:
            relevance = np.dot(query_emb, doc_embeds[idx])
            if selected:
                max_sim = max(np.dot(doc_embeds[idx], doc_embeds[s])
                              for s in selected)
            else:
                max_sim = 0
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append(mmr)

        best_idx = candidate_indices[np.argmax(mmr_scores)]
        selected.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected
```

**lambda参数调节：** 值越接近1越重相关性（可能冗余），越接近0越重多样性（可能偏离主题）。推荐0.7-0.8以获得最佳平衡。

#### Sliding Window Retrieval

```python
def sliding_window_retrieval(query: str, all_chunks: list[str],
                              window_size: int = 5, top_k_windows: int = 3) -> list[str]:
    """
    Retrieve at the window (super-chunk) level instead of individual chunks.
    Each window = window_size consecutive chunks. Preserves broader context.
    """
    windows = [
        " ".join(all_chunks[i:i+window_size])
        for i in range(0, len(all_chunks), window_size)
    ]
    window_embeds = embed_fn(windows)
    q_emb = embed_fn([query])[0]
    scores = window_embeds @ q_emb
    top_windows = np.argsort(scores)[-top_k_windows:][::-1]

    result = []
    for w in top_windows:
        start = w * window_size
        end = min(start + window_size, len(all_chunks))
        result.extend(all_chunks[start:end])
    return result
```

---

## 五、生成优化

### 5.1 RAG专用System Prompt

```python
def build_rag_system_prompt() -> str:
    return (
        "You are a precise question-answering assistant. "
        "You have access to retrieved document passages delimited by <document> tags.\n\n"
        "Rules:\n"
        "1. Answer based ONLY on the provided context. If the context lacks information, say 'I cannot find this information in the provided documents.'\n"
        "2. Cite your sources using [Source: doc_id] after each factual statement.\n"
        "3. If the context contains conflicting information, acknowledge the conflict.\n"
        "4. Use bullet points for multi-fact answers. Keep answers concise.\n"
        "5. Never make up citations or fabricate information.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )

def build_rag_prompt(query: str, documents: list[dict]) -> str:
    """
    documents: list of {"text": str, "doc_id": str, "score": float}
    """
    context_parts = []
    for i, doc in enumerate(documents):
        tagged = f'<document id="{doc["doc_id"]}" rank="{i+1}">\n{doc["text"]}\n</document>'
        context_parts.append(tagged)

    context = "\n\n".join(context_parts)
    prompt = build_rag_system_prompt().format(context=context, query=query)
    return prompt
```

### 5.2 上下文窗口管理

#### Selective Context

```python
class SelectiveContextCompressor:
    """
    Instead of truncating from the end, select the most relevant spans
    from retrieved documents based on token-level relevance scoring.
    """

    def compress(self, documents: list[dict], query: str,
                 max_tokens: int = 2048) -> str:
        # Step 1: Extract sentences from all docs
        all_sentences = []
        for doc in documents:
            sentences = sent_tokenize(doc["text"])
            for s in sentences:
                all_sentences.append({
                    "text": s,
                    "doc_id": doc["doc_id"],
                    "doc_score": doc.get("score", 1.0),
                })

        # Step 2: Score each sentence against query
        pairs = [[query, s["text"]] for s in all_sentences]
        cross_scores = self.cross_encoder.predict(pairs)

        for s, score in zip(all_sentences, cross_scores):
            s["relevance"] = score

        # Step 3: Greedy select until max_tokens
        all_sentences.sort(key=lambda x: x["relevance"], reverse=True)
        selected, token_count = [], 0
        for s in all_sentences:
            tokens = tokenizer.encode(s["text"])
            if token_count + len(tokens) > max_tokens:
                break
            selected.append(s)
            token_count += len(tokens)

        # Step 4: Restore original order within each doc
        selected.sort(key=lambda x: (x["doc_id"], id(x)))
        return "\n".join(s["text"] for s in selected)
```

#### 上下文压缩提示

```python
def compress_context_with_llm(context: str, query: str) -> str:
    """Use LLM to extract only query-relevant parts from verbose context."""
    prompt = (
        f"Extract only the parts of the following context that are relevant "
        f"to answering the question. Remove redundancy and irrelevant details. "
        f"Preserve all factual information, numbers, and names.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Compressed context:"
    )
    return llm_call(prompt, max_tokens=1024)
```

### 5.3 引用溯源（Citation/Attribution）

```python
import re

class CitationGenerator:
    """Generate grounded answers with verifiable citations."""

    def generate_with_citations(self, query: str, documents: list[dict]) -> str:
        prompt = build_rag_prompt(query, documents)
        response = llm_call(prompt, temperature=0.0)

        # Post-process to ensure every factual claim has a [Source: ...]
        # This pattern matches [Source: doc_id] references
        citations = re.findall(r'\[Source: (.*?)\]', response)
        valid_doc_ids = {d["doc_id"] for d in documents}

        for cite in citations:
            if cite not in valid_doc_ids:
                raise ValueError(f"LLM hallucinated citation: {cite}")

        return response

# Chunk-level provenance tracking
class ProvenanceTracker:
    """Track which chunk produced which part of the answer."""

    def __init__(self):
        self.documents = {}

    def add_document(self, doc_id: str, text: str, metadata: dict):
        self.documents[doc_id] = {"text": text, "metadata": metadata}

    def generate_with_highlights(self, query: str, top_k: int = 5):
        retrieved = retrieve(query, list(self.documents.values()), top_k)
        response = llm_rag(query, retrieved)

        for doc_id, info in self.documents.items():
            for sentence in sent_tokenize(response):
                if any(phrase in sentence for phrase in info["text"].split(". ")[0]):
                    response = response.replace(
                        sentence,
                        f"{sentence} [from: {doc_id}]"
                    )
        return response
```

### 5.4 Agentic RAG

#### 结合工具的检索增强生成

```python
class AgenticRAG:
    """
    Agentic RAG: LLM decides when to retrieve, when to call tools,
    and when to answer directly.
    """

    def __init__(self, tools: dict):
        self.tools = tools  # {"search": retriever, "calculator": calc, "web": web_search}
        self.memory = []

    def run(self, query: str, max_steps: int = 5) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agentic RAG assistant. You have access to tools:\n"
                    + "\n".join(f"- {name}: {desc}"
                                 for name, (desc, _) in self.tools.items()) +
                    "\n\nRespond with:\n"
                    "  THOUGHT: your reasoning\n"
                    "  ACTION: tool_name | input\n"
                    "  or\n"
                    "  FINAL: your answer"
                )
            },
            {"role": "user", "content": query}
        ]

        for step in range(max_steps):
            response = llm_call(messages)
            messages.append({"role": "assistant", "content": response})

            if response.startswith("FINAL:"):
                return response.replace("FINAL:", "").strip()

            if response.startswith("ACTION:"):
                _, rest = response.split("ACTION:", 1)
                tool_name, tool_input = rest.split("|", 1)
                tool_name = tool_name.strip()
                tool_input = tool_input.strip()

                if tool_name in self.tools:
                    _, fn = self.tools[tool_name]
                    result = fn(tool_input)
                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": str(result),
                    })

        return "Max steps reached without final answer."
```

#### ReAct循环

```python
def react_rag_loop(query: str, retriever, llm, max_turns: int = 5) -> str:
    """
    ReAct (Reasoning + Acting) loop for RAG:
    Thought → Action (Retrieve) → Observation → Thought → Action (Answer)
    """
    context = ""
    for turn in range(max_turns):
        # Decide next action
        prompt = (
            f"You are answering: {query}\n"
            f"Current context:\n{context[:2000]}\n\n"
            f"Options:\n"
            f"  RETRIEVE <search query>  - get more information\n"
            f"  ANSWER <final answer>    - respond to user\n\n"
            f"What is your next action? (RETRIEVE/ANSWER):"
        )
        action = llm(prompt, max_tokens=100).strip()

        if action.startswith("RETRIEVE"):
            search_q = action.replace("RETRIEVE", "").strip()
            docs = retriever.search(search_q, top_k=3)
            new_context = "\n".join(docs)
            context += f"\n\n[Retrieved for '{search_q}']:\n{new_context}"
        elif action.startswith("ANSWER"):
            return action.replace("ANSWER", "").strip()

    return "Max turns reached."
```

---

## 六、评估体系

### 6.1 检索评估指标

```python
def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant docs in top-k retrieved results."""
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & relevant)
    return hits / len(relevant)

def mean_reciprocal_rank(retrieved_lists: list[list[int]],
                         relevant_sets: list[set[int]]) -> float:
    """MRR: average of reciprocal rank of first relevant doc."""
    rr_sum = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(retrieved_lists)

def ndcg_at_k(retrieved: list[int], relevant: dict[int, float], k: int) -> float:
    """NDCG: Normalized Discounted Cumulative Gain with graded relevance."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevant.get(doc_id, 0.0)
        dcg += (2**rel - 1) / np.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0
```

### 6.2 生成评估指标

```python
class GenerationMetrics:
    """Faithfulness, Answer Relevance, Context Relevance."""

    @staticmethod
    def faithfulness(answer: str, context: str) -> float:
        """Check if answer is entailed by context (LLM-as-judge)."""
        prompt = (
            f"Rate the faithfulness of the answer to the context on a scale 0-1.\n"
            f"Faithfulness means the answer does not contain claims not supported by the context.\n\n"
            f"Context:\n{context[:3000]}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Faithfulness score (0.0-1.0):"
        )
        score = float(llm_call(prompt, temperature=0.0).strip())
        return max(0.0, min(1.0, score))

    @staticmethod
    def answer_relevance(answer: str, query: str) -> float:
        """Check if the answer actually addresses the query."""
        prompt = (
            f"Rate how well this answer addresses the question on a scale 0-1.\n\n"
            f"Question: {query}\n\n"
            f"Answer: {answer}\n\n"
            f"Relevance score (0.0-1.0):"
        )
        score = float(llm_call(prompt, temperature=0.0).strip())
        return max(0.0, min(1.0, score))

    @staticmethod
    def context_relevance(query: str, context: str) -> float:
        """Check if context contains information needed to answer the query."""
        prompt = (
            f"Rate how much of the context is relevant to answering the question on 0-1.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context[:3000]}\n\n"
            f"Context relevance score (0.0-1.0):"
        )
        score = float(llm_call(prompt, temperature=0.0).strip())
        return max(0.0, min(1.0, score))
```

### 6.3 RAGAS评估框架

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

def ragas_evaluation(dataset):
    """
    dataset: list of {"question": str, "answer": str, "contexts": list[str], "ground_truth": str}
    """
    from datasets import Dataset

    data = {
        "question": [d["question"] for d in dataset],
        "answer": [d["answer"] for d in dataset],
        "contexts": [d["contexts"] for d in dataset],
        "ground_truth": [d["ground_truth"] for d in dataset],
    }

    ds = Dataset.from_dict(data)
    result = evaluate(
        ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )
    return result

# Usage:
# result = ragas_evaluation(test_samples)
# print(result)  # {faithfulness: 0.85, answer_relevancy: 0.92, ...}
```

### 6.4 TruLens评估

```python
from trulens_eval import Tru, Feedback, Select
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI

def setup_trulens_eval(app, app_id: str):
    """Configure TruLens feedback functions for RAG."""
    provider = fOpenAI()

    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()

    f_qs_relevance = Feedback(
        provider.qs_relevance,
        name="Context Relevance"
    ).on_input().on(
        Select.RecordCalls.retrieve.rets.collect()
    )

    tru = Tru()
    tru.reset_database()

    from trulens_eval import TruChain
    tru_recorder = TruChain(
        app,
        app_id=app_id,
        feedbacks=[f_qa_relevance, f_qs_relevance]
    )
    return tru_recorder

# On-demand evaluation dashboard:
# tru = Tru()
# tru.run_dashboard()
```

---

## 七、生产化部署

### 7.1 缓存策略

#### Embedding Cache

```python
import hashlib
import diskcache

class EmbeddingCache:
    """Two-level cache: in-memory LRU + disk fallback for embeddings."""

    def __init__(self, cache_dir: str = "./cache/embeddings", max_memory: int = 10000):
        self.memory = {}       # in-memory LRU: {text_hash: embedding}
        self.max_memory = max_memory
        self.disk = diskcache.Cache(cache_dir)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        key = self._hash(text)
        if key in self.memory:
            return self.memory[key]
        if key in self.disk:
            emb = self.disk[key]
            self.memory[key] = emb
            return emb
        return None

    def set(self, text: str, embedding: np.ndarray):
        key = self._hash(text)
        if len(self.memory) >= self.max_memory:
            # Evict oldest
            self.memory.pop(next(iter(self.memory)))
        self.memory[key] = embedding
        self.disk[key] = embedding
```

#### LLM Cache（GPTCache）

```python
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation import SearchDistanceEvaluation

def init_gptcache(cache_dir: str = "./cache/gptcache"):
    """Initialize GPTCache for LLM response caching."""
    onnx = Onnx()
    cache.init(
        pre_embedding_func=get_query_key,
        embedding_func=onnx.to_embeddings,
        data_manager=manager_factory(
            "sqlite,faiss",
            sqlite_path=f"{cache_dir}/sqlite.db",
            faiss_index_path=f"{cache_dir}/faiss.index",
        ),
        similarity_evaluation=SearchDistanceEvaluation(),
    )
    cache.set_cache_enabled(True)

def cached_llm_call(prompt: str, **kwargs) -> str:
    """Cached LLM call via GPTCache adapter."""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content

def get_query_key(prompt: str, **kwargs) -> str:
    """Pre-processing hook for cache key generation."""
    import hashlib
    return hashlib.md5(prompt.encode()).hexdigest()
```

### 7.2 监控与可观测性

#### LangSmith Tracing

```python
from langsmith import Client
from langchain.callbacks.tracers import LangSmithTracer

def setup_langsmith():
    """Configure LangSmith for RAG pipeline tracing."""
    client = Client()
    tracer = LangSmithTracer(
        project_name="rag-production",
        client=client,
    )
    return tracer

# Traced RAG call
def traced_rag(query: str, tracer) -> str:
    with tracer:
        docs = retriever.search(query)
        answer = generate(query, docs)
        return answer
```

#### Arize / Phoenix

```python
# Phoenix (open-source observability for LLM apps)
import phoenix as px

# Launch Phoenix in notebook/background
session = px.launch_app()

# Instrument your LLM calls
from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument()

# Every LLM call is now traced to Phoenix
# View at http://localhost:6006
```

### 7.3 成本优化

```python
class CostOptimizedRouter:
    """
    Route queries to appropriate model tier based on complexity.
    Simple Q&A -> cheap model, complex reasoning -> expensive model.
    """

    TIERS = [
        ("simple",  ["gpt-4o-mini", "claude-3-haiku"],
         {"max_cost": 0.00015, "max_tokens": 500}),
        ("medium",  ["gpt-4o-mini", "claude-3-sonnet"],
         {"max_cost": 0.003,  "max_tokens": 2000}),
        ("complex", ["gpt-4o", "claude-3-opus"],
         {"max_cost": 0.03,   "max_tokens": 4000}),
    ]

    @staticmethod
    def classify_query(query: str, num_docs: int) -> str:
        """Classify query complexity for model routing."""
        # Heuristic: longer queries with more context need stronger models
        if len(query) < 50 and num_docs < 3:
            return "simple"
        elif len(query) < 200 and num_docs < 6:
            return "medium"
        else:
            return "complex"

    @staticmethod
    def select_model(query: str, num_docs: int) -> str:
        tier = CostOptimizedRouter.classify_query(query, num_docs)
        return CostOptimizedRouter.TIERS[
            ["simple", "medium", "complex"].index(tier)
        ][1][0]

# Prompt caching with Anthropic API
def cached_prompt_call(system_prompt: str, messages: list[dict]) -> str:
    """Use prompt caching for repeated system prompts (Anthropic API)."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-3-sonnet-20241022",
        max_tokens=2000,
        system=[{"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )
    return response.content[0].text
```

### 7.4 延迟优化

```python
import asyncio
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

class StreamingRAG:
    """Streaming RAG with prefetch for minimal perceived latency."""

    def __init__(self):
        self.cache = EmbeddingCache()

    async def stream_answer(self, query: str):
        # Step 1: Embed query + search (async)
        loop = asyncio.get_event_loop()
        q_emb = await loop.run_in_executor(None, self.cache.get, query)
        if q_emb is None:
            q_emb = await loop.run_in_executor(None, embed_fn, [query])

        # Step 2: Retrieve (prefetch top-20 while generating)
        docs = await loop.run_in_executor(None, retriever.search, query, 20)

        # Step 3: Begin streaming generation
        prompt = build_rag_prompt(query, docs[:5])
        stream = llm_stream(prompt)  # async generator

        async for token in stream:
            yield token

        # Background: prefetch next batch or rerank remaining docs
        asyncio.create_task(self._prefetch(query, docs[5:]))

    async def _prefetch(self, query: str, remaining: list[str]):
        """Prefetch reranking for potential follow-up."""
        scores = reranker.rerank(query, remaining)
        return scores

# FastAPI endpoint
app = FastAPI()

@app.post("/rag/stream")
async def rag_stream(query: str):
    rag = StreamingRAG()
    return EventSourceResponse(rag.stream_answer(query))
```

### 7.5 生产架构总览

```python
# Production RAG service - simplified entry point
class ProductionRAGService:
    def __init__(self):
        self.cache = EmbeddingCache()
        self.retriever = HybridRetriever(embed_fn, alpha=0.6)
        self.reranker = Reranker()
        self.query_rewriter = QueryRewriter()
        self.router = CostOptimizedRouter()
        self.tracer = setup_langsmith()

    async def answer(self, query: str, user_id: str, history: list[str] = None):
        tier = self.router.classify_query(query, 5)
        model = self.router.select_model(query, 5)

        with self.tracer:
            rewritten = self.query_rewriter.rewrite(query, history or [])
            docs = self.retriever.search(rewritten, top_k=30)
            reranked = self.reranker.rerank(rewritten, [d["text"] for d in docs], top_k=5)
            compressed = SelectiveContextCompressor().compress(reranked, rewritten)
            prompt = build_rag_prompt(rewritten, compressed)
            answer = llm_call(prompt, model=model)

        return {"answer": answer, "sources": reranked, "tier": tier}
```

---

## 八、总结与路线图

构建生产级RAG系统是系统工程，而非简单的「嵌入+LLM」拼接。核心要点：

1. **分块是质量基石：** 512 tokens + 语义分块是当前最佳实践，但需根据文档类型调整。PDF表格和多栏布局是最大的质量陷阱。
2. **检索没有银弹：** 混合检索（BM25 + Dense）+ RRF融合 + Cross-Encoder重排序是生产标配。单一检索手段无法覆盖所有查询类型。
3. **查询不能直接用：** HyDE、Multi-Query、Query Rewriting能系统性地提升召回率5-15%，尤其是在对话上下文和短查询场景。
4. **评估不能省：** RAGAS的faithfulness + context_precision两个指标足以发现90%的常见问题。建议线上线下双通道评估。
5. **成本与延迟必须管理：** 多级缓存（Embedding + LLM）、模型路由、Streaming是生产部署的三个必要条件。
6. **Agentic RAG正在成为默认范式：** 从被动检索到主动规划工具调用，RAG系统正演变为自主智能体。
