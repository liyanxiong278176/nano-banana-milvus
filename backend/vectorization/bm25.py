# 修复 Windows 控制台编码问题
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
BM25 向量化器模块

BM25 是 TF-IDF 的改进版本，在信息检索领域被广泛使用。
相比 TF-IDF，BM25 引入了：
1. 词频饱和函数：防止高频词过度影响得分
2. 文档长度归一化：补偿不同文档长度的影响

【公式】
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))

其中：
- f(qi, D): 词 qi 在文档 D 中的词频
- |D|: 文档 D 的长度
- avgdl: 平均文档长度
- k1: 词频饱和参数 (默认 1.5)
- b: 长度归一化参数 (默认 0.75)

【与 TF-IDF 的对比】
- TF-IDF: 词频线性增长，无长度归一化
- BM25: 词频饱和，有长度归一化，通常提升 5-15% 检索效果
"""
import re
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

# 支持两种实现：纯 Python 和 rank_bm25 库
try:
    from rank_bm25 import BM25Okapi, BM25L, BM25Plus
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    print("提示: rank_bm25 未安装，使用纯 Python 实现")


def tokenize(text: str) -> List[str]:
    """
    简单的分词器

    Args:
        text: 输入文��

    Returns:
        分词列表
    """
    # 转小写
    text = text.lower()
    # 移除特殊字符，保留字母数字和空格
    text = re.sub(r'[^\w\s]', ' ', text)
    # 分词
    return text.split()


class BM25Vectorizer:
    """
    BM25 向量化器

    兼容 scikit-learn TfidfVectorizer 的 API 设计
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        tokenizer=None,
        max_features: int = None
    ):
        """
        初始化 BM25 向量化器

        Args:
            k1: 词频饱和参数 (1.0-2.0)
                - k1 = 0: 不考虑词频（相当于二元模型）
                - k1 = 1.5: 默认值，平衡
                - k1 > 2: 更重视词频
            b: 长度归一化参数 (0-1)
                - b = 0: 不进行长度归一化
                - b = 0.75: 默认值
                - b = 1: 完全归一化
            epsilon: IDF 下界，防止除零
            tokenizer: 分词函数，默认使用简单分词
            max_features: 最大特征数（词汇表大小限制）
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.tokenizer = tokenizer or tokenize
        self.max_features = max_features

        # 训练后的数据
        self.corpus = []           # 分词后的语料
        self.vocab = {}            # 词汇表 {词: 索引}
        self.idf = {}              # IDF 值 {词: IDF}
        self.doc_len = []          # 文档长度列表
        self.avgdl = 0.0           # 平均文档长度
        self.N = 0                 # 文档总数

        # 纯 Python 实现时存储词频
        self.doc_freqs = []        # 每个文档的词频统计

        # rank_bm25 库实现
        self.bm25_model = None

    def _build_vocab(self, corpus: List[List[str]]):
        """
        构建词汇表

        Args:
            corpus: 分词后的语料
        """
        # 统计词频
        word_freq = defaultdict(int)
        for doc in corpus:
            for word in set(doc):
                word_freq[word] += 1

        # 如果限制特征数，取最常见的词
        if self.max_features:
            sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
            vocab_words = [w for w, _ in sorted_words[:self.max_features]]
        else:
            vocab_words = list(word_freq.keys())

        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}

    def _calculate_idf(self):
        """
        计算 IDF 值

        IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)

        其中 df(qi) 是包含词 qi 的文档数
        """
        # 统计每个词的文档频率
        df = defaultdict(int)
        for doc in self.corpus:
            for word in set(doc):
                if word in self.vocab:
                    df[word] += 1

        # 计算 IDF
        for word, idx in self.vocab.items():
            idf = np.log((self.N - df[word] + 0.5) / (df[word] + 0.5) + 1)
            self.idf[word] = max(idf, self.epsilon)

    def fit(self, raw_documents: List[str]):
        """
        训练 BM25 模型

        Args:
            raw_documents: 原始文档列表

        Returns:
            self
        """
        # 分词
        self.corpus = [self.tokenizer(doc) for doc in raw_documents]

        # 统计
        self.N = len(self.corpus)
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 0

        # 构建词汇表
        self._build_vocab(self.corpus)

        # 计算 IDF
        self._calculate_idf()

        # 如果使用 rank_bm25 库
        if RANK_BM25_AVAILABLE:
            # 只使用词汇表内的词
            filtered_corpus = []
            for doc in self.corpus:
                filtered = [w for w in doc if w in self.vocab]
                filtered_corpus.append(filtered)
            self.bm25_model = BM25Okapi(filtered_corpus, k1=self.k1, b=self.b)

        return self

    def _get_scores_python(self, query: List[str]) -> np.ndarray:
        """
        纯 Python 实现 BM25 得分计算

        Args:
            query: 分词后的查询

        Returns:
            每个文档的 BM25 得分
        """
        scores = np.zeros(self.N)

        for i, doc in enumerate(self.corpus):
            doc_len = self.doc_len[i]
            doc_freqs = defaultdict(int)
            for word in doc:
                if word in self.vocab:
                    doc_freqs[word] += 1

            for q in query:
                if q not in self.vocab or q not in doc_freqs:
                    continue

                f = doc_freqs[q]  # 词频
                idf = self.idf.get(q, self.epsilon)

                # BM25 公式
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += idf * numerator / denominator

        return scores

    def transform(self, raw_documents: List[str]) -> List[Dict[int, float]]:
        """
        将文档转换为稀疏向量格式

        Args:
            raw_documents: 原始文档列表

        Returns:
            稀疏向量列表，每个为 {索引: 值} 格式
            兼容 Milvus SparseFloatVector 字段
        """
        if RANK_BM25_AVAILABLE and self.bm25_model is not None:
            # 使用 rank_bm25 库
            result = []
            for doc in raw_documents:
                query = self.tokenizer(doc)
                # 只使用词汇表内的词
                query = [q for q in query if q in self.vocab]
                if query:
                    scores = self.bm25_model.get_scores(query)
                    # 将得分转换为稀疏向量格式
                    sparse_vec = {i: float(s) for i, s in enumerate(scores) if s > 0}
                else:
                    sparse_vec = {}
                result.append(sparse_vec)
            return result
        else:
            # 纯 Python 实现
            result = []
            for doc in raw_documents:
                query = self.tokenizer(doc)
                query = [q for q in query if q in self.vocab]
                if query:
                    scores = self._get_scores_python(query)
                    sparse_vec = {i: float(s) for i, s in enumerate(scores) if s > 0}
                else:
                    sparse_vec = {}
                result.append(sparse_vec)
            return result

    def fit_transform(self, raw_documents: List[str]) -> List[Dict[int, float]]:
        """
        训练并转换

        Args:
            raw_documents: 原始文档列表

        Returns:
            稀疏向量列表
        """
        return self.fit(raw_documents).transform(raw_documents)

    def get_feature_names_out(self) -> List[str]:
        """
        获取特征名称（词汇表）

        Returns:
            词汇列表
        """
        return list(self.vocab.keys())

    @property
    def vocabulary_(self) -> Dict[str, int]:
        """
        兼容 TfidfVectorizer 的 vocabulary_ 属性
        """
        return self.vocab


class BM25EmbeddingGenerator:
    """
    BM25 嵌入生成器

    用于替换原有的 TfidfVectorizer
    设计为与 EmbeddingGenerator 类配合使用
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: int = 500,
        tokenizer=None
    ):
        """
        初始化 BM25 嵌入生成器

        Args:
            k1: 词频饱和参数
            b: 长度归一化参数
            max_features: 最大词汇量
            tokenizer: 分词器
        """
        self.vectorizer = BM25Vectorizer(
            k1=k1,
            b=b,
            max_features=max_features,
            tokenizer=tokenizer
        )

    def build_vectorizer(self, products: List[Dict]) -> BM25Vectorizer:
        """
        构建 BM25 向量化器

        Args:
            products: 商品元数据列表

        Returns:
            BM25Vectorizer 对象
        """
        descriptions = [p.get("description", "") for p in products]
        self.vectorizer.fit(descriptions)

        vocab_size = len(self.vectorizer.vocab)
        print(f"BM25 词汇表大小: {vocab_size}")

        return self.vectorizer

    def generate_sparse_vectors(
        self,
        products: List[Dict],
        vectorizer: BM25Vectorizer = None
    ) -> List[Dict[int, float]]:
        """
        生成文本稀疏向量 (BM25)

        Args:
            products: 商品元数据列表
            vectorizer: BM25Vectorizer 对象，如果为 None 则会新建

        Returns:
            稀疏向量列表 [{index: value}, ...]
        """
        if vectorizer is None:
            vectorizer = self.build_vectorizer(products)
        else:
            self.vectorizer = vectorizer

        descriptions = [p.get("description", "") for p in products]
        sparse_vectors = self.vectorizer.transform(descriptions)

        # 统计非零项数量
        non_zero_counts = [len(v) for v in sparse_vectors]
        avg_non_zero = sum(non_zero_counts) / len(non_zero_counts) if non_zero_counts else 0
        print(f"BM25 稀疏向量: {len(sparse_vectors)} 个, 平均非零项: {avg_non_zero:.1f}")

        return sparse_vectors


def encode_new_product_bm25(
    new_product: Dict,
    vectorizer: BM25Vectorizer
) -> Dict[int, float]:
    """
    对新品进行 BM25 编码

    Args:
        new_product: 新品元数据
        vectorizer: 已训练的 BM25Vectorizer

    Returns:
        稀疏向量 {索引: BM25分数}
    """
    # 构建查询文本
    query_text = (
        f"{new_product.get('category', '')} "
        f"{new_product.get('style', '')} "
        f"{new_product.get('season', '')} "
        f"{new_product.get('color', '')} "
        f"{new_product.get('description', '')}"
    ).strip()

    # 转换为稀疏向量
    result = vectorizer.transform([query_text])
    sparse_vec = result[0] if result else {}

    # 打印日志
    print(f"  [BM25] 编码新品: {new_product.get('category', '')}/{new_product.get('style', '')}, 非零项: {len(sparse_vec)}")

    return sparse_vec


# ==================== 便捷函数 ====================

def create_bm25_vectorizer(
    k1: float = 1.5,
    b: float = 0.75,
    max_features: int = 500
) -> BM25Vectorizer:
    """
    创建 BM25 向量化器的便捷函数

    Args:
        k1: 词频饱和参数
        b: 长度归一化参数
        max_features: 最大词汇量

    Returns:
        BM25Vectorizer 实例
    """
    return BM25Vectorizer(k1=k1, b=b, max_features=max_features)


# ==================== 主函数（测试）====================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BM25 向量化器测试")
    print("=" * 60)

    # 测试数据
    documents = [
        "elegant floral midi dress for summer",
        "casual denim top with vintage style",
        "formal black maxi dress for evening",
        "sporty yoga pants for workout",
        "cozy knit sweater for autumn"
    ]

    # 测试查询
    query = "elegant summer dress"

    print(f"\n文档数: {len(documents)}")
    print(f"查询: {query}")

    # 创建 BM25 向量化器
    print("\n[1/3] 创建 BM25 向量化器...")
    vectorizer = create_bm25_vectorizer(k1=1.5, b=0.75, max_features=100)

    # 训练
    print("[2/3] 训练模型...")
    vectorizer.fit(documents)

    print(f"  词汇表大小: {len(vectorizer.vocab)}")
    print(f"  平均文档长度: {vectorizer.avgdl:.2f}")

    # 转换
    print("[3/3] 转换文档为稀疏向量...")
    sparse_vectors = vectorizer.transform(documents)

    print(f"  稀疏向量数量: {len(sparse_vectors)}")
    for i, vec in enumerate(sparse_vectors):
        non_zero = len(vec)
        print(f"    文档 {i}: {non_zero} 个非零项")

    # 查询测试
    print("\n查询��果:")
    query_scores = vectorizer.transform([query])[0]
    sorted_scores = sorted(query_scores.items(), key=lambda x: -x[1])[:5]

    for idx, score in sorted_scores:
        print(f"  文档 {idx}: {score:.4f} - {documents[idx][:50]}...")

    # 与 TF-IDF 对比
    print("\n" + "=" * 60)
    print("与 TF-IDF 对比")
    print("=" * 60)

    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(max_features=100)
    tfidf.fit(documents)
    tfidf_scores = tfidf.transform([query]).toarray()[0]

    # TF-IDF 转换为稀疏格式
    tfidf_sparse = {i: float(s) for i, s in enumerate(tfidf_scores) if s > 0}
    tfidf_sorted = sorted(tfidf_sparse.items(), key=lambda x: -x[1])[:5]

    print("\nTF-IDF Top 5:")
    for idx, score in tfidf_sorted:
        if idx < len(documents):
            print(f"  文档 {idx}: {score:.4f} - {documents[idx][:50]}...")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
