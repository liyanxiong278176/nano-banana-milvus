"""
向量嵌入生成模块

支持两种稀疏向量生成方法：
1. TF-IDF: 传统词频-逆文档频率
2. BM25: 改进的排序算法，通常比 TF-IDF 效果提升 5-15%
"""
import csv
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

from config import (
    IMAGE_DIR, PRODUCT_CSV, EMBED_BATCH_SIZE,
    NEW_PRODUCT_DIR, NEW_PRODUCT_CSV
)
from utils.core import get_image_embeddings, load_image, sparse_to_dict

# BM25 支持（可选）
try:
    from vectorization.bm25 import BM25Vectorizer, BM25EmbeddingGenerator
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("提示: bm25 模块不可用，仅支持 TF-IDF")


class EmbeddingGenerator:
    """向量嵌入生成器

    支持 TF-IDF 和 BM25 两种稀疏向量生成方法
    """

    def __init__(self, use_bm25: bool = False):
        """初始化生成器

        Args:
            use_bm25: 是否使用 BM25（默认 False 使用 TF-IDF）
        """
        self.use_bm25 = use_bm25 and BM25_AVAILABLE
        self.tfidf_vectorizer = None
        self.bm25_vectorizer = None

    def load_products(self) -> Tuple[List[Dict], List[Image.Image]]:
        """
        加载商品目录数据和图片

        Returns:
            (商品元数据列表, 图片列表)
        """
        products = []
        images = []

        with open(PRODUCT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append(row)

        for p in products:
            img_path = IMAGE_DIR / p["image_path"]
            try:
                img = load_image(str(img_path))
                images.append(img)
            except FileNotFoundError:
                print(f"警告: 图片不存在 {img_path}")
                images.append(Image.new("RGB", (100, 100), (200, 200, 200)))

        return products, images

    def load_new_products(self) -> List[Dict]:
        """
        加载新品数据

        Returns:
            新品元数据列表
        """
        new_products = []

        with open(NEW_PRODUCT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                new_products.append(row)

        return new_products

    def generate_dense_vectors(self, images: List[Image.Image]) -> np.ndarray:
        """
        生成图片稠密向量 (视觉特征)

        Args:
            images: PIL Image 列表

        Returns:
            numpy 向量数组 (n x 2048)
        """
        print(f"\n生成稠密向量 ({len(images)} 张图片)...")
        vectors = get_image_embeddings(images, batch_size=EMBED_BATCH_SIZE)
        print(f"稠密向量形状: {vectors.shape}")
        return vectors

    def build_tfidf_vectorizer(self, products: List[Dict]) -> TfidfVectorizer:
        """
        构建 TF-IDF 向量化器

        Args:
            products: 商品元数据列表

        Returns:
            TfidfVectorizer 对象
        """
        print("\n" + "=" * 50)
        print("✓ 使用 TF-IDF 稀疏向量生成方法")
        print("=" * 50)

        descriptions = [p.get("description", "") for p in products]
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=500
        )
        self.tfidf_vectorizer.fit(descriptions)

        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        print(f"  TF-IDF 词汇表大小: {vocab_size}")

        return self.tfidf_vectorizer

    def _generate_sparse_vectors_tfidf(
        self,
        products: List[Dict],
        tfidf: TfidfVectorizer = None
    ) -> List[Dict[int, float]]:
        """
        生成 TF-IDF 文本稀疏向量 (关键词特征)

        Args:
            products: 商品元数据列表
            tfidf: TfidfVectorizer 对象，如果为 None 则会新建

        Returns:
            稀疏向量列表 [{index: value}, ...]
        """
        if tfidf is None:
            tfidf = self.build_tfidf_vectorizer(products)

        descriptions = [p.get("description", "") for p in products]
        tfidf_matrix = tfidf.transform(descriptions)

        sparse_vectors = [sparse_to_dict(tfidf_matrix[i]) for i in range(len(products))]

        # 统计非零项数量
        non_zero_counts = [len(v) for v in sparse_vectors]
        avg_non_zero = sum(non_zero_counts) / len(non_zero_counts) if non_zero_counts else 0
        print(f"TF-IDF 稀疏向量: {len(sparse_vectors)} 个, 平均非零项: {avg_non_zero:.1f}")

        return sparse_vectors

    # ==================== BM25 方法 ====================

    def build_bm25_vectorizer(
        self,
        products: List[Dict],
        k1: float = 1.5,
        b: float = 0.75
    ) -> "BM25Vectorizer":
        """
        构建 BM25 向量化器

        Args:
            products: 商品元数据列表
            k1: 词频饱和参数 (默认 1.5)
            b: 长度归一化参数 (默认 0.75)

        Returns:
            BM25Vectorizer 对象
        """
        if not BM25_AVAILABLE:
            raise ImportError("BM25 不可用，请安装 rank-bm25: pip install rank-bm25")

        print("\n" + "=" * 50)
        print("✓ 使用 BM25 稀疏向量生成方法")
        print("=" * 50)

        descriptions = [p.get("description", "") for p in products]

        self.bm25_vectorizer = BM25Vectorizer(
            k1=k1,
            b=b,
            max_features=500
        )
        self.bm25_vectorizer.fit(descriptions)

        vocab_size = len(self.bm25_vectorizer.vocab)
        print(f"  BM25 词汇表大小: {vocab_size}")
        print(f"  BM25 参数: k1={k1}, b={b}")

        return self.bm25_vectorizer

    def generate_sparse_vectors_bm25(
        self,
        products: List[Dict],
        bm25: "BM25Vectorizer" = None
    ) -> List[Dict[int, float]]:
        """
        生成 BM25 稀疏向量

        Args:
            products: 商品元数据列表
            bm25: BM25Vectorizer 对象，如果为 None 则会新建

        Returns:
            稀疏向量列表 [{index: value}, ...]
        """
        if not BM25_AVAILABLE:
            raise ImportError("BM25 不可用，请安装 rank-bm25: pip install rank-bm25")

        if bm25 is None:
            bm25 = self.build_bm25_vectorizer(products)
        else:
            self.bm25_vectorizer = bm25

        descriptions = [p.get("description", "") for p in products]
        sparse_vectors = bm25.transform(descriptions)

        # 统计非零项数量
        non_zero_counts = [len(v) for v in sparse_vectors]
        avg_non_zero = sum(non_zero_counts) / len(non_zero_counts) if non_zero_counts else 0
        print(f"BM25 稀疏向量: {len(sparse_vectors)} 个, 平均非零项: {avg_non_zero:.1f}")

        return sparse_vectors

    # ==================== 通用稀疏向量方法 ====================

    def build_sparse_vectorizer(
        self,
        products: List[Dict],
        method: str = "auto",
        **kwargs
    ) -> Union[TfidfVectorizer, "BM25Vectorizer"]:
        """
        构建稀疏向量化器（自动选择 TF-IDF 或 BM25）

        Args:
            products: 商品元数据列表
            method: "tfidf", "bm25", 或 "auto" (根据初始化设置)
            **kwargs: 传递给向量化器的额外参数

        Returns:
            向量化器对象
        """
        if method == "auto":
            method = "bm25" if self.use_bm25 else "tfidf"

        if method == "bm25":
            return self.build_bm25_vectorizer(products, **kwargs)
        else:
            return self.build_tfidf_vectorizer(products)

    def generate_sparse_vectors(
        self,
        products: List[Dict],
        vectorizer: Union[TfidfVectorizer, "BM25Vectorizer"] = None
    ) -> List[Dict[int, float]]:
        """
        生成稀疏向量（自动选择方法）

        Args:
            products: 商品元数据列表
            vectorizer: 向量化器对象，如果为 None 则会新建

        Returns:
            稀疏向量列表
        """
        # 自动检测向量化器类型
        if vectorizer is None:
            if self.use_bm25 and BM25_AVAILABLE:
                return self.generate_sparse_vectors_bm25(products)
            else:
                tfidf_vec = self.build_tfidf_vectorizer(products)
                return self._generate_sparse_vectors_tfidf(products, tfidf_vec)

        # 根据向量化器类型选择方法
        if BM25_AVAILABLE and isinstance(vectorizer, BM25Vectorizer):
            return self.generate_sparse_vectors_bm25(products, vectorizer)
        else:
            return self._generate_sparse_vectors_tfidf(products, vectorizer)

    # ==================== 新品编码 ====================

    def encode_new_product(
        self,
        new_product: Dict,
        tfidf: Optional[TfidfVectorizer] = None,
        bm25: Optional["BM25Vectorizer"] = None
    ) -> Tuple[np.ndarray, Dict[int, float], Image.Image]:
        """
        编码单个新品

        Args:
            new_product: 新品元数据
            tfidf: TF-IDF 向量化器（与 bm25 二选一）
            bm25: BM25 向量化器（与 tfidf 二选一）

        Returns:
            (稠密向量, 稀疏向量, 图片对象)

        Raises:
            RuntimeError: 当编码失败时
        """
        # 加载图片
        img_path = NEW_PRODUCT_DIR / new_product["image_path"]
        img = load_image(str(img_path))

        # Dense 向量: 图片编码
        dense_vectors = get_image_embeddings([img], batch_size=1)

        if len(dense_vectors) == 0:
            raise RuntimeError(
                f"图片编码失败 (API 错误或余额不足)。"
                f"请检查 OpenRouter 账户余额: https://openrouter.ai/settings/credits"
            )

        dense = dense_vectors[0]

        # Sparse 向量: 文本查询编码
        query_text = (
            f"{new_product.get('category', '')} "
            f"{new_product.get('style', '')} "
            f"{new_product.get('season', '')} "
            f"{new_product.get('prompt_hint', '')}"
        )

        # 根据向量化器类型选择编码方式
        if BM25_AVAILABLE and bm25 is not None:
            # 明确使用 BM25
            from vectorization.bm25 import encode_new_product_bm25
            sparse = encode_new_product_bm25(new_product, bm25)
        elif BM25_AVAILABLE and tfidf is not None and hasattr(tfidf, 'vocab'):
            # 检测到 tfidf 参数实际上是 BM25Vectorizer (通过 vocab 属性判断)
            from vectorization.bm25 import encode_new_product_bm25
            sparse = encode_new_product_bm25(new_product, tfidf)
        elif tfidf is not None:
            # 使用 TF-IDF
            sparse = sparse_to_dict(tfidf.transform([query_text])[0])
        else:
            # 默认使用内部存储的向量化器
            if self.use_bm25 and self.bm25_vectorizer is not None:
                from vectorization.bm25 import encode_new_product_bm25
                sparse = encode_new_product_bm25(new_product, self.bm25_vectorizer)
            elif self.tfidf_vectorizer is not None:
                sparse = sparse_to_dict(self.tfidf_vectorizer.transform([query_text])[0])
            else:
                raise ValueError("��初始化向量化器")

        # 打印稀疏向量统计
        print(f"  稀疏向量非零项: {len(sparse)}")  # type: ignore
        
        return dense, sparse, img

    def process_all_embeddings(self) -> Tuple[List[Dict], np.ndarray, List[Dict], Union[TfidfVectorizer, "BM25Vectorizer"]]:
        """
        处理所有商品数据，生成完整的嵌入向量

        Returns:
            (商品列表, 稠密向量, 稀疏向量, 向量化器)
        """
        print("=" * 50)
        print("第一步: 加载商品数据")
        print("=" * 50)

        products, images = self.load_products()
        print(f"已加载 {len(products)} 个商品")

        print("\n" + "=" * 50)
        print("第二步: 生成稠密向量 (图片特征)")
        print("=" * 50)

        dense_vectors = self.generate_dense_vectors(images)

        print("\n" + "=" * 50)
        print(f"第三步: 生成稀疏向量 (文本特征 - {'BM25' if self.use_bm25 else 'TF-IDF'})")
        print("=" * 50)

        if self.use_bm25 and BM25_AVAILABLE:
            vectorizer = self.build_bm25_vectorizer(products)
            sparse_vectors = self.generate_sparse_vectors_bm25(products, vectorizer)
        else:
            vectorizer = self.build_tfidf_vectorizer(products)
            sparse_vectors = self._generate_sparse_vectors_tfidf(products, vectorizer)

        return products, dense_vectors, sparse_vectors, vectorizer


if __name__ == "__main__":
    generator = EmbeddingGenerator()

    # 测试加载商品
    products, images = generator.load_products()
    print(f"\n已加载 {len(products)} 个商品")

    # 测试 TF-IDF
    tfidf = generator.build_tfidf_vectorizer(products)
    sparse = generator.generate_sparse_vectors(products, tfidf)
