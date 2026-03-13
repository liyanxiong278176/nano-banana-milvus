"""
向量嵌入生成模块
"""
import csv
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

from config import (
    IMAGE_DIR, PRODUCT_CSV, EMBED_BATCH_SIZE,
    NEW_PRODUCT_DIR, NEW_PRODUCT_CSV
)
from utils import get_image_embeddings, load_image, sparse_to_dict


class EmbeddingGenerator:
    """向量嵌入生成器"""

    def __init__(self):
        """初始化生成器"""
        self.tfidf_vectorizer = None

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
        descriptions = [p.get("description", "") for p in products]
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=500
        )
        self.tfidf_vectorizer.fit(descriptions)

        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        print(f"TF-IDF 词汇表大小: {vocab_size}")

        return self.tfidf_vectorizer

    def generate_sparse_vectors(
        self,
        products: List[Dict],
        tfidf: TfidfVectorizer = None
    ) -> List[Dict[int, float]]:
        """
        生成文本稀疏向量 (关键词特征)

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
        print(f"稀疏向量: {len(sparse_vectors)} 个, 平均非零项: {avg_non_zero:.1f}")

        return sparse_vectors

    def encode_new_product(
        self,
        new_product: Dict,
        tfidf: TfidfVectorizer
    ) -> Tuple[np.ndarray, Dict[int, float], Image.Image]:
        """
        编码单个新品

        Args:
            new_product: 新品元数据
            tfidf: TF-IDF 向量化器

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
        sparse = sparse_to_dict(tfidf.transform([query_text])[0])

        return dense, sparse, img

    def process_all_embeddings(self) -> Tuple[List[Dict], np.ndarray, List[Dict], TfidfVectorizer]:
        """
        处理所有商品数据，生成完整的嵌入向量

        Returns:
            (商品列表, 稠密向量, 稀疏向量, TF-IDF 向量化器)
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
        print("第三步: 生成稀疏向量 (文本特征)")
        print("=" * 50)

        tfidf = self.build_tfidf_vectorizer(products)
        sparse_vectors = self.generate_sparse_vectors(products, tfidf)

        return products, dense_vectors, sparse_vectors, tfidf


if __name__ == "__main__":
    generator = EmbeddingGenerator()

    # 测试加载商品
    products, images = generator.load_products()
    print(f"\n已加载 {len(products)} 个商品")

    # 测试 TF-IDF
    tfidf = generator.build_tfidf_vectorizer(products)
    sparse = generator.generate_sparse_vectors(products, tfidf)
