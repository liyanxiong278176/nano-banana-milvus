"""
爆款检索模块
"""
from typing import List, Dict, Tuple
from PIL import Image

from config import MIN_SALES_COUNT, IMAGE_DIR
from utils import load_image


class BestsellerRetriever:
    """爆款商品检索器"""

    def __init__(self, milvus_db):
        """
        初始化检索器

        Args:
            milvus_db: FashionMilvusDB 实例
        """
        self.db = milvus_db

    def retrieve_similar_bestsellers(
        self,
        query_dense: List[float],
        query_sparse: dict,
        category: str,
        min_sales: int = MIN_SALES_COUNT,
        top_k: int = 3
    ) -> List[Dict]:
        """
        检索相似爆款商品

        Args:
            query_dense: 稠密查询向量
            query_sparse: 稀疏查询向量
            category: 品类筛选
            min_sales: 最低销量
            top_k: 返回数量

        Returns:
            检索结果列表，包含商品信息和图片
        """
        # 构建过滤表达式
        filter_expr = f'category == "{category}" and sales_count > {min_sales}'

        print(f"\n执行混合检索...")
        print(f"  品类: {category}")
        print(f"  最低销量: {min_sales}")
        print(f"  稠密向量维度: {len(query_dense)}")
        print(f"  稀疏向量非零项: {len(query_sparse)}")

        results = self.db.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            filter_expr=filter_expr,
            top_k=top_k
        )

        print(f"\n找到 {len(results)} 个相似爆款:")
        print("-" * 60)

        retrieved_with_images = []
        for hit in results:
            entity = hit["entity"]
            product_id = entity["product_id"]

            # 加载参考图片
            try:
                img_path = IMAGE_DIR / f"{product_id}.jpg"
                ref_img = load_image(str(img_path))
            except FileNotFoundError:
                ref_img = None

            result = {
                **entity,
                "score": hit["distance"],
                "image": ref_img
            }
            retrieved_with_images.append(result)

            # 打印结果信息
            print(f"  {entity['product_id']} | {entity['category']} | "
                  f"{entity['color']} | {entity['style']}")
            print(f"    销量: {entity['sales_count']} | 价格: ${entity['price']:.1f} | "
                  f"相似度: {hit['distance']:.4f}")
            print(f"    描述: {entity['description']}")
            print("-" * 60)

        return retrieved_with_images

    def format_retrieval_for_prompt(
        self,
        retrieved: List[Dict]
    ) -> Tuple[List[Image.Image], str]:
        """
        格式化检索结果用于 Prompt 生成

        Args:
            retrieved: 检索结果列表

        Returns:
            (图片列表, 文字描述)
        """
        images = []
        description = "Top-selling fashion products analysis:\n\n"

        for i, item in enumerate(retrieved, 1):
            if item.get("image"):
                images.append(item["image"])

            description += f"{i}. Product: {item['product_id']}\n"
            description += f"   Style: {item['style']}, Color: {item['color']}\n"
            description += f"   Description: {item['description']}\n\n"

        return images, description


if __name__ == "__main__":
    from milvus_db import FashionMilvusDB

    # 测试检索
    db = FashionMilvusDB()
    retriever = BestsellerRetriever(db)

    # 假设的查询向量
    query_dense = [0.1] * 2048
    query_sparse = {0: 0.5, 10: 0.3}

    results = retriever.retrieve_similar_bestsellers(
        query_dense, query_sparse, "midi_dress"
    )
