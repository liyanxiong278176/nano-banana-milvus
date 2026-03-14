"""
爆款检索模块 - Milvus 数据库 + 混合检索
"""
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

from config import (
    MILVUS_URI, COLLECTION_NAME, EMBED_DIM,
    MIN_SALES_COUNT, IMAGE_DIR
)
from utils import load_image


class BestsellerRetriever:
    """爆款商品检索器 - 集成数据库管理和混合检索"""

    def __init__(self, uri: str = MILVUS_URI):
        """
        初始化检索器

        Args:
            uri: Milvus 连接 URI
        """
        self.client = MilvusClient(uri=uri)
        self.collection_name = COLLECTION_NAME

    # ==================== 数据库管理 ====================

    def has_collection(self) -> bool:
        """检查 Collection 是否存在"""
        return self.client.has_collection(self.collection_name)

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取 Collection 统计信息"""
        return self.client.get_collection_stats(self.collection_name)

    def create_collection(self, overwrite: bool = False):
        """
        创建混合向量 Collection (Dense + Sparse + Scalar)

        Args:
            overwrite: 是否覆盖已存在的 Collection
        """
        if self.has_collection():
            if overwrite:
                self.client.drop_collection(self.collection_name)
                print(f"已删除旧 Collection: {self.collection_name}")
            else:
                print(f"Collection 已存在: {self.collection_name}")
                return

        # 定义 Schema
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 主键
        schema.add_field("id", DataType.INT64, is_primary=True)

        # 标量字段
        schema.add_field("product_id", DataType.VARCHAR, max_length=20)
        schema.add_field("category", DataType.VARCHAR, max_length=50)
        schema.add_field("color", DataType.VARCHAR, max_length=50)
        schema.add_field("style", DataType.VARCHAR, max_length=50)
        schema.add_field("season", DataType.VARCHAR, max_length=50)
        schema.add_field("sales_count", DataType.INT64)
        schema.add_field("description", DataType.VARCHAR, max_length=500)
        schema.add_field("price", DataType.FLOAT)

        # 向量字段
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        # 配置索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="FLAT",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 创建 Collection
        self.client.create_collection(
            self.collection_name,
            schema=schema,
            index_params=index_params
        )

        print(f"Collection 创建成功: {self.collection_name}")

    def insert_products(
        self,
        products: List[Dict[str, Any]],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[int, float]]
    ):
        """
        插入商品数据

        Args:
            products: 商品元数据列表
            dense_vectors: 图片稠密向量列表
            sparse_vectors: 文本稀疏向量列表
        """
        rows = []
        for i, p in enumerate(products):
            rows.append({
                "product_id": p.get("product_id", f"SKU{i:03d}"),
                "category": p.get("category", ""),
                "color": p.get("color", ""),
                "style": p.get("style", ""),
                "season": p.get("season", ""),
                "sales_count": int(p.get("sales_count", 0)),
                "description": p.get("description", ""),
                "price": float(p.get("price", 0.0)),
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
            })

        print(f"准备插入 {len(rows)} 条数据")
        insert_result = self.client.insert(self.collection_name, rows)

        # 持久化
        self.client.flush(self.collection_name)

        stats = self.get_collection_stats()
        print(f"已插入 {stats.get('row_count', 0)} 条商品数据")

    # ==================== 混合检索 ====================

    def _hybrid_search(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        filter_expr: str,
        top_k: int = 3,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        混合向量检索 (Dense + Sparse + Scalar Filter + RRF)

        Args:
            query_dense: 稠密查询向量
            query_sparse: 稀疏查询向量
            filter_expr: 标量过滤表达式
            top_k: 返回结果数量
            rrf_k: RRF 融合参数

        Returns:
            检索结果列表
        """
        # Dense 向量检索请求
        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=20,
            expr=filter_expr,
        )

        # Sparse 向量检索请求
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=20,
            expr=filter_expr,
        )

        # 执行混合搜索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=rrf_k),
            limit=top_k,
            output_fields=[
                "product_id", "category", "color", "style", "season",
                "sales_count", "description", "price"
            ],
        )

        return results[0] if results else []

    def retrieve_similar_bestsellers(
        self,
        query_dense: List[float],
        query_sparse: dict,
        category: str,
        min_sales: int = MIN_SALES_COUNT,
        top_k: int = 3
    ) -> List[Dict]:
        """
        检索相似爆款商品（带图片加载）

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

        # 执行检索
        results = self._hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            filter_expr=filter_expr,
            top_k=top_k
        )

        print(f"\n找到 {len(results)} 个相似爆款:")
        print("-" * 60)

        # 加载图片并整理结果
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
    # 测试检索器
    retriever = BestsellerRetriever()

    # 检查状态
    if retriever.has_collection():
        stats = retriever.get_collection_stats()
        print(f"Collection 存在，记录数: {stats.get('row_count', 0)}")
    else:
        print("Collection 不存在，请先运行 main.py 初始化数据库")
