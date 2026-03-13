"""
Milvus 数据库操作模块
"""
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

from config import MILVUS_URI, COLLECTION_NAME, EMBED_DIM


class FashionMilvusDB:
    """时尚产品向量数据库管理类"""

    def __init__(self, uri: str = MILVUS_URI):
        """
        初始化 Milvus 客户端

        Args:
            uri: 数据库连接 URI
        """
        self.client = MilvusClient(uri=uri)
        self.collection_name = COLLECTION_NAME

    def create_collection(self, overwrite: bool = False):
        """
       创建混合向量 Collection (Dense + Sparse + Scalar)

        Args:
            overwrite: 是否覆盖已存在的 Collection
        """
        if self.client.has_collection(self.collection_name):
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
            metric_type="IP"  # Inner Product
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
        print(f"第一条数据示例: product_id={rows[0]['product_id']}, dense_vector维度={len(rows[0]['dense_vector'])}")

        insert_result = self.client.insert(self.collection_name, rows)
        print(f"插入结果: {insert_result}")

        # Flush to ensure data is persisted
        self.client.flush(self.collection_name)

        stats = self.client.get_collection_stats(self.collection_name)
        print(f"Collection统计: {stats}")
        print(f"已插入 {stats.get('row_count', 0)} 条商品数据")

    def hybrid_search(
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

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取 Collection 统计信息"""
        return self.client.get_collection_stats(self.collection_name)


if __name__ == "__main__":
    # 测试数据库创建
    db = FashionMilvusDB()
    db.create_collection(overwrite=True)
    print(f"\nCollection 状态: {db.get_collection_stats()}")
