"""
检索包装器 - 纯 Milvus 向量检�� + 循环检索状态机

提供与原 HybridRetriever 兼容的接口，简化为 Milvus 单引擎检索。
"""
from typing import List, Dict, Any

from retrieval import BestsellerRetriever


class RetrievalWrapper:
    """
    检索包装器 - 封装 BestsellerRetriever

    提供与原 HybridRetriever 兼容的接口，使用纯 Milvus 向量检索。
    支持循环检索状态机和查询重写功能。
    """

    def __init__(self):
        """初始化检索包装器"""
        print(f"\n{'='*60}")
        print("【检索包装器初始化】")
        print(f"{'='*60}")

        # 初始化 Milvus 检索器
        print(f"  [1/1] 初始化 Milvus 检索器...")
        self.milvus_retriever = BestsellerRetriever()
        try:
            has_collection = self.milvus_retriever.has_collection()
            print(f"       [OK] Milvus 检索器已就绪 ({'已连接' if has_collection else '集合未创建'})")
        except Exception as e:
            print(f"       [WARN] Milvus 检索器初始化警告: {e}")

        print(f"\n{'='*60}")
        print(f"[OK] 检索包装器初始化完成")
        print(f"     检索模式: Milvus 单引擎 (支持循环检索状态机)")
        print(f"{'='*60}\n")

    def retrieve_similar_bestsellers(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        category: str = "",
        min_sales: int = 500,
        top_k: int = 3,
        enable_cycle: bool = True,
        query_category: str = "",
        query_style: str = "",
        query_season: str = "",
        query_scene_hint: str = ""
    ) -> List[Dict[str, Any]]:
        """
        向量检索（支持循环检索状态机）

        Args:
            query_dense: 稠密查询向量
            query_sparse: 稀疏查询向量
            category: 品类过滤
            min_sales: 最低销量
            top_k: 返回数量
            enable_cycle: 是否启用循环检索状态机（默认True）
            query_category: 查询品类（用于质量评估）
            query_style: 查询风格（用于质量评估）
            query_season: 查询季节（用于质量评估）
            query_scene_hint: 查询场景提示（用于质量评估）

        Returns:
            检索结果列表
        """
        return self.milvus_retriever.retrieve_similar_bestsellers(
            query_dense=query_dense,
            query_sparse=query_sparse,
            category=category,
            min_sales=min_sales,
            top_k=top_k,
            enable_cycle=enable_cycle,
            query_category=query_category,
            query_style=query_style,
            query_season=query_season,
            query_scene_hint=query_scene_hint
        )


def create_retrieval_wrapper() -> RetrievalWrapper:
    """
    创建检索包装器实例（便捷函数）

    Returns:
        检索包装器实例
    """
    return RetrievalWrapper()


if __name__ == "__main__":
    """
    测试检索包装器
    """
    import argparse

    parser = argparse.ArgumentParser(description="检索包装器测试")
    parser.add_argument("--status", action="store_true", help="显示检索器状态")

    args = parser.parse_args()

    wrapper = create_retrieval_wrapper()

    if args.status:
        # 获取 Milvus 状态
        try:
            has_collection = wrapper.milvus_retriever.has_collection()
            if has_collection:
                stats = wrapper.milvus_retriever.get_collection_stats()
                print("\n检索包装器状态:")
                print(f"  Milvus: [OK]")
                print(f"  数据量: {stats.get('row_count', 0)} 条")
            else:
                print("\n检索包装器状态:")
                print(f"  Milvus: [未初始化]")
        except Exception as e:
            print(f"\n状态获取失败: {e}")
