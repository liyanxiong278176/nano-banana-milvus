"""
混合检索Agent - HybridRetrievalAgent

负责执行Milvus向量检索和Neo4j图谱检索的混合融合。

【职责】
1. 调用HybridRetriever进行混合检索（RRF融合算法）
2. 采用多路召回架构：Milvus向量 + Neo4j多跳推理
3. RRF融合结果，平衡两个召回源

【优化策略】
- 关掉循环检索：避免3轮查询重写的开销，提升速度
- 保留多跳推理：让Neo4j通过风格相似性扩展候选集
- 降低销量阈值：提高召回率（min_sales=500）
- 增加返回数量：给后续Agent更多选择（top_k=6）

【架构优势】
- 解耦设计：两个检索器独立，易于扩展
- 故障隔离：一个检索器失败不影响整体
- 权重可调：支持A/B测试优化

【输入】state字段
- query_dense: 稠密查询向量
- query_sparse: 稀疏查询向量
- category, style, season: 过滤条件
- scene_hint: 场景提示

【输出】state字段（新增/更新）
- retrieved_results: 检索结果列表（包含完整元数据）
- ref_images: 参考图片列表
- evidence_chain: 追加证据（含匹配理由）
- metrics: 记录retrieval_time, result_count, milvus_count, graph_count
"""
from typing import List, Dict, Any

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph import HybridRetriever


class HybridRetrievalAgent(BaseAgent):
    """
    混合检索Agent

    执行Milvus + Neo4j混合检索，融合向量相似度和图谱关联度。

    【复用原有逻辑】
    - 复用graph/hybrid_retriever.py的HybridRetriever
    - 支持循环检索和多跳推理
    - RRF融合算法
    """

    def __init__(self, retriever: HybridRetriever):
        """
        初始化混合检索Agent

        Args:
            retriever: HybridRetriever实例（已初始化）
        """
        super().__init__("HybridRetrievalAgent")
        self.retriever = retriever

    @time_decorator("retrieval_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行混合检索流程

        Args:
            state: 包含query_dense, query_sparse和过滤条件的状态

        Returns:
            更新后的状态，包含retrieved_results和ref_images
        """
        self._update_status(state, "processing", "HybridRetrievalAgent")

        try:
            # ==================== 1. 验证输入 ====================
            query_dense = state.get("query_dense")
            query_sparse = state.get("query_sparse")

            if query_dense is None:
                raise ValueError("query_dense为空，请先执行EmbeddingAgent")

            if query_sparse is None:
                # 如果没有稀疏向量，使用空字典
                query_sparse = {}

            # 确保query_dense是列表格式（Milvus API要求）
            if hasattr(query_dense, 'tolist'):
                query_dense = query_dense.tolist()

            category = state.get("category", "")
            style = state.get("style", "")
            season = state.get("season", "")
            scene_hint = state.get("scene_hint", "")

            self._add_evidence(
                state,
                f"开始检索: 品类={category or 'All'}, 风格={style or 'All'}, 季节={season or 'All'}"
            )

            # ==================== 2. 执行混合检索 ====================
            # 调用HybridRetriever的retrieve_similar_bestsellers方法
            #
            # 【优化策略】
            # - 关掉循环检索（enable_cycle=False）：避免3轮查询重写的开销
            # - 保留多跳推理（enable_multi_hop=True）：让Neo4j扩展候选集
            # - 降低销量阈值（min_sales=500）：提高召回率
            # - 增加返回数量（top_k=6）：让后续Agent有更多选择
            retrieved_results = self.retriever.retrieve_similar_bestsellers(
                query_dense=query_dense,
                query_sparse=query_sparse,
                category=category,
                style=style,
                season=season,
                scene_hint=scene_hint,
                min_sales=500,  # 降低销量阈值，提高召回率
                top_k=6,  # 增加返回数量，给后续Agent更多选择
                enable_cycle=False,  # 【优化】关掉循环检索，提升速度
                query_category=category,
                query_style=style,
                query_season=season,
                query_scene_hint=scene_hint,
                enable_multi_hop=True,  # 保留多跳推理，扩展候选集
                max_hops=3
            )

            if not retrieved_results:
                self._add_evidence(state, "检索结果为空，将触发兜底策略")
                # 不抛出异常，让后续流程处理

            # ==================== 3. 提取参考图片 ====================
            ref_images = []
            for result in retrieved_results:
                if result.get("image"):
                    ref_images.append(result["image"])

            state["retrieved_results"] = retrieved_results
            state["ref_images"] = ref_images

            # ==================== 4. 记录检索详情到证据链 ====================
            result_count = len(retrieved_results)
            self._add_evidence(state, f"检索完成: 返回{result_count}个结果")

            # 为每个结果添加匹配理由到证据链
            for i, result in enumerate(retrieved_results[:5], 1):  # 最多记录5个
                product_id = result.get("product_id", "N/A")
                category_match = result.get("category", "N/A")
                style_match = result.get("style", "N/A")
                sales_count = result.get("sales_count", 0)
                score = result.get("score", 0)
                source = result.get("source", "unknown")  # milvus/graph/hybrid

                # 构建匹配理由
                match_reason = (
                    f"结果{i}: {product_id} | 匹配理由: "
                    f"品类={category_match}, 风格={style_match}, "
                    f"销量={sales_count}, 相似度={score:.4f}, 来源={source}"
                )
                self._add_evidence(state, match_reason)

            # ==================== 5. 记录指标 ====================
            self._log_metric(state, "result_count", result_count)

            # 统计来源分布
            source_counts = {"milvus": 0, "graph": 0, "hybrid": 0}
            for result in retrieved_results:
                source = result.get("source", "unknown")
                if source in source_counts:
                    source_counts[source] += 1

            self._log_metric(state, "milvus_count", source_counts["milvus"])
            self._log_metric(state, "graph_count", source_counts["graph"])
            self._log_metric(state, "hybrid_count", source_counts["hybrid"])

            return state

        except Exception as e:
            return self._handle_error(state, f"混合检索失败: {str(e)}")


if __name__ == "__main__":
    print("HybridRetrievalAgent需要真实的HybridRetriever实例才能完整测试")
    print("请通过workflow.py中的完整流程进行测试")
