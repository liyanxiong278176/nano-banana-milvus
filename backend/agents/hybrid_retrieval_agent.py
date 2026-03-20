"""
检索Agent - HybridRetrievalAgent

负责执行Milvus向量检索，支持循环检索状态机和查询重写。

【职责】
1. 调用RetrievalWrapper进行向量检索
2. 支持循环检索状态机：最多3轮查询重写
3. 支持质量评估和智能降级机制

【优化策略】
- 启用循环检索：通过3轮查询重写提高检索质量
- 降低销量阈值：提高召回率（min_sales=500）
- 增加返回数量：给后续Agent更多选择（top_k=6）

【架构优势】
- 简洁设计：单一Milvus检索引擎
- 循环优化：基于质量评分自动调整查询条件
- 智能降级：无结果时自动放宽过滤条件

【输入】state字段
- query_dense: 稠密查询向量
- query_sparse: 稀疏查询向量
- category, style, season: 过滤条件
- scene_hint: 场景提示

【输出】state字段（新增/更新）
- retrieved_results: 检索结果列表（包含完整元数据）
- ref_images: 参考图片列表
- evidence_chain: 追加证据（含匹配理由）
- metrics: 记录retrieval_time, result_count
"""
from typing import List, Dict, Any

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval_wrapper import RetrievalWrapper
from config import (
    QUERY_REWRITE_SALES_LOW,
    MAX_RETRIEVAL_ROUNDS
)


class HybridRetrievalAgent(BaseAgent):
    """
    检索Agent

    执行Milvus向量检索，支持循环检索状态机和查询重写。

    【复用原有逻辑】
    - 使用retrieval_wrapper.py的RetrievalWrapper
    - 支持循环检索和查询重写
    - LLM质量评估驱动查询优化
    """

    def __init__(self, retriever: RetrievalWrapper):
        """
        初始化检索Agent

        Args:
            retriever: RetrievalWrapper实例（已初始化）
        """
        super().__init__("HybridRetrievalAgent")
        self.retriever = retriever

    @time_decorator("retrieval_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行检索流程

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

            # ==================== 2. 执行向量检索（循环检索状态机）====================
            # 调用RetrievalWrapper的retrieve_similar_bestsellers方法
            #
            # 【检索策略】
            # - 启用循环检索（enable_cycle=True）：通过多轮查询重写提高质量
            # - LLM质量评估驱动：根据评分自动调整查询条件
            # - 降低销量阈值：提高召回率
            # - 增加返回数量：让后续Agent有更多选择
            retrieved_results = self.retriever.retrieve_similar_bestsellers(
                query_dense=query_dense,
                query_sparse=query_sparse,
                category=category,
                min_sales=QUERY_REWRITE_SALES_LOW,  # 降低销量阈值，提高召回率
                top_k=6,  # 增加返回数量，给后续Agent更多选择
                enable_cycle=True,  # 【关键】启用循环检索状态机
                query_category=category,
                query_style=style,
                query_season=season,
                query_scene_hint=scene_hint
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

                # 构建匹配理由
                match_reason = (
                    f"结果{i}: {product_id} | 匹配理由: "
                    f"品类={category_match}, 风格={style_match}, "
                    f"销量={sales_count}, 相似度={score:.4f}"
                )
                self._add_evidence(state, match_reason)

            # ==================== 5. 记录指标 ====================
            self._log_metric(state, "result_count", result_count)

            return state

        except Exception as e:
            return self._handle_error(state, f"检索失败: {str(e)}")


if __name__ == "__main__":
    print("HybridRetrievalAgent需要真实的RetrievalWrapper实例才能完整测试")
    print("请通过workflow.py中的完整流程进行测试")
