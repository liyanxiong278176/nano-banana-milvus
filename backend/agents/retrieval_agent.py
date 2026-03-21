"""
检索Agent - RetrievalAgent

【合并】EmbeddingAgent + HybridRetrievalAgent

负责向量编码和向量检索的完整流程。

【职责】
1. 将新品图片编码为稠密向量（视觉特征）和稀疏向量（文本特征）
2. 执行Milvus向量检索，支持循环检索状态机和查询重写
3. 返���检索结果和参考图片

【输入】state字段
- new_image: 新品图片（PIL Image）
- category, style, season, scene_hint: 过滤条件

【输出】state字段（新增/更新）
- query_dense: 稠密向量
- query_sparse: 稀疏向量
- retrieved_results: 检索结果列表
- ref_images: 参考图片列表
- evidence_chain: 追加证据
- metrics: 记录retrieval_time, result_count
"""
from typing import Dict, List
import numpy as np

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from vectorization.embedding import EmbeddingGenerator
from retrieval.wrapper import RetrievalWrapper
from config import QUERY_REWRITE_SALES_LOW

# 【v2.2新增】导入提示词工程
try:
    from prompts.v2 import get_metrics, record_prompt_execution
    PROMPTS_V2_AVAILABLE = True
except ImportError:
    PROMPTS_V2_AVAILABLE = False


class RetrievalAgent(BaseAgent):
    """
    检索Agent

    合并向量编码和向量检索功能，在一个Agent中完成完整检索流程。

    【优势】
    - 减少状态传递开销
    - 逻辑更紧凑，易于维护
    - 保持原有功能完全兼容
    """

    def __init__(self, embed_gen: EmbeddingGenerator, tfidf_vectorizer, retriever: RetrievalWrapper):
        """
        初始化检索Agent

        Args:
            embed_gen: EmbeddingGenerator实例
            tfidf_vectorizer: TF-IDF向量化器
            retriever: RetrievalWrapper实例
        """
        super().__init__("RetrievalAgent")
        self.embed_gen = embed_gen
        self.tfidf_vectorizer = tfidf_vectorizer
        self.retriever = retriever

        # 【v2.2新增】设置提示词版本
        self.set_prompt_version("2.0")

    @time_decorator("retrieval_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行完整检索流程：编码 → 检索

        Args:
            state: 包含new_image和过滤条件的状态

        Returns:
            更新后的状态，包含retrieved_results和ref_images
        """
        import time
        start_time = time.time()

        self._update_status(state, "processing", "RetrievalAgent")

        # 【v2.2新增】记录提示词版本
        self._log_prompt_version(state)

        try:
            # ==================== Step 1: 向量编码 ====================
            new_image = state.get("new_image")
            if not new_image:
                raise ValueError("新品图片为空，请先执行UploadAgent")

            # 构建新品字典
            new_product = {
                "new_id": state.get("product_id", ""),
                "image_path": f"{state.get('product_id', '')}.jpg",
                "category": state.get("category", ""),
                "style": state.get("style", ""),
                "season": state.get("season", ""),
                "prompt_hint": state.get("scene_hint", "")
            }

            self._add_evidence(
                state,
                f"[编码] 品类={new_product['category']}, 风格={new_product['style']}"
            )

            # 生成向量
            dense_vector, sparse_vector, _ = self.embed_gen.encode_new_product(
                new_product=new_product,
                tfidf=self.tfidf_vectorizer
            )

            if dense_vector is None or len(dense_vector) == 0:
                raise RuntimeError("稠密向量生成失败")

            dense_dim = len(dense_vector)
            sparse_nonzero = len(sparse_vector) if sparse_vector else 0

            self._add_evidence(
                state,
                f"[编码] 完成: 稠密={dense_dim}维, 稀疏={sparse_nonzero}项"
            )

            # 更新状态
            state["query_dense"] = dense_vector
            state["query_sparse"] = sparse_vector

            # ==================== Step 2: 向量检索 ====================
            category = state.get("category", "")
            style = state.get("style", "")
            season = state.get("season", "")
            scene_hint = state.get("scene_hint", "")

            self._add_evidence(
                state,
                f"[检索] 品类={category or 'All'}, 风格={style or 'All'}, 季节={season or 'All'}"
            )

            # 确保query_dense是列表格式
            if hasattr(dense_vector, 'tolist'):
                query_dense = dense_vector.tolist()
            else:
                query_dense = dense_vector

            # 执行检索
            retrieved_results = self.retriever.retrieve_similar_bestsellers(
                query_dense=query_dense,
                query_sparse=sparse_vector or {},
                category=category,
                min_sales=QUERY_REWRITE_SALES_LOW,
                top_k=6,
                enable_cycle=True,
                query_category=category,
                query_style=style,
                query_season=season,
                query_scene_hint=scene_hint
            )

            # 提取参考图片
            ref_images = []
            for result in retrieved_results:
                if result.get("image"):
                    ref_images.append(result["image"])

            state["retrieved_results"] = retrieved_results
            state["ref_images"] = ref_images

            # ==================== Step 3: 记录结果 ====================
            result_count = len(retrieved_results)
            self._add_evidence(state, f"[检索] 完成: 返回{result_count}个结果")

            # 记录前5个结果
            for i, result in enumerate(retrieved_results[:5], 1):
                product_id = result.get("product_id", "N/A")
                category_match = result.get("category", "N/A")
                style_match = result.get("style", "N/A")
                sales_count = result.get("sales_count", 0)
                score = result.get("score", 0)

                match_reason = (
                    f"结果{i}: {product_id} | "
                    f"品类={category_match}, 风格={style_match}, "
                    f"销量={sales_count}, 相似度={score:.4f}"
                )
                self._add_evidence(state, match_reason)

            # 记录指标
            self._log_metric(state, "result_count", result_count)
            self._log_metric(state, "dense_dim", dense_dim)

            # 【v2.2新增】记录提示词执行成功
            self._record_prompt_execution(
                state,
                success=True,
                execution_time=time.time() - start_time,
                metadata={
                    "result_count": result_count,
                    "category": category,
                    "style": style
                }
            )

            return state

        except Exception as e:
            # 【v2.2新增】记录失败
            self._record_prompt_execution(
                state,
                success=False,
                execution_time=time.time() - start_time,
                error=str(e)
            )
            return self._handle_error(state, f"检索失败: {str(e)}")


if __name__ == "__main__":
    print("RetrievalAgent需要真实的EmbeddingGenerator和RetrievalWrapper实例才能完整测试")
    print("请通过workflow.py中的完整流程进行测试")
