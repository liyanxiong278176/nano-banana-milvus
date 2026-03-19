"""
向量编码Agent - EmbeddingAgent

负责将新品图片编码为稠密向量（视觉特征）和稀疏向量（文本特征）。

【职责】
1. 调用EmbeddingGenerator生成图片的稠密向量（2048维视觉特征）
2. 使用TF-IDF生成文本稀疏向量（品类、风格、季节等文本特征）
3. 记录向量维度和编码耗时

【输入】state字段
- new_image: 新品图片（PIL Image）
- category, style, season, scene_hint: 用于生成稀疏向量

【输出】state字段（新增/更新）
- query_dense: 稠密向量（numpy数组）
- query_sparse: 稀疏向量（字典格式）
- evidence_chain: 追加证据
- metrics: 记录embedding_time, dense_dim, sparse_nonzero
"""
from typing import Dict
import numpy as np

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from embedding import EmbeddingGenerator


class EmbeddingAgent(BaseAgent):
    """
    向量编码Agent

    将新品图片编码为向量表示，供后续检索使用。

    【复用原有逻辑】
    - 复用embedding.py的EmbeddingGenerator.encode_new_product()
    - 完全兼容现有的向量生成流程
    """

    def __init__(self, embed_gen: EmbeddingGenerator, tfidf_vectorizer):
        """
        初始化向量编码Agent

        Args:
            embed_gen: EmbeddingGenerator实例（复用现有模块）
            tfidf_vectorizer: TF-IDF向量化器（已训练）
        """
        super().__init__("EmbeddingAgent")
        self.embed_gen = embed_gen
        self.tfidf_vectorizer = tfidf_vectorizer

    @time_decorator("embedding_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行向量编码流程

        Args:
            state: 包含new_image和元数据的状态

        Returns:
            更新后的状态，包含query_dense和query_sparse
        """
        self._update_status(state, "processing", "EmbeddingAgent")

        try:
            # ==================== 1. 准备新品数据 ====================
            new_image = state.get("new_image")
            if not new_image:
                raise ValueError("新品图片为空，请先执行UploadAgent")

            # 构建新品字典（复用现有格式）
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
                f"开始编码: 品类={new_product['category']}, 风格={new_product['style']}"
            )

            # ==================== 2. 生成向量 ====================
            # 调用现有的encode_new_product方法
            dense_vector, sparse_vector, _ = self.embed_gen.encode_new_product(
                new_product=new_product,
                tfidf=self.tfidf_vectorizer
            )

            # ==================== 3. 验证向量 ====================
            if dense_vector is None or len(dense_vector) == 0:
                raise RuntimeError("稠密向量生成失败（API错误或余额不足）")

            dense_dim = len(dense_vector)
            sparse_nonzero = len(sparse_vector) if sparse_vector else 0

            self._add_evidence(
                state,
                f"向量生成完成: 稠密向量={dense_dim}维, 稀疏向量={sparse_nonzero}个非零项"
            )

            # ==================== 4. 更新状态 ====================
            state["query_dense"] = dense_vector
            state["query_sparse"] = sparse_vector

            # ==================== 5. 记录指标 ====================
            self._log_metric(state, "dense_dim", dense_dim)
            self._log_metric(state, "sparse_nonzero", sparse_nonzero)

            return state

        except Exception as e:
            return self._handle_error(state, f"向量编码失败: {str(e)}")


if __name__ == "__main__":
    # 测试EmbeddingAgent
    from .state import create_initial_state
    from PIL import Image
    from utils import load_image
    from config import PRODUCT_CSV, IMAGE_DIR

    # 创建测试状态（需要有真实图片）
    test_img = Image.new("RGB", (800, 1200), color=(100, 150, 200))

    state = create_initial_state(
        task_id="test_embed",
        file_bytes=b"test",
        category="midi_dress",
        style="elegant"
    )
    state["new_image"] = test_img
    state["product_id"] = "TEST_001"

    # 注意：实际测试需要初始化EmbeddingGenerator和TF-IDF
    print("EmbeddingAgent需要真实的EmbeddingGenerator实例才能完整测试")
    print("请通过workflow.py中的完整流程进行测试")
