"""
质量评估Agent - QualityJudgeAgent

负责对生成的图片进行多维度质量评分。

【职责】
1. 调用ImageQualityJudge对生成图片评分
2. 判断是否需要重新生成
3. 选择最佳图片
4. 控制重试次数（最多1次）

【输入】state字段
- generated_images: 生成的图片列表
- new_image: 原始新品图（对比用）
- ref_images: 参考爆款图（对比用）
- retry_count: 当前重试次数

【输出】state字段（新增/更新）
- quality_scores: 评分结果
- best_image: 最佳图片
- should_regenerate: 是否需要重新生成
- regenerate_reason: 重新生成原因
- retry_count: 更新的重试次数
- evidence_chain: 追加证据
- metrics: 记录quality_judge_time, best_score, should_regenerate
"""
from typing import Dict, Any, Optional

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from image_gen import ImageQualityJudge


class QualityJudgeAgent(BaseAgent):
    """
    质量评估Agent

    使用多模态LLM对生成的图片进行多维度评分。

    【复用原有逻辑】
    - 复用image_gen.py的ImageQualityJudge.score_image_quality()
    - 复用should_regenerate()判断逻辑
    - 支持多维度评分（服装准确性、姿势自然度、场景质量等）
    """

    def __init__(self, model: str = None, min_score: float = 3.5, max_retry: int = 1):
        """
        初始化质量评估Agent

        Args:
            model: 评分模型（None则使用config.LLM_MODEL）
            min_score: 最低及格分数
            max_retry: 最大重试次数
        """
        super().__init__("QualityJudgeAgent")
        self.judge = ImageQualityJudge(model=model)
        self.min_score = min_score
        self.max_retry = max_retry

    @time_decorator("quality_judge_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行质量评估流程

        Args:
            state: 包含generated_images的状态

        Returns:
            更新后的状态，包含quality_scores和should_regenerate
        """
        self._update_status(state, "processing", "QualityJudgeAgent")

        try:
            # ==================== 1. 验证输入 ====================
            generated_images = state.get("generated_images", [])
            new_image = state.get("new_image")
            ref_images = state.get("ref_images", [])

            if not generated_images:
                raise ValueError("生成的图片列表为空，请先执行ImageGenAgent")

            self._add_evidence(
                state,
                f"开始质量评估: {len(generated_images)}张图片"
            )

            # ==================== 2. 选择第一张图片进行评分 ====================
            # 注意：目前只对第一张图片评分，如需多选可改为循环评分
            generated_image = generated_images[0]

            # 调用质量评分
            scores = self.judge.score_image_quality(
                generated_image=generated_image,
                original_image=new_image,
                reference_images=ref_images
            )

            # ==================== 3. 记录评分详情 ====================
            avg_score = scores.get("average", 0)
            is_fallback = scores.get("is_fallback", False)

            self._add_evidence(
                state,
                f"评分完成: 总分={avg_score:.1f}/5 "
                f"{'(默认值)' if is_fallback else '(AI评分)'}"
            )

            # 记录各维度得分
            dimensions = ["clothing_accuracy", "pose_naturalness",
                         "scene_quality", "lighting_quality", "commercial_value"]
            for dim in dimensions:
                score = scores.get(dim, 0)
                self._add_evidence(state, f"  {dim}: {score}/5")

            # ==================== 4. 判断是否需要重新生成 ====================
            should_regenerate, reason = self.judge.should_regenerate(
                scores, threshold=self.min_score
            )

            retry_count = state.get("retry_count", 0)

            # 控制重试逻辑
            if should_regenerate and retry_count < self.max_retry:
                # 允许重试
                state["should_regenerate"] = True
                state["retry_count"] = retry_count + 1
                state["regenerate_reason"] = reason
                self._add_evidence(state, f"需要重新生成: {reason}")
                self._add_evidence(state, f"当前重试次数: {retry_count + 1}/{self.max_retry}")
                self._log_metric(state, "should_regenerate", 1)
            else:
                # 不再重试（质量合格或已达最大重试次数）
                state["should_regenerate"] = False
                state["best_image"] = generated_image

                if should_regenerate:
                    self._add_evidence(
                        state,
                        f"已达最大重试次数({self.max_retry})，不再重试"
                    )
                else:
                    self._add_evidence(state, "质量合格，无需重试")

                self._log_metric(state, "should_regenerate", 0)

            # ==================== 5. 更新状态 ====================
            state["quality_scores"] = scores

            # ==================== 6. 记录指标 ====================
            self._log_metric(state, "best_score", avg_score)
            self._log_metric(state, "is_fallback", 1 if is_fallback else 0)

            return state

        except Exception as e:
            # 出错时标记不需要重试，避免死循环
            state["should_regenerate"] = False
            state["best_image"] = state.get("generated_images", [None])[0]
            state["quality_scores"] = {"average": 0, "error": str(e)}
            return self._handle_error(state, f"质量评估失败: {str(e)}")


if __name__ == "__main__":
    print("QualityJudgeAgent需要真实的图片数据才能完整测试")
    print("请通过workflow.py中的完整流程进行测试")
