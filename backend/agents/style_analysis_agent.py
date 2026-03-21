"""
风格分析Agent - StyleAnalysisAgent

负责使用LLM分析爆款参考图的拍摄风格。

【职责】
1. 调用ImageGenerator.analyze_style_with_llm()分析风格
2. 提取拍摄场景、光线、构图等风格特征
3. 利用缓存机制避免重复分析相同图片
4. 生成风格提示词供生图使用

【输入】state字段
- ref_images: 参考爆款图片列表
- category, style: 品类和风格（辅助分析）

【输出】state字段（新增/更新）
- style_prompt: 风格提示词（拍摄场景、光线、构图描述）
- evidence_chain: 追加证据
- metrics: 记录style_analysis_time, cache_hit, individual_count
"""
from typing import List, Dict

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from generation.image_gen import ImageGenerator

# 【v2.2新增】导入提示词工程
try:
    from prompts.v2 import get_metrics, record_prompt_execution
    PROMPTS_V2_AVAILABLE = True
except ImportError:
    PROMPTS_V2_AVAILABLE = False


class StyleAnalysisAgent(BaseAgent):
    """
    风格分析Agent

    使用LLM分析爆款参考图的拍摄风格，生成风格提示词。

    【复用原有逻辑】
    - 复用image_gen.py的ImageGenerator.analyze_style_with_llm()
    - 支持缓存机制（相同图片组合直接返回缓存结果）
    - 只提取拍摄风格，不描述服装款式
    """

    def __init__(self, image_gen: ImageGenerator):
        """
        初始化风格分析Agent

        【v2.2】集成提示词版本管理

        Args:
            image_gen: ImageGenerator实例（复用现有模块）
        """
        super().__init__("StyleAnalysisAgent")
        self.image_gen = image_gen

        # 【v2.2新增】设置提示词版本
        self.set_prompt_version("2.0")

    @time_decorator("style_analysis_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行风格分析流程

        【v2.2】集成提示词工程：
        - 记录提示词版本
        - 追踪执行时间
        - 记录缓存命中情况

        Args:
            state: 包含ref_images的状态

        Returns:
            更新后的状态，包含style_prompt
        """
        import time
        start_time = time.time()

        self._update_status(state, "processing", "StyleAnalysisAgent")

        # 【v2.2新增】记录提示词版本
        self._log_prompt_version(state)

        try:
            # ==================== 1. 验证输入 ====================
            ref_images = state.get("ref_images", [])

            if not ref_images:
                self._add_evidence(state, "无参考图片，使用默认风格模板")
                # 使用默认风格模板
                default_prompt = "专业电商宣传照，简洁背景，柔和光线，模特全身展示"
                state["style_prompt"] = default_prompt

                # 记录执行（无参考图的情况）
                self._record_prompt_execution(
                    state,
                    success=True,
                    execution_time=time.time() - start_time,
                    metadata={"cache_hit": False, "ref_count": 0}
                )
                return state

            self._add_evidence(state, f"开始分析风格: {len(ref_images)}张参考图")

            # ==================== 2. 调用LLM分析风格 ====================
            style_analysis = self.image_gen.analyze_style_with_llm(
                reference_images=ref_images
            )

            # ==================== 3. 提取风格提示词 ====================
            cache_hit = False

            if isinstance(style_analysis, dict):
                style_prompt = style_analysis.get("combined_style", "")
                individual_analyses = style_analysis.get("individual_analyses", [])

                # 记录是否命中缓存
                cache_hit = len(individual_analyses) == 0  # 空列表说明来自缓存
                self._log_metric(state, "cache_hit", 1 if cache_hit else 0)

                self._add_evidence(
                    state,
                    f"缓存{'命中' if cache_hit else '未命中'}"
                )

                # 记录每张图的分析结果（如果有）
                for i, analysis in enumerate(individual_analyses, 1):
                    self._add_evidence(state, f"参考图{i}风格: {analysis}")

                self._log_metric(state, "individual_count", len(individual_analyses))

            else:
                # 兼容旧格式（直接返回字符串）
                style_prompt = str(style_analysis)
                self._log_metric(state, "cache_hit", 0)
                self._log_metric(state, "individual_count", 0)

            if not style_prompt:
                # 记录失败
                self._record_prompt_execution(
                    state,
                    success=False,
                    execution_time=time.time() - start_time,
                    error="风格提示词生成失败"
                )
                raise ValueError("风格提示词生成失败")

            # ==================== 4. 记录风格分析结果 ====================
            self._add_evidence(
                state,
                f"风格提示词生成完成: {style_prompt[:100]}..."
            )

            # 提取关键风格特征到证据链
            self._extract_style_features(state, style_prompt)

            # ==================== 5. 更新状态 ====================
            state["style_prompt"] = style_prompt

            # 【v2.2新增】记录提示词执行成功
            self._record_prompt_execution(
                state,
                success=True,
                execution_time=time.time() - start_time,
                metadata={
                    "cache_hit": cache_hit,
                    "ref_count": len(ref_images),
                    "prompt_length": len(style_prompt)
                }
            )

            return state

        except Exception as e:
            # 出错时使用默认风格模板，保证流程继续
            self._add_evidence(state, f"风格分析失败: {e}，使用默认风格模板")
            default_prompt = "专业电商宣传照，简洁背景，柔和光线"
            state["style_prompt"] = default_prompt

            # 【v2.2新增】记录失败
            self._record_prompt_execution(
                state,
                success=False,
                execution_time=time.time() - start_time,
                error=str(e)
            )

            return state

    def _extract_style_features(self, state: PipelineState, style_prompt: str):
        """
        从风格提示词中提取关键特征

        Args:
            state: 当前状态
            style_prompt: 风格提示词
        """
        # 关键词提取（简单实现）
        features = {
            "场景": [],
            "光线": [],
            "构图": []
        }

        scene_keywords = ["室内", "室外", "街拍", "工作室", "背景", "场景"]
        lighting_keywords = ["光", "亮", "影", "色调"]
        pose_keywords = ["姿势", "站", "坐", "全身", "特写"]

        for keyword in scene_keywords:
            if keyword in style_prompt:
                features["场景"].append(keyword)

        for keyword in lighting_keywords:
            if keyword in style_prompt:
                features["光线"].append(keyword)

        for keyword in pose_keywords:
            if keyword in style_prompt:
                features["构图"].append(keyword)

        # 记录提取的特征
        feature_summary = ", ".join([
            f"{k}={'+'.join(v) if v else '未明确'}"
            for k, v in features.items()
        ])
        self._add_evidence(state, f"提取风格特征: {feature_summary}")


if __name__ == "__main__":
    print("StyleAnalysisAgent需要真实的ImageGenerator实例才能完整测试")
    print("请通过workflow.py中的完整流程进行测试")
