"""
生图Agent - GenAgent

【合并】StyleAnalysisAgent + ImageGenAgent

负责风格分析和图像生成的完整流程。

【职责】
1. 使用LLM分析爆款参考图的拍摄风格，生成风格提示词
2. 调用图像生成模型生成宣传图
3. 支持重试机制（质量不合格时重新生成）

【输入】state字段
- new_image: 新品平铺图
- ref_images: 参考爆款图
- scene_hint: 用户场景提示

【输出】state字段（新增/更新）
- style_prompt: 风格提示词
- generated_images: 生成的图片列表
- evidence_chain: 追加证据
- metrics: 记录style_analysis_time, image_gen_time, generated_count
"""
from typing import List

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from generation.image_gen import ImageGenerator
from config import DEFAULT_ASPECT_RATIO, DEFAULT_IMAGE_SIZE

# 【v2.2新增】导入提示词工程
try:
    from prompts.v2 import get_metrics, record_prompt_execution
    PROMPTS_V2_AVAILABLE = True
except ImportError:
    PROMPTS_V2_AVAILABLE = False


class GenAgent(BaseAgent):
    """
    生图Agent

    合并风格分析和图像生成功能，在一个Agent中完成完整生图流程。

    【优势】
    - 风格分析是生图的前置步骤，放一起更自然
    - 减少状态传递开销
    - 逻辑更紧凑，易于维护
    """

    def __init__(self, image_gen: ImageGenerator):
        """
        初始化生图Agent

        Args:
            image_gen: ImageGenerator实例
        """
        super().__init__("GenAgent")
        self.image_gen = image_gen

        # 【v2.2新增】设置提示词版本
        self.set_prompt_version("2.0")

    @time_decorator("gen_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行完整生图流程：风格分析 → 图像生成

        Args:
            state: 包含new_image, ref_images等的状态

        Returns:
            更新后的状态，包含style_prompt和generated_images
        """
        import time
        start_time = time.time()

        self._update_status(state, "processing", "GenAgent")

        # 【v2.2新增】记录提示词版本
        self._log_prompt_version(state)

        try:
            # ==================== Step 1: 风格分析 ====================
            ref_images = state.get("ref_images", [])

            if not ref_images:
                self._add_evidence(state, "[风格分析] 无参考图片，使用默认风格模板")
                style_prompt = "专业电商宣传照，简洁背景，柔和光线，模特全身展示"
            else:
                self._add_evidence(state, f"[风格分析] 开始分析: {len(ref_images)}张参考图")

                # 调用LLM分析风格
                style_analysis = self.image_gen.analyze_style_with_llm(
                    reference_images=ref_images
                )

                # 提取风格提示词
                if isinstance(style_analysis, dict):
                    style_prompt = style_analysis.get("combined_style", "")
                    individual_analyses = style_analysis.get("individual_analyses", [])

                    # 记录是否命中缓存
                    cache_hit = len(individual_analyses) == 0
                    self._log_metric(state, "cache_hit", 1 if cache_hit else 0)
                    self._add_evidence(state, f"[风格分析] 缓存{'命中' if cache_hit else '未命中'}")

                    # 记录每张图的分析结果
                    for i, analysis in enumerate(individual_analyses, 1):
                        self._add_evidence(state, f"[风格分析] 参考图{i}: {analysis}")

                else:
                    style_prompt = str(style_analysis)

                if not style_prompt:
                    self._add_evidence(state, "[风格分析] 生成失败，使用默认风格模板")
                    style_prompt = "专业电商宣传照，简洁背景，柔和光线"

            self._add_evidence(state, f"[风格分析] 完成: {style_prompt[:100]}...")

            # 更新状态
            state["style_prompt"] = style_prompt

            # ==================== Step 2: 图像生成 ====================
            new_image = state.get("new_image")
            if not new_image:
                raise ValueError("新品图片为空，请先执行UploadAgent")

            scene_hint = state.get("scene_hint", "")
            retry_count = state.get("retry_count", 0)
            is_retry = retry_count > 0

            self._add_evidence(
                state,
                f"[图像生成] 开始: {'重试' if is_retry else '首次生成'} (第{retry_count + 1}次)"
            )

            # 调用图像生成
            generated_images = self.image_gen.generate_promotional_photo(
                new_product_image=new_image,
                reference_images=ref_images,
                style_prompt=style_prompt,
                scene_hint=scene_hint,
                aspect_ratio=DEFAULT_ASPECT_RATIO,
                image_size=DEFAULT_IMAGE_SIZE
            )

            # 验证生成结果
            if not generated_images or len(generated_images) == 0:
                raise RuntimeError("图像生成失败（API错误或余额不足）")

            generated_count = len(generated_images)
            img_size = generated_images[0].size if generated_images else (0, 0)

            self._add_evidence(
                state,
                f"[图像生成] 完成: 生成{generated_count}张图片, 尺寸={img_size[0]}x{img_size[1]}"
            )

            # 更新状态
            state["generated_images"] = generated_images

            # 记录指标
            self._log_metric(state, "generated_count", generated_count)
            self._log_metric(state, "retry_count", retry_count)

            # 【v2.2新增】记录提示词执行成功
            self._record_prompt_execution(
                state,
                success=True,
                execution_time=time.time() - start_time,
                metadata={
                    "generated_count": generated_count,
                    "is_retry": is_retry,
                    "image_size": f"{img_size[0]}x{img_size[1]}",
                    "ref_count": len(ref_images)
                }
            )

            return state

        except Exception as e:
            # 出错时使用默认风格模板，保证流程继续
            self._add_evidence(state, f"[生图] 失败: {e}")

            # 【v2.2新增】记录失败
            self._record_prompt_execution(
                state,
                success=False,
                execution_time=time.time() - start_time,
                error=str(e)
            )

            return self._handle_error(state, f"生图失败: {str(e)}")


if __name__ == "__main__":
    print("GenAgent需要真实的ImageGenerator实例才能完整测试")
    print("请通过workflow.py中的完整流程进行测试")
