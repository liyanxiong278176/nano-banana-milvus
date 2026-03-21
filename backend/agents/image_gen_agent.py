"""
生图Agent - ImageGenAgent

负责调用图像生成模型生成宣传图。

【职责】
1. 调用ImageGenerator.generate_promotional_photo()生成图片
2. 支持重试机制（质量不合格时重新生成）
3. 记录生成参数和结果

【输入】state字段
- new_image: 新品平铺图
- ref_images: 参考爆款图（仅用于提取风格，不传给生成模型）
- style_prompt: LLM分析的风格提示词
- scene_hint: 用户场景提示

【输出】state字段（新增/更新）
- generated_images: 生成的图片列表
- evidence_chain: 追加证据
- metrics: 记录image_gen_time, generated_count
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


class ImageGenAgent(BaseAgent):
    """
    生图Agent

    调用图像生成模型生成电商宣传图。

    【复用原有逻辑】
    - 复用image_gen.py的ImageGenerator.generate_promotional_photo()
    - 第一性原则：只传用户服装图，不传参考图（避免服装变化）
    - 参考图的风格已通过LLM提取到style_prompt中
    """

    def __init__(self, image_gen: ImageGenerator):
        """
        初始化生图Agent

        【v2.2】集成提示词版本管理

        Args:
            image_gen: ImageGenerator实例（复用现有模块）
        """
        super().__init__("ImageGenAgent")
        self.image_gen = image_gen

        # 【v2.2新增】设置提示词版本
        self.set_prompt_version("2.0")

    @time_decorator("image_gen_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行图像生成流程

        【v2.2】集成提示词工程：
        - 记录提示词版本
        - 追踪执行时间
        - 记录生成结果

        Args:
            state: 包含new_image, style_prompt等的状态

        Returns:
            更新后的状态，包含generated_images
        """
        import time
        start_time = time.time()

        self._update_status(state, "processing", "ImageGenAgent")

        # 【v2.2新增】记录提示词版本
        self._log_prompt_version(state)

        try:
            # ==================== 1. 验证输入 ====================
            new_image = state.get("new_image")
            style_prompt = state.get("style_prompt", "")

            if not new_image:
                raise ValueError("新品图片为空，请先执行UploadAgent")

            if not style_prompt:
                self._add_evidence(state, "风格提示词为空，使用默认风格")
                style_prompt = "专业电商宣传照，简洁背景，柔和光线"

            # 获取重试次数
            retry_count = state.get("retry_count", 0)
            is_retry = retry_count > 0

            self._add_evidence(
                state,
                f"开始生成宣传图: {'重试' if is_retry else '首次生成'} (第{retry_count + 1}次)"
            )

            # ==================== 2. 准备生成参数 ====================
            ref_images = state.get("ref_images", [])
            scene_hint = state.get("scene_hint", "")

            # 注意：ref_images不传给生成模型，风格已通过style_prompt传递
            self._add_evidence(
                state,
                f"生成参数: 宽高比={DEFAULT_ASPECT_RATIO}, 分辨率={DEFAULT_IMAGE_SIZE}"
            )

            # ==================== 3. 调用图像生成 ====================
            generated_images = self.image_gen.generate_promotional_photo(
                new_product_image=new_image,
                reference_images=ref_images,  # 仅用于内部逻辑，模型不直接使用
                style_prompt=style_prompt,
                scene_hint=scene_hint,
                aspect_ratio=DEFAULT_ASPECT_RATIO,
                image_size=DEFAULT_IMAGE_SIZE
            )

            # ==================== 4. 验证生成结果 ====================
            if not generated_images or len(generated_images) == 0:
                raise RuntimeError("图像生成失败（API错误或余额不足）")

            generated_count = len(generated_images)
            img_size = generated_images[0].size if generated_images else (0, 0)

            self._add_evidence(
                state,
                f"生图完成: 生成{generated_count}张图片, 尺寸={img_size[0]}x{img_size[1]}"
            )

            # ==================== 5. 更新状态 ====================
            state["generated_images"] = generated_images

            # ==================== 6. 记录指标 ====================
            self._log_metric(state, "generated_count", generated_count)
            self._log_metric(state, "retry_count", retry_count)

            # 如果是重试，记录重试指标
            if is_retry:
                self._log_metric(state, "is_retry", 1)

            # 【v2.2新增】记录提示词执行成功
            self._record_prompt_execution(
                state,
                success=True,
                execution_time=time.time() - start_time,
                metadata={
                    "generated_count": generated_count,
                    "is_retry": is_retry,
                    "image_size": f"{img_size[0]}x{img_size[1]}"
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
            return self._handle_error(state, f"图像生成失败: {str(e)}")


if __name__ == "__main__":
    print("ImageGenAgent需要真实的ImageGenerator实例才能完整测试")
    print("请通过workflow.py中的完整流程进行测试")
