"""
兜底Agent - FallbackAgent

负责处理各种异常情况，保证流程不崩溃。

【职责】
1. 根据错误类型选择兜底策略
2. 检索失败 → 使用通用风格模板
3. 生图失败 → 返回占位提示
4. 其他错误 → 记录日志并返回友好结果
5. 确保最终能生成final_result

【输入】state字段
- error_msg: 错误消息
- current_step: 失败的步骤名称
- 其他已有数据（用于兜底处理）

【输出】state字段（新增/更新）
- final_result: 兜底结果
- status: 更新为completed（即使出错也完成）
- evidence_chain: 追加兜底记录
- metrics: 记录fallback_triggered
"""
import time
from pathlib import Path
from typing import Dict, Any

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from utils.core import save_image

# 【v2.2新增】导入提示词工程
try:
    from prompts.v2 import get_metrics, record_prompt_execution
    PROMPTS_V2_AVAILABLE = True
except ImportError:
    PROMPTS_V2_AVAILABLE = False


class FallbackAgent(BaseAgent):
    """
    兜底Agent

    在任何步骤出错时执行，保证流程不中断。

    【设计原则】
    1. 永远不抛出异常
    2. 尽可能利用已有数据生成结果
    3. 记录详细的错误信息供调试
    4. 返回用户友好的错误提示

    【兜底策略】
    - 检索失败：使用默认风格模板生图
    - 生图失败：返回占位图片或友好提示
    - 其他失败：返回错误信息但标记为completed
    """

    def __init__(self):
        """
        初始化兜底Agent

        【v2.2】集成提示词版本管理
        """
        super().__init__("FallbackAgent")
        OUTPUT_DIR.mkdir(exist_ok=True)

        # 【v2.2新增】设置提示词版本
        self.set_prompt_version("2.0")

    @time_decorator("fallback_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行兜底处理流程

        【v2.2】集成提示词工程：
        - 记录提示词版本
        - 追踪执行时间
        - 记录错误信息

        Args:
            state: 包含错误信息的状态

        Returns:
            更新后的状态，包含final_result
        """
        import time
        start_time = time.time()

        self._update_status(state, "processing", "FallbackAgent")

        # 【v2.2新增】记录提示词版本
        self._log_prompt_version(state)

        # ==================== 1. 记录兜底触发 ====================
        error_msg = state.get("error_msg", "未知错误")
        current_step = state.get("current_step", "unknown")

        self._add_evidence(
            state,
            f"触发兜底: 失败步骤={current_step}, 错误={error_msg}"
        )
        self._log_metric(state, "fallback_triggered", 1)
        self._log_metric(state, "error_step", current_step)

        # ==================== 2. 根据失败步骤选择策略 ====================
        if "retrieval" in current_step.lower() or "embedding" in current_step.lower():
            result = self._handle_retrieval_error(state, error_msg)
        elif "image" in current_step.lower() or "gen" in current_step.lower():
            result = self._handle_gen_error(state, error_msg)
        else:
            result = self._handle_generic_error(state, error_msg)

        # 【v2.2新增】记录提示词执行（兜底场景）
        self._record_prompt_execution(
            result,
            success=False,  # 兜底总是失败场景
            execution_time=time.time() - start_time,
            error=error_msg,
            metadata={"fallback_type": current_step}
        )

        return result

    def _handle_retrieval_error(self, state: PipelineState, error_msg: str) -> PipelineState:
        """
        处理检索相关的错误

        策略：使用默认风格模板，跳过检索步骤直接生图
        """
        self._add_evidence(
            state,
            "兜底策略: 使用默认风格模板（跳过检索）"
        )

        # 设置默认风格提示词
        state["style_prompt"] = (
            "专业电商宣传照，简洁纯色背景，柔和均匀光线，"
            "模特全身站立姿势，自然展示服装，8K高清，商业布光"
        )

        # 设置空的检索结果
        state["retrieved_results"] = []
        state["ref_images"] = []

        # 如果已经有新品图片，可以尝试直接生图
        if state.get("new_image"):
            self._add_evidence(
                state,
                "已有新品图片，可继续生图流程"
            )
            # 不返回错误，让流程继续
            return state
        else:
            # 连新品图片都没有，直接返回错误结果
            return self._create_error_result(state, error_msg, "检索失败且无新品图片")

    def _handle_gen_error(self, state: PipelineState, error_msg: str) -> PipelineState:
        """
        处理生图相关的错误

        策略：保存已有数据，返回友好错误提示
        """
        self._add_evidence(
            state,
            "兜底策略: 保存已有数据，返回错误提示"
        )

        # 创建错误结果
        return self._create_error_result(
            state,
            error_msg,
            "图像生成失败，可能是API错误或余额不足"
        )

    def _handle_generic_error(self, state: PipelineState, error_msg: str) -> PipelineState:
        """
        处理其他通用错误

        策略：保存已有数据，返回错误提示
        """
        self._add_evidence(
            state,
            "兜底策略: 保存已有数据，返回通用错误提示"
        )

        return self._create_error_result(
            state,
            error_msg,
            "处理过程中发生错误"
        )

    def _create_error_result(
        self,
        state: PipelineState,
        error_msg: str,
        user_message: str
    ) -> PipelineState:
        """
        创建错误结果

        Args:
            state: 当前状态
            error_msg: 技术错误信息
            user_message: 用户友好的错误描述

        Returns:
            包含错误信息的final_result状态
        """
        product_id = state.get("product_id", "unknown")

        # 创建输出目录
        output_dir = OUTPUT_DIR / product_id
        output_dir.mkdir(exist_ok=True)

        # 保存已有图片（如果有）
        saved_images = []
        new_image = state.get("new_image")
        if new_image:
            original_path = output_dir / f"{product_id}_original.png"
            save_image(new_image, str(original_path))
            saved_images.append(f"/api/output/{product_id}/{original_path.name}")

        # 保存检索结果（如果有）
        ref_images_saved = []
        retrieved_results = state.get("retrieved_results", [])
        for i, ref in enumerate(retrieved_results):
            if ref.get("image"):
                ref_path = output_dir / f"{product_id}_reference_{i+1}.png"
                save_image(ref["image"], str(ref_path))
                ref_images_saved.append(f"/api/output/{product_id}/{ref_path.name}")

        # 整理错误结果
        final_result = {
            "product_id": product_id,
            "success": False,
            "error": error_msg,
            "user_message": user_message,
            "category": state.get("category", ""),
            "style": state.get("style", ""),
            "output_dir": str(output_dir),
            "saved_images": saved_images,
            "reference_images": ref_images_saved,
            "evidence_chain": state.get("evidence_chain", []),
            "metrics": state.get("metrics", {})
        }

        state["final_result"] = final_result
        state["status"] = "completed"  # 即使失败也标记为completed，让任务结束

        self._add_evidence(state, f"错误结果已生成: {user_message}")

        # 记录总耗时
        created_at = state.get("created_at", time.time())
        total_time = time.time() - created_at
        self._log_metric(state, "total_time", round(total_time, 2))

        return state


if __name__ == "__main__":
    # 测试兜底Agent
    from .state import create_initial_state

    state = create_initial_state(
        task_id="test_fallback",
        file_bytes=b"test",
        category="midi_dress",
        style="elegant"
    )
    state["status"] = "failed"
    state["error_msg"] = "测试错误: 检索失败"
    state["current_step"] = "HybridRetrievalAgent"

    agent = FallbackAgent()
    result = agent.run(state)

    print("\n兜底Agent测试结果：")
    print(f"  status: {result['status']}")
    print(f"  final_result: {result.get('final_result', {}).get('user_message')}")
    print(f"  metrics: {result['metrics']}")
