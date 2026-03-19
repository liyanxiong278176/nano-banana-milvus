"""
结果管理Agent - ResultAgent

负责保存生成结果并整理最终输出。

【职责】
1. 保存生成的图片到output目录
2. 保存原图和参考图
3. 保存风格提示词和质量评分
4. 整理最终结果供API返回

【输入】state字段
- product_id: 商品ID
- new_image: 原始新品图
- generated_images: 生成的图片列表
- best_image: 最佳图片
- retrieved_results: 检索结果
- ref_images: 参考图片
- style_prompt: 风格提示词
- quality_scores: 质量评分
- evidence_chain, metrics: 证据链和指标

【输出】state字段（新增/更新）
- final_result: 最终结果字典
- status: 更新为completed
- evidence_chain: 追加保存记录
- metrics: 记录total_time
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from utils import save_image


class ResultAgent(BaseAgent):
    """
    结果管理Agent

    保存生成结果并整理最终输出。

    【复用原有逻辑】
    - 复用api.py的保存逻辑
    - 保持与原有输出格式完全兼容
    """

    def __init__(self):
        """初始化结果Agent"""
        super().__init__("ResultAgent")
        OUTPUT_DIR.mkdir(exist_ok=True)

    @time_decorator("result_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行结果保存流程

        Args:
            state: 包含所有生成结果的状态

        Returns:
            更新后的状态，包含final_result
        """
        self._update_status(state, "processing", "ResultAgent")

        try:
            # ==================== 1. 验证输入 ====================
            product_id = state.get("product_id", "")
            if not product_id:
                raise ValueError("product_id为空")

            # 创建输出目录
            output_dir = OUTPUT_DIR / product_id
            output_dir.mkdir(exist_ok=True)

            self._add_evidence(state, f"创建输出目录: {output_dir.name}")

            # ==================== 2. 保存原图 ====================
            new_image = state.get("new_image")
            if new_image:
                original_path = output_dir / f"{product_id}_original.png"
                save_image(new_image, str(original_path))
                self._add_evidence(state, f"保存原图: {original_path.name}")

            # ==================== 3. 保存参考图 ====================
            retrieved_results = state.get("retrieved_results", [])
            ref_images_saved = []

            for i, ref in enumerate(retrieved_results):
                if ref.get("image"):
                    ref_path = output_dir / f"{product_id}_reference_{i+1}.png"
                    save_image(ref["image"], str(ref_path))
                    ref_images_saved.append(f"/api/output/{product_id}/{ref_path.name}")

            self._add_evidence(state, f"保存参考图: {len(ref_images_saved)}张")

            # ==================== 4. 保存生成图 ====================
            generated_images = state.get("generated_images", [])
            generated_paths = []

            for i, gen_img in enumerate(generated_images):
                gen_path = output_dir / f"{product_id}_generated_{i+1}.png"
                save_image(gen_img, str(gen_path))
                generated_paths.append(f"/api/output/{product_id}/{gen_path.name}")

            self._add_evidence(state, f"保存生成图: {len(generated_paths)}张")

            # ==================== 5. 保存风格提示词 ====================
            style_prompt = state.get("style_prompt", "")
            if style_prompt:
                prompt_path = output_dir / f"{product_id}_style_prompt.txt"
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(style_prompt)
                self._add_evidence(state, f"保存风格提示词: {prompt_path.name}")

            # ==================== 6. 保存质量评分 ====================
            quality_scores = state.get("quality_scores", {})
            if quality_scores:
                score_path = output_dir / f"{product_id}_quality_scores.json"
                with open(score_path, "w", encoding="utf-8") as f:
                    json.dump(quality_scores, f, ensure_ascii=False, indent=2)
                self._add_evidence(state, f"保存质量评分: {score_path.name}")

            # ==================== 7. 整理最终结果 ====================
            final_result = {
                "product_id": product_id,
                "category": state.get("category", ""),
                "style": state.get("style", ""),
                "season": state.get("season", ""),
                "retrieved_count": len(retrieved_results),
                "generated_count": len(generated_images),
                "style_prompt": style_prompt,
                "quality_scores": quality_scores,
                "generated_images": generated_paths,
                "reference_images": ref_images_saved,
                "output_dir": str(output_dir),
                "evidence_chain": state.get("evidence_chain", []),
                "metrics": state.get("metrics", {})
            }

            state["final_result"] = final_result

            # ==================== 8. 记录总耗时 ====================
            created_at = state.get("created_at", time.time())
            total_time = time.time() - created_at
            self._log_metric(state, "total_time", round(total_time, 2))

            self._add_evidence(
                state,
                f"结果保存完成: 总耗时={total_time:.2f}秒"
            )

            # ==================== 9. 更新状态为完成 ====================
            state["status"] = "completed"
            self._add_evidence(state, "任务完成")

            return state

        except Exception as e:
            return self._handle_error(state, f"结果保存失败: {str(e)}")


if __name__ == "__main__":
    print("ResultAgent需要完整的state才能完整测试")
    print("请通过workflow.py中的完整流程进行测试")
