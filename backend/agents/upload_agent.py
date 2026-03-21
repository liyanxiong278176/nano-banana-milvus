"""
上传解析Agent - UploadAgent

负责处理用户上传的图片，验证格式、保存文件、生成唯一ID。

【职责】
1. 验证上传文件是否为有效图片
2. 将图片保存到new_products目录
3. 生成唯一的product_id和task_id
4. 保存商品元数据到CSV

【输入】state字段
- file_bytes: 上传的图片字节流
- category, style, season, scene_hint: 商品元数据

【输出】state字段（新增/更新）
- product_id: 生成的商品ID
- new_image: PIL Image对象
- evidence_chain: 追加证据
- metrics: 记录upload_time
"""
import io
import time
import uuid
from PIL import Image
from typing import Optional
from pathlib import Path

from .base import BaseAgent, time_decorator
from .state import PipelineState

# 导入项目配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NEW_PRODUCT_DIR, NEW_PRODUCT_CSV

# 【v2.2新增】导入提示词工程
try:
    from prompts.v2 import get_metrics, record_prompt_execution
    PROMPTS_V2_AVAILABLE = True
except ImportError:
    PROMPTS_V2_AVAILABLE = False


class UploadAgent(BaseAgent):
    """
    上传解析Agent

    处理用户上传的商品图片，完成文件验证、保存和元数据记录。

    【复用原有逻辑】
    - 复用api.py中的文件验证、保存、CSV写入逻辑
    - 保持与原有流程的完全兼容
    """

    def __init__(self):
        """
        初始化上传Agent

        【v2.2】集成提示词版本管理
        """
        super().__init__("UploadAgent")
        # 确保目录存在
        NEW_PRODUCT_DIR.mkdir(exist_ok=True)

        # 【v2.2新增】设置提示词版本
        self.set_prompt_version("2.0")

    @time_decorator("upload_time")
    def run(self, state: PipelineState) -> PipelineState:
        """
        执行上传解析流程

        【v2.2】集成提示词工程：
        - 记录提示词版本
        - 追踪执行时间

        Args:
            state: 包含file_bytes等上传数据的状态

        Returns:
            更新后的状态，包含product_id和new_image
        """
        import time
        start_time = time.time()

        self._update_status(state, "processing", "UploadAgent")

        # 【v2.2新增】记录提示词版本
        self._log_prompt_version(state)

        try:
            # ==================== 1. 文件验证 ====================
            file_bytes = state.get("file_bytes")
            if not file_bytes:
                raise ValueError("上传文件为空")

            # 尝试打开图片（验证格式）
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            except Exception as e:
                raise ValueError(f"无效的图片文件: {e}")

            file_size_mb = len(file_bytes) / (1024 * 1024)
            img_format = img.format or "PNG"
            img_size = f"{img.size[0]}x{img.size[1]}"

            self._add_evidence(
                state,
                f"文件验证通过: 格式={img_format}, 尺寸={img_size}, 大小={file_size_mb:.2f}MB"
            )

            # ==================== 2. 生成唯一ID ====================
            product_id = f"NEW_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            state["product_id"] = product_id

            self._add_evidence(state, f"生成商品ID: {product_id}")

            # ==================== 3. 保存图片文件 ====================
            img_path = NEW_PRODUCT_DIR / f"{product_id}.jpg"
            img.save(img_path, quality=90)

            self._add_evidence(state, f"图片已保存: {img_path.name}")

            # ==================== 4. 更新状态 ====================
            state["new_image"] = img

            # ==================== 5. 保存元数据到CSV ====================
            self._save_to_csv(
                product_id=product_id,
                category=state.get("category", ""),
                style=state.get("style", ""),
                season=state.get("season", "all_season"),
                scene_hint=state.get("scene_hint", "")
            )

            self._add_evidence(state, "元数据已保存到CSV")

            # ==================== 6. 记录指标 ====================
            self._log_metric(state, "file_size_mb", round(file_size_mb, 2))
            self._log_metric(state, "image_width", img.size[0])
            self._log_metric(state, "image_height", img.size[1])

            # 【v2.2新增】记录提示词执行成功
            self._record_prompt_execution(
                state,
                success=True,
                execution_time=time.time() - start_time,
                metadata={
                    "file_size_mb": round(file_size_mb, 2),
                    "image_format": img_format
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
            return self._handle_error(state, f"上传处理失败: {str(e)}")

    def _save_to_csv(
        self,
        product_id: str,
        category: str,
        style: str,
        season: str,
        scene_hint: str
    ):
        """
        保存新品元数据到CSV文件

        【复用api.py的save_to_csv逻辑】

        Args:
            product_id: 商品ID
            category: 品类
            style: 风格
            season: 季节
            scene_hint: 场景提示
        """
        import csv

        file_exists = NEW_PRODUCT_CSV.exists()
        fieldnames = ['new_id', 'image_path', 'category', 'style', 'season', 'prompt_hint']

        with open(NEW_PRODUCT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'new_id': product_id,
                'image_path': f"{product_id}.jpg",
                'category': category,
                'style': style,
                'season': season,
                'prompt_hint': scene_hint
            })


if __name__ == "__main__":
    # 测试UploadAgent
    from .state import create_initial_state

    # 创建一个测试图片字节流
    test_img = Image.new("RGB", (800, 1200), color=(200, 200, 200))
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # 创建初始状态
    state = create_initial_state(
        task_id="test_upload",
        file_bytes=img_bytes.read(),
        category="midi_dress",
        style="elegant",
        season="summer"
    )

    # 执行Agent
    agent = UploadAgent()
    result = agent.run(state)

    print("\nUploadAgent测试结果：")
    print(f"  product_id: {result.get('product_id')}")
    print(f"  new_image: {result.get('new_image')}")
    print(f"  status: {result['status']}")
    print(f"  metrics: {result['metrics']}")
    print(f"  evidence_chain:")
    for e in result['evidence_chain']:
        print(f"    - {e}")
