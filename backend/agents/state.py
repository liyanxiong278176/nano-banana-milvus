"""
全局状态定义 - PipelineState

兼容 LangGraph 1.0.0+ 和 LangChain 1.0.0+

使用 TypedDict 定义工作流中所有Agent之间传递的状态数据。
这是LangGraph的核心概念，所有Agent都接收完整的State，更新后返回新的State。

【面试讲解要点】
1. TypedDict提供类型安全，IDE有自动补全
2. 所有Agent共享同一State，实现解耦
3. evidence_chain和metrics是亮点：全链路追踪+可观测性
"""
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

# LangGraph 1.0.0+ 推荐使用 typing_extensions
try:
    from typing_extensions import TypedDict, Required
except ImportError:
    from typing import TypedDict, Required


class PipelineState(TypedDict):
    """
    工作流全局状态

    包含所有Agent之间需要传递的数据字段。
    每个Agent只需更新自己负责的字段，其他字段保持不变。
    """

    # ==================== 任务元数据 ====================
    task_id: str                    # 任务唯一标识（UUID）
    product_id: str                 # 商品ID（如 NEW_123456_abc123）
    created_at: float               # 任务创建时间戳

    # ==================== 上传数据 ====================
    file_bytes: bytes               # 上传的图片文件字节流
    new_image: Optional[Image.Image]  # 新品平铺图（PIL Image对象）
    category: str                   # 商品品类（如 midi_dress, maxi_dress）
    style: str                      # 商品风格（如 elegant, casual）
    season: str                     # 季节（如 summer, winter, all_season）
    scene_hint: str                 # 场景提示（用户输入的额外描述）

    # ==================== 向量与检索 ====================
    query_dense: Optional[np.ndarray]  # 稠密查询向量（图片视觉特征，2048维）
    query_sparse: Optional[Dict[int, float]]  # 稀疏查询向量（文本TF-IDF特征）
    retrieved_results: List[Dict]     # 检索到的爆款商品列表（包含完整元数据）
    ref_images: List[Image.Image]    # 参考爆款图片列表（PIL Image对象）

    # ==================== 风格与生图 ====================
    style_prompt: str                # LLM生成的风格提示词（拍摄场景、光线、构图描述）
    generated_images: List[Image.Image]  # 生成的宣传图列表（可能有多张）
    best_image: Optional[Image.Image]   # 质量评估后的最佳图片

    # ==================== 质量评估 ====================
    enable_quality_check: bool       # 是否启用AI质量评估（前端控制）
    quality_scores: Dict[str, Any]   # 质量评分结果（各维度得分+总分）
    should_regenerate: bool          # 是否需要重新生成（评分低于阈值时为True）
    regenerate_reason: str           # 重新生成的原因说明（如"服装准确性不足"）
    retry_count: int                 # 当前重试次数（最多重试1次）

    # ==================== 证据链与指标（面试亮点）====================
    # 证据链：记录每一步的决策依据，方便调试和面试讲解
    # 例如："文件验证通过，格式PNG，大小2.1MB；生成任务ID：xxx"
    evidence_chain: List[str]

    # 指标埋点：记录各步骤的耗时和关键指标，方便性能分析
    # 例如：{"upload_time": 0.3, "embedding_time": 0.8, "retrieval_time": 1.2, ...}
    metrics: Dict[str, float]

    # ==================== 流程控制 ====================
    status: str                      # 当前状态：pending/processing/completed/failed
    error_msg: Optional[str]         # 错误信息（如果有）
    current_step: str                # 当前执行步骤名称（用于兜底Agent判断）

    # ==================== 最终结果 ====================
    final_result: Optional[Dict[str, Any]]  # 最终结果（包含所有输出文件路径和评分）

    # ==================== 组件注入（内部使用）====================
    # 以下字段用于在工作流中传递组件实例，使用下划线前缀避免与业务字段冲突
    # 注意：LangGraph 1.0+ TypedDict 需要包含所有可能的字段
    _embed_gen: Optional[Any]        # EmbeddingGenerator实例
    _retriever: Optional[Any]        # RetrievalWrapper实例
    _image_gen: Optional[Any]        # ImageGenerator实例
    _tfidf: Optional[Any]            # TF-IDF向量化器
    _judge_model: Optional[str]      # 质量评估模型名称


def create_initial_state(
    task_id: str,
    file_bytes: bytes,
    category: str,
    style: str,
    season: str = "all_season",
    scene_hint: str = "",
    enable_quality_check: bool = False
) -> PipelineState:
    """
    创建初始状态

    Args:
        task_id: 任务ID
        file_bytes: 上传的图片字节流
        category: 商品品类
        style: 商品风格
        season: 季节
        scene_hint: 场景提示
        enable_quality_check: 是否启用AI质量评估

    Returns:
        初始化的PipelineState字典
    """
    import time

    return PipelineState(
        # 任务元数据
        task_id=task_id,
        product_id="",  # 将由UploadAgent生成
        created_at=time.time(),

        # 上传数据
        file_bytes=file_bytes,
        new_image=None,
        category=category,
        style=style,
        season=season,
        scene_hint=scene_hint,

        # 向量与检索（初始为空）
        query_dense=None,
        query_sparse=None,
        retrieved_results=[],
        ref_images=[],

        # 风格与生图（初始为空）
        style_prompt="",
        generated_images=[],
        best_image=None,

        # 质量评估（初始为空）
        enable_quality_check=enable_quality_check,
        quality_scores={},
        should_regenerate=False,
        regenerate_reason="",
        retry_count=0,

        # 证据链与指标（初始化）
        evidence_chain=[],
        metrics={},

        # 流程控制
        status="pending",
        error_msg=None,
        current_step="",

        # 最终结果（初始为空）
        final_result=None,

        # 组件注入（初始为空，由 prepare_state_with_components 填充）
        _embed_gen=None,
        _retriever=None,
        _image_gen=None,
        _tfidf=None,
        _judge_model=None
    )


if __name__ == "__main__":
    # 测试状态创建
    state = create_initial_state(
        task_id="test_001",
        file_bytes=b"fake_image_bytes",
        category="midi_dress",
        style="elegant",
        season="summer"
    )

    print("初始状态创建成功：")
    print(f"  task_id: {state['task_id']}")
    print(f"  category: {state['category']}")
    print(f"  status: {state['status']}")
    print(f"  evidence_chain: {state['evidence_chain']}")
