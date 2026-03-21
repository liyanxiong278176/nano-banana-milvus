"""
多Agent工作流编排模块

基于LangGraph实现电商AI生图流水线的多Agent架构，
实现解耦、容错、并行执行，并提供全链路证据链追踪和核心指标埋点。

【简化架构 v2.0】
- UploadAgent: 文件上传和验证
- RetrievalAgent: 向量编码 + 向量检索（合并）
- GenAgent: 风格分析 + 图像生成（合并）
- QualityJudgeAgent: 质量评估（可选）
- ResultAgent: 结果整理
- FallbackAgent: 异常处理

使用示例：
    from agents import create_workflow, PipelineState
    from agents.base import BaseAgent

    # 创建工作流
    workflow = create_workflow()
    app = workflow.compile()

    # 执行工作流
    initial_state = PipelineState(task_id="xxx", category="midi_dress", ...)
    final_state = app.invoke(initial_state)
"""
from .state import PipelineState, create_initial_state
from .base import BaseAgent

# 保留原有Agent（向后兼容）
from .upload_agent import UploadAgent
from .embedding_agent import EmbeddingAgent
from .hybrid_retrieval_agent import HybridRetrievalAgent
from .style_analysis_agent import StyleAnalysisAgent
from .image_gen_agent import ImageGenAgent
from .quality_judge_agent import QualityJudgeAgent
from .result_agent import ResultAgent
from .fallback_agent import FallbackAgent

# 新增合并Agent（推荐使用）
from .retrieval_agent import RetrievalAgent
from .gen_agent import GenAgent

__all__ = [
    # 基础
    "PipelineState",
    "create_initial_state",
    "BaseAgent",
    # 原有Agent（向后兼容）
    "UploadAgent",
    "EmbeddingAgent",
    "HybridRetrievalAgent",
    "StyleAnalysisAgent",
    "ImageGenAgent",
    "QualityJudgeAgent",
    "ResultAgent",
    "FallbackAgent",
    # 新Agent（推荐）
    "RetrievalAgent",
    "GenAgent",
]
