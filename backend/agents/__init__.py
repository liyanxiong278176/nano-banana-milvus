"""
多Agent工作流编排模块

基于LangGraph实现电商AI生图流水线的多Agent架构，
实现解耦、容错、并行执行，并提供全链路证据链追踪和核心指标埋点。

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
from .upload_agent import UploadAgent
from .embedding_agent import EmbeddingAgent
from .hybrid_retrieval_agent import HybridRetrievalAgent
from .style_analysis_agent import StyleAnalysisAgent
from .image_gen_agent import ImageGenAgent
from .quality_judge_agent import QualityJudgeAgent
from .result_agent import ResultAgent
from .fallback_agent import FallbackAgent

__all__ = [
    "PipelineState",
    "create_initial_state",
    "BaseAgent",
    "UploadAgent",
    "EmbeddingAgent",
    "HybridRetrievalAgent",
    "StyleAnalysisAgent",
    "ImageGenAgent",
    "QualityJudgeAgent",
    "ResultAgent",
    "FallbackAgent",
]
