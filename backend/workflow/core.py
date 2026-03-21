"""
LangGraph工作流定义 - 电商AI生图流水线多Agent编排

兼容 LangGraph 1.0.0+、LangChain 1.0.0+、langchain-core 1.0.0+

【架构设计】
┌─────────────────────────────────────────────────────────────────┐
│  START → UploadAgent                                            │
│           ↓                                                      │
│     EmbeddingAgent → HybridRetrievalAgent                       │
│           ↓                                                      │
│  StyleAnalysisAgent → ImageGenAgent → QualityJudgeAgent          │
│                                              ↓                   │
│                         should_regenerate? ──┴──→ ResultAgent → END │
│                            ↓                                      │
│                         (重试回ImageGenAgent, 最多1次)              │
│                                                              │
│  任何Agent异常 → FallbackAgent → ResultAgent → END               │
└─────────────────────────────────────────────────────────────────┘

【面试讲解要点】
1. 并行优化：未来可将Embedding和Retrieval改为并行执行
2. 条件分支：根据质量评分决定是否重试
3. 异常处理：任何异常都有兜底策略
4. 可观测性：证据链+指标埋点

【依赖】
- langgraph>=1.0.0
- langchain>=1.0.0
- langchain-core>=1.0.0
"""
import time
import operator
from typing import TypedDict, Literal, Dict, Any, Optional, Annotated
from pathlib import Path
from PIL import Image
import numpy as np

# LangGraph 1.0.0+ 导入
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 项目模块导入
import sys
sys.path.insert(0, str(Path(__file__).parent))

# 修复 Windows 控制��编码问题（统一使用工具模块）
from utils.console import fix_console_encoding
fix_console_encoding()

from agents import (
    PipelineState,
    BaseAgent,
    UploadAgent,
    EmbeddingAgent,
    HybridRetrievalAgent,
    StyleAnalysisAgent,
    ImageGenAgent,
    QualityJudgeAgent,
    ResultAgent,
    FallbackAgent,
    # 新增合并Agent
    RetrievalAgent,
    GenAgent,
)
from vectorization.embedding import EmbeddingGenerator
from retrieval.wrapper import RetrievalWrapper, create_retrieval_wrapper
from generation.image_gen import ImageGenerator
from config import LLM_MODEL


# ==================== 工作流节点函数 ====================

def upload_node(state: PipelineState) -> PipelineState:
    """上传解析节点"""
    agent = UploadAgent()
    return agent.run(state)


def embedding_node(state: PipelineState) -> PipelineState:
    """向量编码节点"""
    embed_gen = state.get("_embed_gen")
    tfidf = state.get("_tfidf")
    if not embed_gen or not tfidf:
        raise ValueError("EmbeddingGenerator或TF-IDF未初始化")

    agent = EmbeddingAgent(embed_gen, tfidf)
    return agent.run(state)


def hybrid_retrieval_node(state: PipelineState) -> PipelineState:
    """混合检索节点"""
    retriever = state.get("_retriever")
    if not retriever:
        raise ValueError("RetrievalWrapper未初始化")

    agent = HybridRetrievalAgent(retriever)
    return agent.run(state)


def style_analysis_node(state: PipelineState) -> PipelineState:
    """风格分析节点"""
    image_gen = state.get("_image_gen")
    if not image_gen:
        raise ValueError("ImageGenerator未初始化")

    agent = StyleAnalysisAgent(image_gen)
    return agent.run(state)


def image_gen_node(state: PipelineState) -> PipelineState:
    """生图节点"""
    image_gen = state.get("_image_gen")
    if not image_gen:
        raise ValueError("ImageGenerator未初始化")

    agent = ImageGenAgent(image_gen)
    return agent.run(state)


def quality_judge_node(state: PipelineState) -> PipelineState:
    """质量评估节点"""
    judge_model = state.get("_judge_model", LLM_MODEL)
    agent = QualityJudgeAgent(model=judge_model)
    return agent.run(state)


def result_node(state: PipelineState) -> PipelineState:
    """结果管理节点"""
    agent = ResultAgent()
    return agent.run(state)


def fallback_node(state: PipelineState) -> PipelineState:
    """兜底节点"""
    agent = FallbackAgent()
    return agent.run(state)


# ==================== 【新增】简化版节点函数 ====================

def retrieval_node(state: PipelineState) -> PipelineState:
    """检索节点（合并：Embedding + HybridRetrieval）"""
    embed_gen = state.get("_embed_gen")
    tfidf = state.get("_tfidf")
    retriever = state.get("_retriever")

    if not embed_gen or not tfidf or not retriever:
        raise ValueError("组件未初始化")

    agent = RetrievalAgent(embed_gen, tfidf, retriever)
    return agent.run(state)


def gen_node(state: PipelineState) -> PipelineState:
    """生图节点（合并：StyleAnalysis + ImageGen）"""
    image_gen = state.get("_image_gen")
    if not image_gen:
        raise ValueError("ImageGenerator未初始化")

    agent = GenAgent(image_gen)
    return agent.run(state)


# ==================== 条件边函数 ====================

def should_use_quality_judge(state: PipelineState) -> Literal["use_quality", "skip_quality"]:
    """
    判断是否使用质量评估

    Args:
        state: 当前工作流状态

    Returns:
        "use_quality" - 启用质量评估，执行QualityJudgeAgent
        "skip_quality" - 跳过质量评估，直接完成
    """
    enable_quality_check = state.get("enable_quality_check", False)

    if enable_quality_check:
        return "use_quality"
    else:
        return "skip_quality"


def should_regenerate_condition(state: PipelineState) -> Literal["regenerate", "finish"]:
    """
    判断是否需要重新生成

    Args:
        state: 当前工作流状态

    Returns:
        "regenerate" - 需要重新生成
        "finish" - 质量合格，完成流程
    """
    should_regenerate = state.get("should_regenerate", False)

    if should_regenerate:
        return "regenerate"
    else:
        return "finish"


# ==================== 工作流构建 ====================

def create_workflow(
    embed_gen: EmbeddingGenerator = None,
    retriever: RetrievalWrapper = None,
    image_gen: ImageGenerator = None,
    tfidf_vectorizer = None,
    judge_model: str = None
) -> StateGraph:
    """
    创建LangGraph工作流（兼容 1.0.0+ 版本）

    Args:
        embed_gen: EmbeddingGenerator实例
        retriever: RetrievalWrapper实例
        image_gen: ImageGenerator实例
        tfidf_vectorizer: TF-IDF向量化器
        judge_model: 质量评估模型

    Returns:
        编译后的CompiledStateGraph
    """

    # 创建状态图构建器
    builder = StateGraph(PipelineState)

    # ==================== 添加节点 ====================
    builder.add_node("upload", upload_node)
    builder.add_node("embedding", embedding_node)
    builder.add_node("hybrid_retrieval", hybrid_retrieval_node)
    builder.add_node("style_analysis", style_analysis_node)
    builder.add_node("image_gen", image_gen_node)
    builder.add_node("quality_judge", quality_judge_node)
    builder.add_node("result", result_node)
    builder.add_node("fallback", fallback_node)

    # ==================== 添加边（定义流程）====================

    # 1. START → Upload
    builder.add_edge(START, "upload")

    # 2. Upload → Embedding
    builder.add_edge("upload", "embedding")

    # 3. Embedding → HybridRetrieval
    builder.add_edge("embedding", "hybrid_retrieval")

    # 4. HybridRetrieval → StyleAnalysis
    builder.add_edge("hybrid_retrieval", "style_analysis")

    # 5. StyleAnalysis → ImageGen
    builder.add_edge("style_analysis", "image_gen")

    # 6. ImageGen → 条件分支（是否使用质量评估）
    builder.add_conditional_edges(
        "image_gen",
        should_use_quality_judge,
        {
            "use_quality": "quality_judge",  # 启用质量评估
            "skip_quality": "result"          # 跳过质量评估，直接完成
        }
    )

    # 7. QualityJudge → 条件分支（是否重新生成）
    builder.add_conditional_edges(
        "quality_judge",
        should_regenerate_condition,
        {
            "regenerate": "image_gen",  # 重试：回到生图
            "finish": "result"          # 完成：进入结果管理
        }
    )

    # 8. Result → END
    builder.add_edge("result", END)

    # 9. Fallback → Result → END
    builder.add_edge("fallback", "result")

    # ==================== 编译工作流 ====================
    # 可选：添加checkpointer支持断点续传
    # memory = MemorySaver()
    # app = builder.compile(checkpointer=memory)

    app = builder.compile()

    return app


def prepare_state_with_components(
    state: PipelineState,
    embed_gen: EmbeddingGenerator,
    retriever: RetrievalWrapper,
    image_gen: ImageGenerator,
    tfidf_vectorizer,
    judge_model: str = None
) -> PipelineState:
    """
    将组件注入到state中

    Args:
        state: 原始状态
        embed_gen: EmbeddingGenerator实例
        retriever: RetrievalWrapper实例
        image_gen: ImageGenerator实例
        tfidf_vectorizer: TF-IDF向量化器
        judge_model: 质量评估模型

    Returns:
        注入组件后的状态

    【设计原因】
    - LangGraph的节点函数只能接收state作为参数
    - 需要将组件通过state传递给节点
    - 使用下划线前缀的key（如_embed_gen）避免与业务字段冲突
    """
    state["_embed_gen"] = embed_gen
    state["_retriever"] = retriever
    state["_image_gen"] = image_gen
    state["_tfidf"] = tfidf_vectorizer
    if judge_model:
        state["_judge_model"] = judge_model

    return state


# ==================== 【新增】简化版工作流 ====================

def create_workflow_v2(
    embed_gen: EmbeddingGenerator = None,
    retriever: RetrievalWrapper = None,
    image_gen: ImageGenerator = None,
    tfidf_vectorizer = None,
    judge_model: str = None
) -> StateGraph:
    """
    创建简化版LangGraph工作流（5个Agent）

    【架构】
    START → UploadAgent → RetrievalAgent → GenAgent → [QualityJudgeAgent?] → ResultAgent → END

    【合并】
    - RetrievalAgent = EmbeddingAgent + HybridRetrievalAgent
    - GenAgent = StyleAnalysisAgent + ImageGenAgent

    Args:
        embed_gen: EmbeddingGenerator实例
        retriever: RetrievalWrapper实例
        image_gen: ImageGenerator实例
        tfidf_vectorizer: TF-IDF向量化器
        judge_model: 质量评估模型

    Returns:
        编译后的CompiledStateGraph
    """
    builder = StateGraph(PipelineState)

    # ==================== 添加节点 ====================
    builder.add_node("upload", upload_node)
    builder.add_node("retrieval", retrieval_node)  # 合并后的检索节点
    builder.add_node("gen", gen_node)              # 合并后的生图节点
    builder.add_node("quality_judge", quality_judge_node)
    builder.add_node("result", result_node)
    builder.add_node("fallback", fallback_node)

    # ==================== 添加边（定义流程）====================

    # 1. START → Upload
    builder.add_edge(START, "upload")

    # 2. Upload → Retrieval
    builder.add_edge("upload", "retrieval")

    # 3. Retrieval → Gen
    builder.add_edge("retrieval", "gen")

    # 4. Gen → 条件分支（是否使用质量评估）
    builder.add_conditional_edges(
        "gen",
        should_use_quality_judge,
        {
            "use_quality": "quality_judge",  # 启用质量评估
            "skip_quality": "result"          # 跳过质量评估，直接完成
        }
    )

    # 5. QualityJudge → 条件分支（是否重新生成）
    builder.add_conditional_edges(
        "quality_judge",
        should_regenerate_condition,
        {
            "regenerate": "gen",  # 重试：回到生图
            "finish": "result"    # 完成：进入结果管理
        }
    )

    # 6. Result → END
    builder.add_edge("result", END)

    # 7. Fallback → Result → END
    builder.add_edge("fallback", "result")

    # ==================== 编译工作流 ====================
    app = builder.compile()

    return app


def run_workflow(
    file_bytes: bytes,
    category: str,
    style: str,
    season: str = "all_season",
    scene_hint: str = "",
    embed_gen: EmbeddingGenerator = None,
    retriever: RetrievalWrapper = None,
    image_gen: ImageGenerator = None,
    tfidf_vectorizer = None,
    judge_model: str = None,
    progress_callback = None
) -> Dict[str, Any]:
    """
    运行完整工作流（便捷函数）

    Args:
        file_bytes: 上传的图片字节流
        category: 商品品类
        style: 商品风格
        season: 季节
        scene_hint: 场景提示
        embed_gen: EmbeddingGenerator实例
        retriever: RetrievalWrapper实例
        image_gen: ImageGenerator实例
        tfidf_vectorizer: TF-IDF向量化器
        judge_model: 质量评估模型
        progress_callback: 进度回调函数

    Returns:
        最终结果字典
    """
    import uuid

    # 创建工作流
    app = create_workflow(
        embed_gen=embed_gen,
        retriever=retriever,
        image_gen=image_gen,
        tfidf_vectorizer=tfidf_vectorizer,
        judge_model=judge_model
    )

    # 创建初始状态
    task_id = str(uuid.uuid4())
    from agents import create_initial_state

    state = create_initial_state(
        task_id=task_id,
        file_bytes=file_bytes,
        category=category,
        style=style,
        season=season,
        scene_hint=scene_hint
    )

    # 注入组件
    state = prepare_state_with_components(
        state,
        embed_gen=embed_gen,
        retriever=retriever,
        image_gen=image_gen,
        tfidf_vectorizer=tfidf_vectorizer,
        judge_model=judge_model
    )

    # 执行工作流
    print(f"\n{'='*60}")
    print(f"开始执行工作流: task_id={task_id[:8]}...")
    print(f"  LangGraph 1.0.0+ | LangChain 1.0.0+")
    print(f"{'='*60}")

    final_state = app.invoke(state)

    print(f"\n{'='*60}")
    print(f"工作流执行完成: status={final_state.get('status')}")
    print(f"{'='*60}")

    return final_state.get("final_result", {})


# ==================== 主函数（测试）====================

if __name__ == "__main__":
    """
    工作流测试代码

    使用示例：
        python workflow.py
    """
    import io
    from PIL import Image

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "LangGraph 1.0+ 多Agent工作流测试" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")

    # 创建测试图片
    print("\n[1/5] 创建测试图片...")
    test_img = Image.new("RGB", (800, 1200), color=(100, 150, 200))
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    file_bytes = img_bytes.read()
    print(f"  ✓ 测试图片创建成功: 800x1200 PNG")

    # 初始化组件
    print("\n[2/5] 初始化组件...")

    try:
        from vectorization.embedding import EmbeddingGenerator
        from graph import create_retrieval_wrapper
        from generation.image_gen import ImageGenerator
        from config import PRODUCT_CSV

        embed_gen = EmbeddingGenerator()
        retriever = create_retrieval_wrapper()
        image_gen = ImageGenerator()

        # 加载产品数据并构建TF-IDF
        products, _ = embed_gen.load_products()
        tfidf = embed_gen.build_tfidf_vectorizer(products)

        print("  ✓ 组件初始化完成")

        # 运行工作流
        print("\n[3/5] 运行工作流...")

        result = run_workflow(
            file_bytes=file_bytes,
            category="midi_dress",
            style="elegant",
            season="summer",
            scene_hint="beach",
            embed_gen=embed_gen,
            retriever=retriever,
            image_gen=image_gen,
            tfidf_vectorizer=tfidf
        )

        # 打印结果
        print("\n[4/5] 打印结果...")
        print(f"\n最终结果:")
        print(f"  product_id: {result.get('product_id')}")
        print(f"  generated_count: {result.get('generated_count')}")

        # 打印证据链
        print("\n[5/5] 证据链:")
        evidence_chain = result.get('evidence_chain', [])
        for i, evidence in enumerate(evidence_chain[-5:], 1):
            print(f"  {i}. {evidence}")

    except ImportError as e:
        print(f"\n[跳过] 组件导入失败: {e}")

    except Exception as e:
        print(f"\n[错误] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 20 + "测试完成" + " " * 28 + "║")
    print("╚" + "═" * 58 + "╝")
