"""
FastAPI 后端服务 - 电商 AI 生��流水线 API
"""
import asyncio
import csv
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# 添加 backend 目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

# 修复 Windows 控制台编码问题（统一使用工具模块）
from utils.console import fix_console_encoding
fix_console_encoding()

# 配置日志系统（输出到 log_config 目录）
from log_config import setup_logging, get_logger
setup_logging("api", level=logging.INFO)
app_logger = get_logger("api")
app_logger.info("=" * 60)
app_logger.info("API 服务启动")
app_logger.info("=" * 60)

from config import (
    OPENROUTER_API_KEY, MILVUS_URI, COLLECTION_NAME,
    IMAGE_DIR, NEW_PRODUCT_DIR, OUTPUT_DIR, NEW_PRODUCT_CSV,
    DEFAULT_ASPECT_RATIO, DEFAULT_IMAGE_SIZE,
    USE_BM25, BM25_K1, BM25_B
)
from vectorization.embedding import EmbeddingGenerator
from retrieval.retrieval import BestsellerRetriever
from retrieval.wrapper import RetrievalWrapper, create_retrieval_wrapper
from generation.image_gen import ImageGenerator
from utils.core import save_image

# ==================== 【新增】LangGraph 工作流导入 ====================
# 导入工作流相关模块
try:
    from workflow.core import create_workflow_v2, prepare_state_with_components
    from agents import create_initial_state
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    print(f"警告: LangGraph 工作流模块导入失败: {e}")
    print("将使用原有线性流程")
    WORKFLOW_AVAILABLE = False

# ==================== 【新增】任务管理器导入 ====================
from task import TaskManager, get_task_manager, TaskStatus


# ==================== Pydantic 模型 ====================

class GenerationRequest(BaseModel):
    """生图请求"""
    category: str = Field(..., description="商品品类")
    style: str = Field(..., description="商品风格")
    season: str = Field(default="all_season", description="季节")
    scene_hint: str = Field(default="", description="场景提示")
    enable_quality_check: bool = Field(default=False, description="是否启用质量评估")
    judge_model: str = Field(default="", description="裁判模型（空则使用默认模型）")


class GenerationResponse(BaseModel):
    """生图响应"""
    task_id: str
    status: str
    product_id: str
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    progress: float
    result: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    milvus_connected: bool
    database_ready: bool


# ==================== 全局状态 ====================

app = FastAPI(
    title="电商 AI 生图流水线 API",
    description="基于向量检索和多模态 AIGC 的电商宣传图自动生成系统",
    version="1.0.0"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 任务管理器（替代原来的 tasks = {} 字典）
# 使用 get_task_manager() 获取全局单例
task_manager: TaskManager = None

# 流水线实例（延迟初始化）
pipeline_components = None


# ==================== 初始化 ====================

def init_pipeline():
    """初始化流水线组件"""
    global pipeline_components, task_manager

    if pipeline_components is not None:
        return pipeline_components

    print("初始化流水线组件...")

    # 初始化任务管理器
    if task_manager is None:
        task_manager = get_task_manager()
        print(f"任务管理器已初始化 (最大并发: {task_manager.max_concurrent})")

    # 使用检索包装器（Milvus 向量检索 + 循环检索状态机）
    retriever = create_retrieval_wrapper()
    embed_gen = EmbeddingGenerator(use_bm25=USE_BM25)
    image_gen = ImageGenerator()
    tfidf = None

    # 检查数据库是否已初始化
    # 混合检索器内部有 BestsellerRetriever，可以通过它访问 Milvus
    milvus_retriever = retriever.milvus_retriever
    if milvus_retriever.has_collection():
        stats = milvus_retriever.get_collection_stats()
        if stats.get('row_count', 0) > 0:
            print(f"数据库已就绪，包含 {stats['row_count']} 条记录")
            # 加载向量化器 (TF-IDF 或 BM25)
            products, _ = embed_gen.load_products()
            if USE_BM25:
                vectorizer = embed_gen.build_bm25_vectorizer(products, k1=BM25_K1, b=BM25_B)
            else:
                vectorizer = embed_gen.build_tfidf_vectorizer(products)
            tfidf = vectorizer  # 保持向后兼容

    pipeline_components = {
        'retriever': retriever,
        'embed_gen': embed_gen,
        'image_gen': image_gen,
        'tfidf': tfidf
    }

    return pipeline_components


def ensure_database():
    """确保数据库已初始化"""
    components = init_pipeline()

    # 混合检索器内部有 BestsellerRetriever，通过它访问 Milvus
    milvus_retriever = components['retriever'].milvus_retriever

    if not milvus_retriever.has_collection():
        print("数据库未初始化，开始初始化...")
        milvus_retriever.create_collection()

        # 生成嵌入向量
        products, dense_vectors, sparse_vectors, tfidf = \
            components['embed_gen'].process_all_embeddings()

        # 插入数据库
        milvus_retriever.insert_products(products, dense_vectors, sparse_vectors)

        # 保存 TF-IDF
        components['tfidf'] = tfidf

        stats = milvus_retriever.get_collection_stats()
        print(f"数据库初始化完成! 共 {stats['row_count']} 条记录")

    return components


# ==================== 辅助函数 ====================

def save_to_csv(product_id: str, category: str, style: str, season: str, scene_hint: str):
    """保存新品信息到 CSV"""
    file_exists = NEW_PRODUCT_CSV.exists()

    with open(NEW_PRODUCT_CSV, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['new_id', 'image_path', 'category', 'style', 'season', 'prompt_hint']
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


# ==================== 【新增】LangGraph 工作流版本 ====================

def update_task_status(task_id: str, status: str, progress: float = None, error: str = None, result: dict = None):
    """
    更新任务状态的辅助函数（使用 TaskManager）

    Args:
        task_id: 任务ID
        status: 状态 (pending, processing, completed, failed)
        progress: 进度 (0-1)
        error: 错误信息
        result: 结果数据
    """
    global task_manager, app_logger
    if task_manager is None:
        app_logger.warning(f"[任务状态] TaskManager 未初始化，跳过更新 | task_id={task_id}")
        return

    record = task_manager.get_task(task_id)
    if not record:
        app_logger.warning(f"[任务状态] 任务不存在，跳过更新 | task_id={task_id}")
        return

    # 更新状态
    from task import TaskStatus
    status_map = {
        'pending': TaskStatus.PENDING,
        'processing': TaskStatus.RUNNING,
        'queued': TaskStatus.QUEUED,
        'completed': TaskStatus.COMPLETED,
        'failed': TaskStatus.FAILED
    }

    status_cn_map = {
        'pending': '等待中',
        'processing': '处理中',
        'queued': '排队中',
        'completed': '已完成',
        'failed': '失败'
    }

    if status in status_map:
        new_status = status_map[status]
        old_status = record.status
        if new_status == TaskStatus.RUNNING and old_status != TaskStatus.RUNNING:
            record.mark_running()
            app_logger.info(f"[任务状态] 状态更新 | task_id={task_id}, {old_status.value} → {status_cn_map[status]}")
        elif new_status == TaskStatus.COMPLETED:
            record.mark_completed(result)
            app_logger.info(f"[任务状态] 状态更新 | task_id={task_id}, {old_status.value} → 已完成 | 耗时={record.duration_seconds:.2f}秒")
        elif new_status == TaskStatus.FAILED:
            record.mark_failed(error or 'Unknown error')
            app_logger.error(f"[任务状态] 状态更新 | task_id={task_id}, {old_status.value} → 失败 | 错误={error}")
        elif new_status == TaskStatus.QUEUED:
            record.mark_queued()
            app_logger.info(f"[任务状态] 状态更新 | task_id={task_id}, {old_status.value} → 排队中")

    # 更新进度
    if progress is not None:
        record.update_progress(
            step=record.progress.current_step or "processing",
            percent=int(progress * 100),
            message=record.progress.message or ""
        )
        app_logger.info(f"[任务进度] task_id={task_id}, 进度={int(progress*100)}%")

    # 更新错误信息
    if error and status != 'failed':  # 如果已经mark_failed则不重复设置
        record.error = error

    # 更新结果
    if result and status != 'completed':  # 如果已经mark_completed则不重复设置
        record.result = result


def process_image_task_with_workflow(
    task_id: str,
    file_bytes: bytes,
    category: str,
    style: str,
    season: str,
    scene_hint: str,
    enable_quality_check: bool = False,
    judge_model: str = ""
):
    """
    【新增】使用 LangGraph 工作流处理图片生成任务

    Args:
        task_id: 任务ID
        file_bytes: 上传的图片字节流
        category: 商品品类
        style: 商品风格
        season: 季节
        scene_hint: 场景提示
        enable_quality_check: 是否启用AI质量评估
        judge_model: 质量评估模型
    """
    try:
        app_logger.info("=" * 60)
        app_logger.info("LangGraph 工作流开始 | Task ID: %s", task_id[:8])
        app_logger.info("参数: category=%s, style=%s, season=%s", category, style, season)
        app_logger.info("=" * 60)

        print("\n" + "=" * 60)
        print("  LangGraph Multi-Agent Workflow")
        print("=" * 60)
        print(f"  Task ID: {task_id[:8]}...")
        print("=" * 60)

        update_task_status(task_id, 'processing', 0.05)

        # ==================== 1. 初始化组件 ====================
        app_logger.info("[工作流] [1/7] 初始化工作流组件...")
        print("\n[1/7] 初始化工作流组件...")
        components = ensure_database()

        # 创建工作流实例（使用简化版 v2）
        workflow_app = create_workflow_v2(
            embed_gen=components['embed_gen'],
            retriever=components['retriever'],
            image_gen=components['image_gen'],
            tfidf_vectorizer=components['tfidf'],
            judge_model=judge_model or None
        )

        update_task_status(task_id, 'processing', 0.1)

        # ==================== 2. 创建初始状态 ====================
        app_logger.info("[工作流] [2/7] 创建工作流状态...")
        print("[2/7] 创建工作流状态...")
        state = create_initial_state(
            task_id=task_id,
            file_bytes=file_bytes,
            category=category,
            style=style,
            season=season,
            scene_hint=scene_hint,
            enable_quality_check=enable_quality_check
        )

        # 注入组件到状态
        state = prepare_state_with_components(
            state,
            embed_gen=components['embed_gen'],
            retriever=components['retriever'],
            image_gen=components['image_gen'],
            tfidf_vectorizer=components['tfidf'],
            judge_model=judge_model or None
        )

        update_task_status(task_id, 'processing', 0.15)

        # ==================== 3. 定义进度映射 ====================
        step_progress_map = {
            "upload": 0.2,
            "embedding": 0.3,
            "hybrid_retrieval": 0.5,
            "style_analysis": 0.6,
            "image_gen": 0.75,
            "quality_judge": 0.85,
            "result": 0.95,
        }

        # ==================== 4. 执行工作流（使用stream模式获取实时进度）====================
        app_logger.info("[工作流] [3/7] 执行工作流...")
        print("[3/7] 执行工作流...")
        print("      流程: Upload → Embedding → Retrieval → Style → ImageGen → Quality → Result")

        final_state = None
        for chunk in workflow_app.stream(state):
            # chunk 是节点名称到状态更新的映射
            for node_name, node_state in chunk.items():
                if node_name in step_progress_map:
                    update_task_status(task_id, 'processing', step_progress_map[node_name])
                    progress_pct = step_progress_map[node_name]*100
                    app_logger.info("[工作流] [%s] 完成 - 进度: %.0f%%", node_name, progress_pct)
                    print(f"      [{node_name}] 完成 - 进度: {progress_pct:.0f}%")
                # 检查是否有错误
                if isinstance(node_state, dict) and node_state.get("status") == "failed":
                    update_task_status(task_id, 'failed', error=node_state.get('error_msg', '未知错误'))
                    raise Exception(node_state.get('error_msg', '工作流执行失败'))
                # 保存最终状态
                if isinstance(node_state, dict):
                    final_state = node_state

        # 确保 final_state 存在
        if final_state is None:
            final_state = state

        update_task_status(task_id, 'processing', 0.95)

        # ==================== 5. 提取最终结果 ====================
        app_logger.info("[工作流] [4/7] 提取最终结果...")
        print("\n[4/7] 提取最终结果...")
        final_result = final_state.get("final_result", {})

        if not final_result:
            raise Exception("工作流执行失败：未生成最终结果")

        # 记录指标
        metrics = final_result.get('metrics', {})
        app_logger.info("[工作流] 指标: 总耗时=%.2f秒, 检索耗时=%.2f秒, 生成耗时=%.2f秒",
                       metrics.get('total_time', 0), metrics.get('retrieval_time', 0), metrics.get('gen_time', 0))

        # ==================== 6. 更新任务结果 ====================
        app_logger.info("[工作流] [5/7] 更新任务状态...")
        print("[5/7] 更新任务状态...")
        update_task_status(task_id, 'completed', 1.0, result=final_result)

        # ==================== 7. 打印证据链（调试用）====================
        print("\n[6/7] 证据链追踪:")
        evidence_chain = final_result.get('evidence_chain', [])
        for i, evidence in enumerate(evidence_chain[-5:], 1):  # 只显示最后5条
            print(f"  {i}. {evidence}")

        # 打印指标埋点
        print("\n[7/7] 指标埋点:")

        # 指标名称中文映射
        metric_name_map = {
            "best_score": "最佳评分",
            "cache_hit": "缓存命中",
            "dense_dim": "稠密向量维度",
            "embedding_time": "向量编码耗时(秒)",
            "file_size_mb": "文件大小(MB)",
            "generated_count": "生成图片数量",
            "image_gen_time": "图片生成耗时(秒)",
            "image_height": "图片高度",
            "image_width": "图片宽度",
            "individual_count": "独立分析数量",
            "is_fallback": "是否使用默认值",
            "quality_judge_time": "质量评估耗时(秒)",
            "result_count": "检索结果数量",
            "result_time": "结果处理耗时(秒)",
            "retrieval_time": "检索耗时(秒)",
            "retry_count": "重试次数",
            "should_regenerate": "是否需要重新生成",
            "sparse_nonzero": "稀疏向量非零元素",
            "style_analysis_time": "风格分析耗时(秒)",
            "total_time": "总耗时(秒)",
            "upload_time": "上传耗时(秒)",
            "fallback_triggered": "是否触发兜底",
            "error_step": "错误发生步骤"
        }

        for key, value in sorted(metrics.items()):
            name = metric_name_map.get(key, key)
            print(f"  {name}: {value}")

        print("\n" + "=" * 60)
        print("  Task Completed!")
        print("=" * 60 + "\n")

        app_logger.info("=" * 60)
        app_logger.info("LangGraph 工作流完成 | Task ID: %s", task_id[:8])
        app_logger.info("=" * 60)

    except Exception as e:
        error_msg = str(e)
        update_task_status(task_id, 'failed', error=error_msg)
        app_logger.error("=" * 60)
        app_logger.error("工作流任务失败 | Task ID: %s, 错误: %s", task_id, error_msg)
        app_logger.error("=" * 60)

        print(f"\n{'='*60}")
        print(f"  工作流任务 {task_id} 失败!")
        print(f"{'='*60}")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {error_msg}")
        print(f"{'='*60}")

        # 打印详细堆栈
        import traceback
        print("\n[详细堆栈信息]:")
        traceback.print_exc()
        print(f"{'='*60}\n")


# ==================== 原有线性流程版本（保留兼容）====================

def process_image_task(
    task_id: str,
    product_id: str,
    category: str,
    style: str,
    season: str,
    scene_hint: str,
    enable_quality_check: bool = False,
    judge_model: str = ""
):
    """后台处理图片生成任务"""
    try:
        print("\n" + "=" * 60)
        print("  E-Commerce AI Image Generation Pipeline")
        print("=" * 60)
        print(f"  Task ID: {task_id[:8]}...")
        print(f"  Product ID: {product_id}")
        print("=" * 60)

        update_task_status(task_id, 'processing', 0.1)

        # 初始化流水线
        print("\n[1/6] 初始化流水线组件...")
        components = ensure_database()
        update_task_status(task_id, 'processing', 0.15)

        # 加载新品图片
        print("[2/6] 加载新品图片...")
        img_path = NEW_PRODUCT_DIR / f"{product_id}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {img_path}")

        from utils.core import load_image
        new_img = load_image(str(img_path))
        print(f"      [OK] Image loaded: {new_img.size[0]}x{new_img.size[1]}")
        update_task_status(task_id, 'processing', 0.25)

        # 构建新品数据
        new_product = {
            'new_id': product_id,
            'image_path': f"{product_id}.jpg",
            'category': category,
            'style': style,
            'season': season,
            'prompt_hint': scene_hint
        }
        print(f"      品类: {category} | 风格: {style} | 季节: {season}")

        # 编码新品
        print("\n[3/6] 编码新品向量 (Dense + Sparse)...")
        query_dense, query_sparse, _ = components['embed_gen'].encode_new_product(
            new_product, components['tfidf']
        )
        print(f"      [OK] Dense vector: {len(query_dense)} dim")
        print(f"      [OK] Sparse vector: {len(query_sparse)} non-zero entries")
        update_task_status(task_id, 'processing', 0.35)

        # 检索相似爆款（循环检索状态机）
        print("\n[4/6] 检索相似爆款...")
        print("      检索模式: Milvus向量检索 + 循环检索状态机")

        retrieved = components['retriever'].retrieve_similar_bestsellers(
            query_dense=query_dense.tolist(),
            query_sparse=query_sparse,
            category=category,
            top_k=3,           # 期望返回数量
            enable_cycle=True,  # 启用循环检索状态机
            query_category=category,    # 用于质量评估
            query_style=style,          # 用于质量评估
            query_season=season,        # 用于质量评估
            query_scene_hint=scene_hint # 用于质量评估
        )

        update_task_status(task_id, 'processing', 0.55)

        if not retrieved:
            raise Exception("未找到相似爆款")

        # 提取参考图片
        ref_images = [r["image"] for r in retrieved if r["image"]]
        print(f"\n      [OK] Retrieval complete: {len(retrieved)} reference products")
        update_task_status(task_id, 'processing', 0.6)

        # 根据是否启用质量评估选择不同的处理方式
        judge_model_param = judge_model if judge_model else None

        print("\n[5/6] 分析风格并生成宣传图...")
        if enable_quality_check:
            print(f"      模式: 质量评估模式 (裁判: {judge_model or '默认'})")
            result = components['image_gen'].process_with_quality_check(
                new_product_image=new_img,
                reference_images=ref_images,
                scene_hint=scene_hint,
                judge_model=judge_model_param
            )

            if result.get('error'):
                raise Exception(result['error'])

            style_prompt = result['style_prompt']
            # 保存所有生成的图片，而不仅��是最佳图片
            generated = result.get('all_images', [])
            quality_scores = result.get('best_score', {})
            all_scores = result.get('all_scores', [])
            individual_analyses = result.get('individual_analyses', [])
        else:
            # 标准模式：生成单张图片
            style_prompt, generated, judge_result = components['image_gen'].process_single_product_with_judge(
                new_product_image=new_img,
                reference_images=ref_images,
                scene_hint=scene_hint,
                enable_judge=False  # 标准模式不启用裁判
            )
            quality_scores = judge_result if judge_result else {}
            all_scores = []
            individual_analyses = []

        update_task_status(task_id, 'processing', 0.8)

        if not generated:
            raise Exception("图片生成失败")

        # 保存结果到 output 目录
        print("\n[6/6] 保存结果文件...")
        output_dir = OUTPUT_DIR / product_id
        output_dir.mkdir(exist_ok=True)

        saved_paths = []
        for i, gen_img in enumerate(generated):
            filename = f"{product_id}_generated_{i+1}.png"
            filepath = output_dir / filename
            save_image(gen_img, str(filepath))
            # 返回相对路径
            saved_paths.append(f"/api/output/{product_id}/{filename}")

        # 保存原图和参考图
        save_image(new_img, str(output_dir / f"{product_id}_original.png"))
        for i, ref in enumerate(retrieved):
            if ref.get("image"):
                save_image(
                    ref["image"],
                    str(output_dir / f"{product_id}_reference_{i+1}.png")
                )

        # 保存风格 prompt
        with open(output_dir / f"{product_id}_style_prompt.txt", "w", encoding="utf-8") as f:
            f.write(style_prompt)

        # 保存质量评分
        if quality_scores:
            with open(output_dir / f"{product_id}_quality_scores.json", "w", encoding="utf-8") as f:
                json.dump(quality_scores, f, ensure_ascii=False, indent=2)

        # 准备结果数据
        result_data = {
            'product_id': product_id,
            'category': category,
            'style': style,
            'retrieved_count': len(retrieved),
            'retrieved_products': [
                {
                    'product_id': r.get('product_id'),
                    'color': r.get('color'),
                    'style': r.get('style'),
                    'image': f"/api/output/{product_id}/{product_id}_reference_{i+1}.png"
                }
                for i, r in enumerate(retrieved) if r.get("image")
            ],
            'individual_analyses': individual_analyses if enable_quality_check else [],
            'generated_count': len(generated),
            'style_prompt': style_prompt,
            'generated_images': saved_paths,
            'output_dir': str(output_dir),
            'quality_scores': quality_scores,
            'all_scores': all_scores,
            'quality_enabled': enable_quality_check
        }

        # 完成日志
        print(f"\n      [OK] Generated images: {len(generated)}")
        print(f"      [OK] Saved reference images: {len(retrieved)}")
        print(f"      [OK] Output directory: {output_dir}")

        print("\n" + "=" * 60)
        print("  Task Completed!")
        print("=" * 60 + "\n")

        update_task_status(task_id, 'completed', 1.0, result=result_data)

    except Exception as e:
        update_task_status(task_id, 'failed', error=str(e))
        print(f"任务 {task_id} 失败: {e}")


# ==================== API 端点 ====================

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查"""
    # 检查 Milvus 连接
    milvus_connected = False
    database_ready = False

    try:
        components = init_pipeline()
        milvus_connected = True
        if components['retriever'].has_collection():
            stats = components['retriever'].get_collection_stats()
            database_ready = stats.get('row_count', 0) > 0
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        milvus_connected=milvus_connected,
        database_ready=database_ready
    )


@app.post("/api/upload", response_model=GenerationResponse, tags=["生图"])
async def upload_and_generate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
    style: str = Form(...),
    season: str = Form(default="all_season"),
    scene_hint: str = Form(default=""),
    enable_quality_check: bool = Form(default=False),
    judge_model: str = Form(default=""),
    use_workflow: bool = Form(default=True)  # 【新增】是否使用LangGraph工作流
):
    """
    上传图片并生成宣传图

    - **file**: 新品平铺图片
    - **category**: 商品品类（如：midi_dress, maxi_dress）
    - **style**: 商品风格（如：knitted, drawstring）
    - **season**: 季节（如：summer, winter, all_season）
    - **scene_hint**: 场景提示（可选）
    - **enable_quality_check**: 是否启用质量评估（默认False）
    - **judge_model**: 裁判模型（空则使用默���模型）
    """
    # 验证文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    # 生成唯一 ID
    product_id = f"NEW_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    try:
        # 读取上传的图片字节流
        contents = await file.read()

        # 使用 TaskManager 注册任务
        task_id = task_manager.register_task(
            file_bytes=contents,
            file_name=file.filename or f"{product_id}.jpg",
            category=category,
            style=style,
            season=season,
            scene_hint=scene_hint,
            enable_quality_check=enable_quality_check,
            judge_model=judge_model
        )
        # 保存 product_id 到任务记录的 metadata 中
        task_manager.get_task(task_id).metadata['product_id'] = product_id

        # ==================== 【新增】选择工作流模式 ====================
        if use_workflow and WORKFLOW_AVAILABLE:
            # 使用 LangGraph 工作流
            background_tasks.add_task(
                process_image_task_with_workflow,
                task_id,
                contents,  # file_bytes
                category,
                style,
                season,
                scene_hint,
                enable_quality_check,
                judge_model
            )
            workflow_msg = "（LangGraph工作流）"
        else:
            # 使用原有线性流程（保持兼容）
            # 需要先保存图片文件
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img_path = NEW_PRODUCT_DIR / f"{product_id}.jpg"
            img.save(img_path, quality=90)

            # 保存到 CSV
            save_to_csv(product_id, category, style, season, scene_hint)

            background_tasks.add_task(
                process_image_task,
                task_id,
                product_id,
                category,
                style,
                season,
                scene_hint,
                enable_quality_check,
                judge_model
            )
            workflow_msg = "（传统流程）" if not use_workflow else "（工作流不可用）"

        quality_msg = "（启用质量评估）" if enable_quality_check else ""
        return GenerationResponse(
            task_id=task_id,
            status="pending",
            product_id=product_id,
            message=f"图片上传成功，正在处理中{quality_msg}{workflow_msg if use_workflow else ''}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse, tags=["任务"])
async def get_task_status(task_id: str):
    """获取任务状态"""
    task_dict = task_manager.get_task_status(task_id)
    if not task_dict:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 将 TaskManager 的状态映射到 API 响应格式
    progress_info = task_dict.get('progress', {})
    progress_percent = progress_info.get('percent', 0) / 100.0 if progress_info.get('percent') else 0.0

    return TaskStatusResponse(
        task_id=task_id,
        status=task_dict.get('status', 'pending'),
        progress=progress_percent,
        result=task_dict.get('result'),
        error=task_dict.get('error')
    )


@app.get("/api/output/{product_id}/{filename}", tags=["输出"])
async def get_output_image(product_id: str, filename: str):
    """获取生成的图片"""
    file_path = OUTPUT_DIR / product_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")

    # 读取图片并返回
    with open(file_path, "rb") as f:
        contents = f.read()

    from fastapi.responses import Response
    return Response(content=contents, media_type="image/png")


@app.get("/api/categories", tags=["辅助"])
async def get_categories():
    """获取所有商品品类"""
    try:
        components = init_pipeline()
        products, _ = components['embed_gen'].load_products()

        categories = set(p.get('category', '') for p in products)

        # 中文标签映射
        category_labels = {
            'midi_dress': '中长裙',
            'maxi_dress': '长裙',
            'mini_dress': '短裙',
            'skirt': '半身裙',
            'top': '上装',
            'pants': '裤装'
        }

        return {"categories": [
            {"value": cat, "label": category_labels.get(cat, cat)}
            for cat in sorted(list(categories))
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/styles", tags=["辅助"])
async def get_styles():
    """获取所有商品风格"""
    try:
        components = init_pipeline()
        products, _ = components['embed_gen'].load_products()

        styles = set(p.get('style', '') for p in products)

        # 中文标签映射
        style_labels = {
            'casual': '休闲',
            'formal': '正式',
            'sporty': '运动',
            'elegant': '优雅',
            'vintage': '复古',
            'modern': '现代'
        }

        return {"styles": [
            {"value": style, "label": style_labels.get(style, style)}
            for style in sorted(list(styles))
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 主函数 ====================

def main():
    """启动服务"""
    print("""
============================================================
  E-Commerce AI Image Generation Pipeline - FastAPI Service
============================================================
    """)

    # 预初始化流水线
    try:
        init_pipeline()
    except Exception as e:
        print(f"警告: 流水线初始化失败: {e}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
