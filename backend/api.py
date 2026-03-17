"""
FastAPI 后端服务 - 电商 AI 生��流水线 API
"""
import asyncio
import csv
import io
import json
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

from config import (
    OPENROUTER_API_KEY, MILVUS_URI, COLLECTION_NAME,
    IMAGE_DIR, NEW_PRODUCT_DIR, OUTPUT_DIR, NEW_PRODUCT_CSV,
    DEFAULT_ASPECT_RATIO, DEFAULT_IMAGE_SIZE
)
from embedding import EmbeddingGenerator
from retrieval import BestsellerRetriever
from image_gen import ImageGenerator
from utils import save_image


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

# 任务存储
tasks = {}

# 流水线实例（延迟初始化）
pipeline_components = None


# ==================== 初始化 ====================

def init_pipeline():
    """初始化流水线组件"""
    global pipeline_components

    if pipeline_components is not None:
        return pipeline_components

    print("初始化流水线组件...")
    retriever = BestsellerRetriever()
    embed_gen = EmbeddingGenerator()
    image_gen = ImageGenerator()
    tfidf = None

    # 检查数据库是否已初始化
    if retriever.has_collection():
        stats = retriever.get_collection_stats()
        if stats.get('row_count', 0) > 0:
            print(f"数据库已就绪，包含 {stats['row_count']} 条记录")
            # 加载 TF-IDF
            products, _ = embed_gen.load_products()
            tfidf = embed_gen.build_tfidf_vectorizer(products)

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

    if not components['retriever'].has_collection():
        print("数据库未初始化，开始初始化...")
        components['retriever'].create_collection()

        # 生成嵌入向量
        products, dense_vectors, sparse_vectors, tfidf = \
            components['embed_gen'].process_all_embeddings()

        # 插入数据库
        components['retriever'].insert_products(products, dense_vectors, sparse_vectors)

        # 保存 TF-IDF
        components['tfidf'] = tfidf

        stats = components['retriever'].get_collection_stats()
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
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "电商 AI 生图流水线" + " " * 25 + "║")
        print("║" + "═" * 58 + "║")
        print(f"║  任务ID: {task_id[:8]}...                                      ║")
        print(f"║  商品ID: {product_id}                              ║")
        print("╚" + "═" * 58 + "╝")

        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['progress'] = 0.1

        # 初始化流水线
        print("\n[1/6] 初始化流水线组件...")
        components = ensure_database()
        tasks[task_id]['progress'] = 0.15

        # 加载新品图片
        print("[2/6] 加载新品图片...")
        img_path = NEW_PRODUCT_DIR / f"{product_id}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {img_path}")

        from utils import load_image
        new_img = load_image(str(img_path))
        print(f"      ✓ 图片加载成功: {new_img.size[0]}x{new_img.size[1]}")
        tasks[task_id]['progress'] = 0.25

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
        print(f"      ✓ Dense向量: {len(query_dense)}维")
        print(f"      ✓ Sparse向量: {len(query_sparse)}个非零项")
        tasks[task_id]['progress'] = 0.35

        # 检索相似爆款（循环检索 + 查询重写 + 质量评估）
        print("\n[4/6] 检索相似爆款...")

        retrieved = components['retriever'].retrieve_similar_bestsellers(
            query_dense=query_dense.tolist(),
            query_sparse=query_sparse,
            category=category,
            top_k=3,           # 期望返回数量
            min_similarity=1.0, # 相似度阈值（放宽，允许更多结果）
            max_results=6,      # 最多返回6张
            enable_cycle=True,  # 启用循环检索状态机
            query_category=category,    # 用于质量评估
            query_style=style,          # 用于质量评估
            query_season=season,        # 用于质量评估
            query_scene_hint=scene_hint # 用于质量评估
        )
        tasks[task_id]['progress'] = 0.55

        if not retrieved:
            raise Exception("未找到相似爆款")

        # 提取参考图片
        ref_images = [r["image"] for r in retrieved if r["image"]]
        print(f"\n      ✓ 检索完成: 获得 {len(retrieved)} 个参考商品")
        tasks[task_id]['progress'] = 0.6

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

        tasks[task_id]['progress'] = 0.8

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

        tasks[task_id]['progress'] = 1.0
        tasks[task_id]['status'] = 'completed'

        # 完成日志
        print(f"\n      ✓ 生成图片: {len(generated)} 张")
        print(f"      ✓ 保存参考图: {len(retrieved)} 张")
        print(f"      ✓ 输出目录: {output_dir}")

        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 20 + "任务完成!" + " " * 24 + "║")
        print("╚" + "═" * 58 + "╝\n")
        tasks[task_id]['result'] = {
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

    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
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
    judge_model: str = Form(default="")
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
    task_id = str(uuid.uuid4())

    try:
        # 保存上传的图片
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        img_path = NEW_PRODUCT_DIR / f"{product_id}.jpg"
        img.save(img_path, quality=90)

        # 保存到 CSV
        save_to_csv(product_id, category, style, season, scene_hint)

        # 创建任务
        tasks[task_id] = {
            'task_id': task_id,
            'product_id': product_id,
            'status': 'pending',
            'progress': 0.0,
            'created_at': time.time()
        }

        # 添加后台任务
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

        quality_msg = "（启用质量评估）" if enable_quality_check else ""
        return GenerationResponse(
            task_id=task_id,
            status="pending",
            product_id=product_id,
            message=f"图片上传成功，正在处理中{quality_msg}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse, tags=["任务"])
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task['status'],
        progress=task.get('progress', 0.0),
        result=task.get('result'),
        error=task.get('error')
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
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║        电商 AI 生图流水线 - FastAPI 服务                  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
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
