"""
工作流集成测试脚本

测试 LangGraph 1.0+ 多Agent工作流是否正确集成到主流程中
"""
import sys
import io
import time
from pathlib import Path

# 修复 Windows 控制台编码问题 - 需要在导���其他模块前完成
import io
if sys.platform == "win32":
    # 确保stdout和stderr没有被关闭
    if hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            # Python < 3.7
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加backend目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from config import PROJECT_ROOT, NEW_PRODUCT_DIR, NEW_PRODUCT_CSV
from utils import save_image


def test_workflow_integration():
    """测试工作流集成"""

    print("\n" + "="*60)
    print("多Agent工作流集成测试")
    print("="*60)

    # ==================== 1. 检查依赖 ====================
    print("\n[1/6] 检查依赖...")

    try:
        import langgraph
        # 使用 importlib.metadata 获取版本（LangGraph 1.0+ 不直接暴露 __version__）
        try:
            import importlib.metadata
            version = importlib.metadata.version("langgraph")
            print(f"  ✓ langgraph 版本: {version}")
        except Exception:
            print(f"  ✓ langgraph 已安装")
    except ImportError as e:
        print(f"  ✗ langgraph 未安装: {e}")
        print("  请运行: pip install langgraph>=1.0.0")
        return False

    try:
        from workflow import create_workflow, prepare_state_with_components
        from agents import create_initial_state
        print(f"  ✓ workflow 模块导入成功")
    except ImportError as e:
        print(f"  ✗ workflow 模块导入失败: {e}")
        return False

    # ==================== 2. 检查数据文件 ====================
    print("\n[2/6] 检查数据文件...")

    # 检查新品图片
    test_images = ["NEW001", "NEW1773539814_bf6e04db", "NEW1773819800_6c4bc9a0"]
    test_image = None
    test_image_name = None

    for img_name in test_images:
        img_path = NEW_PRODUCT_DIR / img_name
        if img_path.exists():
            try:
                test_image = Image.open(img_path).convert("RGB")
                test_image_name = img_name
                print(f"  ✓ 找到测试图片: {img_name}")
                break
            except Exception as e:
                print(f"  ! 图片 {img_name} 无法打开: {e}")

    if not test_image:
        print("  ✗ 未找到可用的测试图片")
        return False

    # 检查CSV
    if not NEW_PRODUCT_CSV.exists():
        print(f"  ! CSV文件不存在: {NEW_PRODUCT_CSV}")

    # ==================== 3. 初始化组件 ====================
    print("\n[3/6] 初始化组件...")

    try:
        from embedding import EmbeddingGenerator
        from retrieval_wrapper import create_retrieval_wrapper
        from image_gen import ImageGenerator

        embed_gen = EmbeddingGenerator()

        # 检查数据库是否已初始化
        retriever = create_retrieval_wrapper()
        milvus_retriever = retriever.milvus_retriever

        if not milvus_retriever.has_collection():
            print("  ! Milvus Collection 不存在，需要先初始化数据库")
            print("  提示: 运行 python main.py --reinit 初始化数据库")
            return False

        stats = milvus_retriever.get_collection_stats()
        print(f"  ✓ Milvus 已就绪: {stats.get('row_count', 0)} 条记录")

        # 加载TF-IDF
        products, _ = embed_gen.load_products()
        tfidf = embed_gen.build_tfidf_vectorizer(products)
        print(f"  ✓ TF-IDF 已构建: {len(tfidf.vocabulary_)} 个词汇")

        image_gen = ImageGenerator()
        print(f"  ✓ 图像生成器已初始化")

    except Exception as e:
        print(f"  ✗ 组件初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ==================== 4. 创建工作流 ====================
    print("\n[4/6] 创建工作流...")

    try:
        app = create_workflow(
            embed_gen=embed_gen,
            retriever=retriever,
            image_gen=image_gen,
            tfidf_vectorizer=tfidf
        )
        print("  ✓ 工作流编译成功")
    except Exception as e:
        print(f"  ✗ 工作流创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ==================== 5. 准备测试数据 ====================
    print("\n[5/6] 准备测试数据...")

    # 将图片转换为字节流
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    file_bytes = img_bytes.read()

    print(f"  图片大小: {len(file_bytes)} bytes")
    print(f"  图片尺寸: {test_image.size}")

    # 创建初始状态
    task_id = f"test_{int(time.time())}"
    state = create_initial_state(
        task_id=task_id,
        file_bytes=file_bytes,
        category="midi_dress",
        style="elegant",
        season="autumn",
        scene_hint="cozy cafe setting with warm ambient lighting"
    )

    # 注入组件
    state = prepare_state_with_components(
        state,
        embed_gen=embed_gen,
        retriever=retriever,
        image_gen=image_gen,
        tfidf_vectorizer=tfidf
    )

    print(f"  ✓ 初始状态创建完成: task_id={task_id}")

    # ==================== 6. 执行工作流测试 ====================
    print("\n[6/6] 执行工作流测试...")
    print("  流程: Upload → Embedding → Retrieval → Style → ImageGen → Quality → Result")

    try:
        start_time = time.time()

        # 执行工作流
        final_state = app.invoke(state)

        elapsed = time.time() - start_time

        # ==================== 7. 分析结果 ====================
        print(f"\n{'='*60}")
        print(f"工作流执行完成")
        print(f"{'='*60}")

        status = final_state.get("status", "unknown")
        print(f"  状态: {status}")
        print(f"  耗时: {elapsed:.2f}秒")

        # 打印证据链
        evidence_chain = final_state.get("evidence_chain", [])
        print(f"\n  证据链追踪 ({len(evidence_chain)} 条):")
        for i, evidence in enumerate(evidence_chain, 1):
            # 只显示前10条和后5条
            if i <= 10 or i > len(evidence_chain) - 5:
                print(f"    {i}. {evidence}")
            elif i == 11:
                print(f"    ... (省略 {len(evidence_chain) - 15} 条)")

        # 打印指标
        metrics = final_state.get("metrics", {})
        if metrics:
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
            print(f"\n  指标埋点:")
            for key, value in sorted(metrics.items()):
                name = metric_name_map.get(key, key)
                print(f"    {name}: {value}")

        # 检查最终结果
        final_result = final_state.get("final_result")
        if final_result:
            print(f"\n  最终结果:")
            print(f"    product_id: {final_result.get('product_id')}")
            print(f"    generated_count: {final_result.get('generated_count')}")

            if final_result.get("generated_images"):
                print(f"    生成图片路径:")
                for path in final_result.get("generated_images", []):
                    print(f"      - {path}")

        # ==================== 8. 判断测试结果 ====================
        if status == "completed":
            print(f"\n  ✅ 测试通过！工作流正常运行")
            return True
        elif status == "failed":
            error_msg = final_state.get("error_msg", "未知错误")
            print(f"\n  ⚠️  测试失败: {error_msg}")
            return False
        else:
            print(f"\n  ⚠️  未知状态: {status}")
            return False

    except Exception as e:
        print(f"\n  ✗ 工作流执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_workflow_integration()

    print(f"\n{'='*60}")
    if success:
        print("✅ 所有测试通过！多Agent工作流已正确集成到主流程中")
    else:
        print("❌ 测试失败，请检查错误信息并修复")
    print(f"{'='*60}")
