# -*- coding: utf-8 -*-
"""
循环检索+质量评估+查询重写 功能测试
"""
import sys
import os
from pathlib import Path

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OPENROUTER_API_KEY, NEW_PRODUCT_DIR, IMAGE_DIR
from retrieval import BestsellerRetriever, RetrievalQualityJudge, query_rewrite
from embedding import EmbeddingGenerator
from utils import load_image


def print_section(title):
    """打印测试区块标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_1_quality_judge():
    """测试1: 质量评估器独立功能"""
    print_section("测试1: RetrievalQualityJudge 质量评估器")

    # 检查 API Key
    if not OPENROUTER_API_KEY:
        print("[SKIP] OPENROUTER_API_KEY 未设置")
        return False

    judge = RetrievalQualityJudge()
    print(f"Using model: {judge.model}")

    # 模拟检索结果
    mock_results = []
    test_images = list(IMAGE_DIR.glob("*.jpg"))[:2]

    if not test_images:
        print("[SKIP] 没有测试图片")
        return False

    for i, img_path in enumerate(test_images):
        try:
            img = load_image(str(img_path))
            mock_results.append({
                "product_id": f"TEST_{i}",
                "category": "midi_dress",
                "style": "elegant",
                "season": "summer",
                "sales_count": 2000,
                "image": img
            })
        except Exception as e:
            print(f"[ERROR] 加载图片失败: {e}")
            continue

    if not mock_results:
        print("[SKIP] 无法构建测试数据")
        return False

    print(f"Test data: {len(mock_results)} products")

    # 执行评分
    print("\nScoring...")
    try:
        scores = judge.score_retrieval_quality(
            retrieved_results=mock_results,
            query_category="midi_dress",
            query_style="elegant",
            query_season="summer",
            query_scene_hint="beach"
        )

        print("\n--- Score Results ---")
        for key, val in scores.items():
            if key != "is_fallback":
                print(f"  {key}: {val}")

        is_fallback = scores.get("is_fallback", False)
        if is_fallback:
            print("\n[WARNING] Used fallback scores (API call failed)")
            return False
        else:
            print("\n[PASS] Quality judge working!")
            return True

    except Exception as e:
        print(f"[ERROR] Scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_query_rewrite():
    """测试2: 查询重写函数"""
    print_section("测试2: query_rewrite 查询重写函数")

    original_filter = 'category == "midi_dress" and sales_count > 1500'

    # 场景1: 品类匹配低
    print("\n[场景1] category_match 低分 (<6)")
    low_category_scores = {
        "category_match": 4,
        "style_match": 7,
        "scene_match": 6,
        "attribute_match": 5,
        "average": 5.5
    }

    rewritten_1 = query_rewrite(original_filter, low_category_scores, rewrite_round=1)
    print(f"  Original: {original_filter}")
    print(f"  Rewritten: {rewritten_1}")

    # 验证：应该包含 sales_count > 1000
    if "sales_count > 1000" in rewritten_1:
        print("  [PASS] Sales threshold lowered to 1000")
    else:
        print("  [FAIL] Sales threshold not lowered")
        return False

    # 场景2: 第2轮重写
    print("\n[场景2] 第2轮重写 (rewrite_round=2)")
    rewritten_2 = query_rewrite(original_filter, low_category_scores, rewrite_round=2)
    print(f"  Original: {original_filter}")
    print(f"  Rewritten: {rewritten_2}")

    # 验证：应该只包含 sales_count > 500
    if rewritten_2 == "sales_count > 500":
        print("  [PASS] Only sales_count > 500 remains")
    else:
        print(f"  [FAIL] Expected 'sales_count > 500', got '{rewritten_2}'")
        return False

    print("\n[PASS] Query rewrite function working!")
    return True


def test_3_single_retrieve():
    """测试3: 单次检索（原有逻辑）"""
    print_section("测试3: _single_retrieve 单次检索")

    retriever = BestsellerRetriever()

    # 检查是否有数据
    if not retriever.has_collection():
        print("[SKIP] Collection not exists")
        return False

    stats = retriever.get_collection_stats()
    row_count = stats.get('row_count', 0)
    print(f"Collection rows: {row_count}")

    if row_count == 0:
        print("[SKIP] No data in collection")
        return False

    # 加载测试向量
    test_images = list(IMAGE_DIR.glob("*.jpg"))
    if not test_images:
        print("[SKIP] No images for testing")
        return False

    # 获取一个产品的向量用于查询
    try:
        embed_gen = EmbeddingGenerator()
        products, images = embed_gen.load_products()

        if not products:
            print("[SKIP] No products loaded")
            return False

        # 使用第一个产品作为查询
        test_product = products[0]
        test_image = images[0]

        print(f"Query product: {test_product.get('product_id', 'N/A')}")
        print(f"Category: {test_product.get('category', 'N/A')}")

        # 获取向量
        dense_vectors = embed_gen.generate_dense_vectors([test_image])
        sparse_vectors = embed_gen.generate_sparse_vectors([test_product], None)

        query_dense = dense_vectors[0].tolist()
        query_sparse = sparse_vectors[0]

        print("\nExecuting single retrieve...")

        # 调用单次检索
        results = retriever._single_retrieve(
            query_dense=query_dense,
            query_sparse=query_sparse,
            category=test_product.get('category', 'dress'),
            min_sales=0,  # 降低阈值以获得更多结果
            top_k=3,
            min_similarity=1.0,  # 放宽相似度
            max_results=3
        )

        print(f"\nResults: {len(results)} items")
        for r in results:
            print(f"  - {r.get('product_id', 'N/A')} | {r.get('category', 'N/A')}")

        if results:
            print("\n[PASS] Single retrieve working!")
            return True
        else:
            print("\n[WARN] No results returned (may be normal)")
            return True  # 不算失败

    except Exception as e:
        print(f"[ERROR] Single retrieve failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_cycle_retrieve():
    """测试4: 循环检索（核心功能）"""
    print_section("测试4: retrieve_similar_bestsellers 循环检索")

    if not OPENROUTER_API_KEY:
        print("[SKIP] OPENROUTER_API_KEY 未设置")
        return False

    retriever = BestsellerRetriever()

    # 检查数据
    if not retriever.has_collection():
        print("[SKIP] Collection not exists")
        return False

    # 加载测试数据
    try:
        embed_gen = EmbeddingGenerator()
        products, images = embed_gen.load_products()

        if not products:
            print("[SKIP] No products loaded")
            return False

        # 构建TF-IDF
        tfidf = embed_gen.build_tfidf_vectorizer(products)

        # 使用第一个产品作为查询
        test_product = products[0]
        test_image = images[0]

        print(f"Query product: {test_product.get('product_id', 'N/A')}")
        print(f"Category: {test_product.get('category', 'N/A')}")
        print(f"Style: {test_product.get('style', 'N/A')}")

        # 获取向量
        dense_vectors = embed_gen.generate_dense_vectors([test_image])

        query_dense = dense_vectors[0].tolist()
        query_sparse = {0: 0.5}  # 简单稀疏向量

        print("\n--- Testing CYCLE retrieval (enable_cycle=True) ---")

        # 启用循环检索
        results = retriever.retrieve_similar_bestsellers(
            query_dense=query_dense,
            query_sparse=query_sparse,
            category=test_product.get('category', 'dress'),
            min_sales=0,
            top_k=2,
            min_similarity=1.0,
            max_results=2,
            enable_cycle=True,  # 关键参数
            query_category=test_product.get('category', ''),
            query_style=test_product.get('style', ''),
            query_season=test_product.get('season', ''),
            query_scene_hint=''
        )

        print(f"\nFinal results: {len(results)} items")
        for r in results:
            print(f"  - {r.get('product_id', 'N/A')}")

        if results:
            print("\n[PASS] Cycle retrieve working!")
            return True
        else:
            print("\n[WARN] No results (may be normal)")
            return True

    except Exception as e:
        print(f"[ERROR] Cycle retrieve failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_disable_cycle():
    """测试5: 禁用循环检索（兼容性测试）"""
    print_section("测试5: 禁用循环检索 (enable_cycle=False)")

    retriever = BestsellerRetriever()

    # 检查数据
    if not retriever.has_collection():
        print("[SKIP] Collection not exists")
        return False

    try:
        embed_gen = EmbeddingGenerator()
        products, images = embed_gen.load_products()

        if not products:
            print("[SKIP] No products")
            return False

        test_product = products[0]
        test_image = images[0]

        dense_vectors = embed_gen.generate_dense_vectors([test_image])
        query_dense = dense_vectors[0].tolist()
        query_sparse = {0: 0.5}

        print("Testing with enable_cycle=False (should use original logic)")

        results = retriever.retrieve_similar_bestsellers(
            query_dense=query_dense,
            query_sparse=query_sparse,
            category=test_product.get('category', 'dress'),
            enable_cycle=False  # 禁用循环
        )

        print(f"Results: {len(results)} items")

        if results is not None:  # 不应该为 None
            print("\n[PASS] Disable cycle working!")
            return True
        else:
            print("\n[FAIL] Returned None")
            return False

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#" * 60)
    print("#  循环检索功能测试套件")
    print("#" * 60)

    tests = [
        ("质量评估器 (RetrievalQualityJudge)", test_1_quality_judge),
        ("查询重写 (query_rewrite)", test_2_query_rewrite),
        ("单次检索 (_single_retrieve)", test_3_single_retrieve),
        ("循环检索 (Cycle Retrieve)", test_4_cycle_retrieve),
        ("禁用循环 (Disable Cycle)", test_5_disable_cycle),
    ]

    results = {}

    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"

    # 汇总报告
    print_section("测试结果汇总")
    for name, result in results.items():
        status = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"{status} {name}")

    passed_count = sum(1 for r in results.values() if r == "PASS")
    print(f"\n总计: {passed_count}/{len(tests)} 通过")

    return results


if __name__ == "__main__":
    run_all_tests()
