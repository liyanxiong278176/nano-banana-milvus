# -*- coding: utf-8 -*-
"""
循环检索 - 低分场景测试
模拟第1轮评分低，触发查询重写和后续轮次
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import BestsellerRetriever, RetrievalQualityJudge, query_rewrite
from embedding import EmbeddingGenerator
from utils import load_image


def test_low_score_cycle():
    """
    模拟低分场景：用裙子的图片搜裤子的品类
    这样第1轮评分会很低，触发查询重写
    """
    print("=" * 60)
    print("低分场景测试 - 触发循环检索")
    print("=" * 60)

    retriever = BestsellerRetriever()
    embed_gen = EmbeddingGenerator()

    # 加载产品
    products, images = embed_gen.load_products()
    tfidf = embed_gen.build_tfidf_vectorizer(products)

    # 使用裙子图片
    test_image = images[0]
    print(f"\n测试图片: {products[0].get('product_id')}")
    print(f"实际品类: {products[0].get('category')}")

    # 获取向量
    dense_vectors = embed_gen.generate_dense_vectors([test_image])
    query_dense = dense_vectors[0].tolist()
    query_sparse = {0: 0.5}

    print("\n" + "-" * 60)
    print("【故意制造不匹配】")
    print("  图片品类: midi_dress (裙子)")
    print("  查询品类: pants (裤子)")
    print("  预期: 第1轮评分会很低，触发重写")
    print("-" * 60)

    # 故意用错误的品类查询
    results = retriever.retrieve_similar_bestsellers(
        query_dense=query_dense,
        query_sparse=query_sparse,
        category="pants",  # ❌ 故意用错误的品类
        min_sales=0,
        top_k=2,
        min_similarity=1.0,
        max_results=2,
        enable_cycle=True,
        query_category="pants",      # ❌ 查询裤子
        query_style="casual",
        query_season="summer",
        query_scene_hint="casual wear"
    )

    print("\n" + "=" * 60)
    print(f"最终结果: {len(results)} 个")
    for r in results:
        print(f"  - {r.get('product_id')} | {r.get('category')}")
    print("=" * 60)


def test_query_rewrite_directly():
    """直接测试查询重写逻辑"""
    print("\n" + "=" * 60)
    print("查询重写逻辑测试")
    print("=" * 60)

    original = 'category == "midi_dress" and sales_count > 1500'

    # 场景1: 品类匹配低 (category_match < 6)
    print("\n[场景1] category_match = 3 (低分)")
    low_scores = {
        "category_match": 3,
        "style_match": 7,
        "scene_match": 6,
        "attribute_match": 5,
        "average": 5.25
    }

    round1 = query_rewrite(original, low_scores, rewrite_round=1)
    print(f"  原始: {original}")
    print(f"  第1轮重写: {round1}")

    # 场景2: 风格匹配低
    print("\n[场景2] style_match = 4 (低分)")
    low_style_scores = {
        "category_match": 8,
        "style_match": 4,
        "scene_match": 6,
        "attribute_match": 5,
        "average": 5.75
    }

    round1_b = query_rewrite(original, low_style_scores, rewrite_round=1)
    print(f"  原始: {original}")
    print(f"  第1轮重写: {round1_b}")

    # 场景3: 第2轮重写
    print("\n[场景3] 第2轮重写 (rewrite_round=2)")
    round2 = query_rewrite(original, low_scores, rewrite_round=2)
    print(f"  原始: {original}")
    print(f"  第2轮重写: {round2}")

    # 验证结果
    print("\n[验证]")
    if "sales_count > 1000" in round1:
        print("  [OK] 第1轮: 销量阈值降到1000")
    if "sales_count > 500" in round2:
        print("  [OK] 第2轮: 销量阈值降到500")


def test_judge_with_mismatch():
    """测试质量评估器对不匹配结果的评分"""
    print("\n" + "=" * 60)
    print("质量评估器 - 不匹配场景测试")
    print("=" * 60)

    judge = RetrievalQualityJudge()

    # 加载一些裙子图片
    from config import IMAGE_DIR
    test_images = list(IMAGE_DIR.glob("*.jpg"))[:2]

    mock_results = []
    for i, img_path in enumerate(test_images):
        try:
            img = load_image(str(img_path))
            mock_results.append({
                "product_id": f"TEST_{i}",
                "category": "midi_dress",  # 实际是裙子
                "style": "elegant",
                "season": "summer",
                "sales_count": 2000,
                "image": img
            })
        except:
            continue

    # 用裤子的品类去评分
    print("\n[场景] 检索结果是裙子，但查询的是裤子")
    print("  检索结果: midi_dress (裙子)")
    print("  查询品类: pants (裤子)")

    scores = judge.score_retrieval_quality(
        retrieved_results=mock_results,
        query_category="pants",     # ❌ 查询裤子
        query_style="casual",
        query_season="summer",
        query_scene_hint="casual"
    )

    print(f"\n评分结果:")
    print(f"  category_match: {scores.get('category_match', 0)}/10 (应该很低)")
    print(f"  average: {scores.get('average', 0)}/10")


if __name__ == "__main__":
    # 测试1: 查询重写逻辑
    test_query_rewrite_directly()

    # 测试2: 质量评估器对不匹配的评分
    test_judge_with_mismatch()

    # 测试3: 完整循环检索（低分场景）
    test_low_score_cycle()
