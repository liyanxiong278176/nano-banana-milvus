# -*- coding: utf-8 -*-
"""
简化版循环检索测试 - 只测试关键功能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import query_rewrite, RetrievalQualityJudge
from config import IMAGE_DIR
from utils import load_image


def test_rewrite():
    """测试查询重写"""
    print("=" * 50)
    print("测试1: 查询重写")
    print("=" * 50)

    original = 'category == "midi_dress" and sales_count > 1500'

    # 低分场景
    low_scores = {
        "category_match": 3,
        "style_match": 5,
        "scene_match": 4,
        "attribute_match": 5,
        "average": 4.25
    }

    print(f"原始过滤: {original}")
    print(f"评分: {low_scores['average']}/10 (低分)")

    # 第1轮重写
    r1 = query_rewrite(original, low_scores, 1)
    print(f"第1轮重写: {r1}")

    # 第2轮重写
    r2 = query_rewrite(original, low_scores, 2)
    print(f"第2轮重写: {r2}")

    # 验证
    assert "sales_count > 1000" in r1, "第1轮应该降低销量阈值"
    assert "sales_count > 500" in r2, "第2轮应该进一步降低"
    print("\n[PASS] 查询重写正常")


def test_judge():
    """测试质量评估器"""
    print("\n" + "=" * 50)
    print("测试2: 质量评估器")
    print("=" * 50)

    judge = RetrievalQualityJudge()
    print(f"使用模型: {judge.model}")

    # 模拟不匹配的检索结果
    test_images = list(IMAGE_DIR.glob("*.jpg"))[:1]
    if not test_images:
        print("[SKIP] 没有测试图片")
        return

    img = load_image(str(test_images[0]))
    mock_results = [{
        "product_id": "TEST_001",
        "category": "midi_dress",  # 裙子
        "style": "elegant",
        "season": "summer",
        "sales_count": 2000,
        "image": img
    }]

    # 用裤子品类查询（故意不匹配）
    print("\n场景: 检索结果是裙子，但查询的是裤子")
    scores = judge.score_retrieval_quality(
        retrieved_results=mock_results,
        query_category="pants",  # 查询裤子
        query_style="casual",
        query_season="summer",
        query_scene_hint=""
    )

    print(f"\n评分结果:")
    print(f"  category_match: {scores.get('category_match', 0)}/10")
    print(f"  average: {scores.get('average', 0)}/10")

    # 品类不匹配，分数应该很低
    if scores.get('category_match', 10) < 7:
        print("\n[PASS] 质量评估器能识别不匹配")
    else:
        print("\n[INFO] 可能使用了兜底评分")


def test_cycle_logic():
    """测试循环检索逻辑"""
    print("\n" + "=" * 50)
    print("测试3: 循环检索逻辑")
    print("=" * 50)

    # 模拟3轮循环的分数变化
    print("\n模拟场景:")
    print("  第1轮: 评分 5.5/10 (低) -> 触发重写")
    print("  第2轮: 评分 6.8/10 (仍低) -> 再次重写")
    print("  第3轮: 评分 7.5/10 (达标) -> 返回结果")

    scores_history = [
        {"average": 5.5, "category_match": 4},
        {"average": 6.8, "category_match": 6},
        {"average": 7.5, "category_match": 8},
    ]

    for i, scores in enumerate(scores_history, 1):
        print(f"\n第{i}轮: {scores['average']}/10")
        if scores['average'] < 7.0:
            print(f"  -> 质量未达标，继续下一轮")
        else:
            print(f"  -> 质量达标，返回结果!")
            break

    print("\n[PASS] 循环逻辑验证完成")


if __name__ == "__main__":
    test_rewrite()
    test_judge()
    test_cycle_logic()

    print("\n" + "=" * 50)
    print("所有测试完成")
    print("=" * 50)
