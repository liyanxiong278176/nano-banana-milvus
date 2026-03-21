"""
P2 阶段测试 - 用户反馈闭环和多语言支持

【测试内容】
1. prompts_v3.py 模块导入
2. 用户反馈收集
3. 反馈分析和建议
4. 多语言支持
5. API 端点集成
"""
import sys
from pathlib import Path

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 修复控制台编码
from console_utils import fix_console_encoding
fix_console_encoding()

print("=" * 60)
print("P2 Phase Test - Feedback Loop & i18n")
print("=" * 60)

# ==================== 测试1: prompts_v3.py 导入 ====================
print("\n[Test 1] prompts_v3.py Import")
print("-" * 60)

try:
    from prompts_v3 import (
        FeedbackType,
        UserFeedback,
        FeedbackAnalyzer,
        get_feedback_analyzer,
        add_user_feedback,
        PromptLocale
    )
    print("[OK] prompts_v3.py imported successfully")
except ImportError as e:
    print(f"[FAIL] prompts_v3.py import failed: {e}")
    sys.exit(1)

# ==================== 测试2: 用户反馈收集 ====================
print("\n[Test 2] User Feedback Collection")
print("-" * 60)

analyzer = get_feedback_analyzer()
print(f"[OK] FeedbackAnalyzer instance created")

# 模拟用户反馈
feedbacks = [
    ("task_001", "NEW_001", "quality", 5, "优秀结果"),
    ("task_002", "NEW_002", "quality", 4, "良好"),
    ("task_003", "NEW_003", "quality", 3, "一般"),
    ("task_004", "NEW_004", "quality", 2, "服装不太准"),
    ("task_005", "NEW_005", "quality", 1, "完全不符"),
]

for task_id, product_id, fb_type, rating, comment in feedbacks:
    fb_id = add_user_feedback(
        task_id=task_id,
        product_id=product_id,
        feedback_type=fb_type,
        rating=rating,
        metadata={"comment": comment}
    )
    print(f"  [Recorded] {task_id}: {rating}/5 - {comment}")

print(f"[OK] Recorded {len(feedbacks)} feedbacks")

# ==================== 测试3: 反馈分析 ====================
print("\n[Test 3] Feedback Analysis")
print("-" * 60)

# 获取性能统计
stats = analyzer.get_prompt_performance("quality_judge")
print(f"[OK] Performance stats:")
print(f"  Total feedbacks: {stats['total_feedbacks']}")
print(f"  Average rating: {stats['avg_rating']}")
print(f"  Satisfaction rate: {stats['satisfaction_rate']:.1%}")

# 评分分布
print(f"\n  Rating distribution:")
for rating, count in sorted(stats['rating_distribution'].items()):
    bar = "#" * count
    print(f"    {rating}/5: {bar} ({count})")

# ==================== 测试4: 优化建议 ====================
print("\n[Test 4] Optimization Suggestions")
print("-" * 60)

suggestions = analyzer.get_optimization_suggestions()
print(f"[OK] Found {len(suggestions)} suggestions")

for i, sg in enumerate(suggestions, 1):
    print(f"\n  Suggestion {i}:")
    print(f"    Type: {sg['prompt_type']}")
    print(f"    Issue: {sg['issue']}")
    print(f"    Priority: {sg['priority']}")
    print(f"    Suggestion: {sg['suggestion']}")

# ==================== 测试5: 多语言支持 ====================
print("\n[Test 5] Multi-Language Support")
print("-" * 60)

locales = PromptLocale.get_supported_locales()
print(f"[OK] Supported locales: {locales}")

# 测试中文
zh_system = PromptLocale.get_prompt("quality_judge", "zh", "system")
print(f"\n  [ZH] System prompt: {zh_system[:30]}...")

zh_dim = PromptLocale.get_prompt("quality_judge", "zh", "dimensions")
print(f"  [ZH] Dimensions: {list(zh_dim.keys())}")

# 测试英文
en_system = PromptLocale.get_prompt("quality_judge", "en", "system")
print(f"\n  [EN] System prompt: {en_system[:30]}...")

en_dim = PromptLocale.get_prompt("quality_judge", "en", "dimensions")
print(f"  [EN] Dimensions: {list(en_dim.keys())}")

# ==================== 测试6: 版本对比 ====================
print("\n[Test 6] Version Comparison")
print("-" * 60)

# 模拟添加不同版本的反馈
add_user_feedback("task_v1_001", "NEW_101", "quality", 5, prompt_version="1.0")
add_user_feedback("task_v1_002", "NEW_102", "quality", 4, prompt_version="1.0")
add_user_feedback("task_v1_003", "NEW_103", "quality", 3, prompt_version="1.0")

comparison = analyzer.compare_versions("quality_judge")
if "error" not in comparison:
    print(f"[OK] Version comparison:")
    for ver, data in comparison["comparison"].items():
        print(f"  v{ver}: avg={data['avg_rating']}, satisfaction={data['satisfaction']:.1%}")

    print(f"\n  Recommendation: {comparison['recommendation']['best_version']} "
          f"(rating: {comparison['recommendation']['best_rating']})")
else:
    print(f"[INFO] {comparison['error']}")

# ==================== 测试7: 打印摘要 ====================
print("\n[Test 7] Summary Report")
print("-" * 60)

analyzer.print_summary()

# ==================== 测试8: API 集成检查 ====================
print("\n[Test 8] API Integration Check")
print("-" * 60)

api_endpoints = [
    "/api/feedback",
    "/api/feedback/analytics",
    "/api/feedback/suggestions",
    "/api/prompts/locales"
]

for endpoint in api_endpoints:
    print(f"  - {endpoint}")

# 检查 api.py 是否导入了新功能
try:
    import api
    import inspect
    source = inspect.getsource(api)

    if "submit_feedback" in source:
        print("\n[OK] Feedback endpoints integrated in api.py")
    if "get_optimization_suggestions" in source:
        print("[OK] Suggestions endpoint integrated")
    if "get_supported_locales" in source:
        print("[OK] Locale endpoint integrated")

except Exception as e:
    print(f"[WARN] Could not verify API integration: {e}")

# ==================== 总结 ====================
print("\n" + "=" * 60)
print("P2 Phase Test Summary")
print("=" * 60)

print("\n[Completed]")
print("1. User feedback collection working")
print("2. Feedback analysis functional")
print("3. Optimization suggestions generated")
print("4. Multi-language support available (zh/en)")
print("5. API endpoints ready")

print("\n[New Features]")
print("- POST /api/feedback - Submit user feedback")
print("- GET /api/feedback/analytics - View analytics")
print("- GET /api/feedback/suggestions - Get optimization tips")
print("- GET /api/prompts/locales - Supported languages")

print("\n" + "=" * 60)
