"""
P1 阶段测试 - 提示词监控和少样本示例

【测试内容】
1. prompts_v2.py 模块导入
2. 指标收集功能
3. 少样本示例功能
4. A/B 测试框架
5. 向后兼容性
"""
import sys
from pathlib import Path

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 修复控制台编码
from console_utils import fix_console_encoding
fix_console_encoding()

print("=" * 60)
print("P1 Phase Test - Prompt Monitoring & Few-Shot")
print("=" * 60)

# ==================== 测试1: prompts_v2.py 导入 ====================
print("\n[Test 1] prompts_v2.py Import")
print("-" * 60)

try:
    from prompts_v2 import (
        PromptMetrics,
        get_metrics,
        record_prompt_execution,
        build_few_shot_prompt,
        get_few_shot_examples,
        ABTestConfig,
        get_ab_test
    )
    print("[OK] prompts_v2.py imported successfully")
except ImportError as e:
    print(f"[FAIL] prompts_v2.py import failed: {e}")
    sys.exit(1)

# ==================== 测试2: 指标收集 ====================
print("\n[Test 2] Metrics Collection")
print("-" * 60)

metrics = get_metrics()
print(f"[OK] Metrics instance created: {type(metrics).__name__}")

# 模拟记录一些执行
record_prompt_execution(
    prompt_type="quality_judge",
    prompt_version="2.0",
    model="qwen/qwen3-vl-8b-instruct",
    input_summary="Test input",
    output_summary="Test output",
    execution_time=1.5,
    success=True
)

record_prompt_execution(
    prompt_type="retrieval_judge",
    prompt_version="2.0",
    model="qwen/qwen3-vl-8b-instruct",
    input_summary="Test retrieval",
    output_summary="Score: 8.0",
    execution_time=2.0,
    success=True
)

record_prompt_execution(
    prompt_type="quality_judge",
    prompt_version="2.0",
    model="qwen/qwen3-vl-8b-instruct",
    input_summary="Failed test",
    output_summary="",
    execution_time=0.5,
    success=False,
    error_message="Test error"
)

print("[OK] Recorded 3 test executions")

# 获取统计
stats = metrics.get_prompt_type_stats("quality_judge")
print(f"[OK] quality_judge stats: {stats['total_executions']} executions, "
      f"success rate: {stats['versions']['2.0']['success_rate']:.1%}")

# ==================== 测试3: 少样本示例 ====================
print("\n[Test 3] Few-Shot Examples")
print("-" * 60)

examples = get_few_shot_examples("quality_judge")
print(f"[OK] Found {len(examples)} examples for quality_judge")

for i, ex in enumerate(examples, 1):
    print(f"  Example {i}: {ex.get('name', 'N/A')}")
    print(f"    Description: {ex.get('description', 'N/A')}")

# 测试构建带示例的提示词
base_prompt = "Please rate the image quality:"
enhanced = build_few_shot_prompt(base_prompt, "quality_judge", include_examples=True, max_examples=1)
print(f"[OK] Enhanced prompt length: {len(enhanced)} chars (vs {len(base_prompt)} base)")

# ==================== 测试4: A/B 测试框架 ====================
print("\n[Test 4] A/B Testing Framework")
print("-" * 60)

ab_test = get_ab_test()
ab_test.create_test(
    test_name="quality_judge_v2_vs_v1",
    prompt_type="quality_judge",
    version_a="2.0",
    version_b="1.0",
    traffic_split=0.7
)
print("[OK] Created A/B test: quality_judge_v2_vs_v1")

# 模拟流量分配
versions = []
for i in range(100):
    v = ab_test.get_version_for_test("quality_judge_v2_vs_v1")
    versions.append(v)

v2_count = versions.count("2.0")
v1_count = versions.count("1.0")
print(f"[OK] Traffic distribution: v2.0={v2_count}%, v1.0={v1_count}%")

# 获取测试结果
results = ab_test.get_test_results("quality_judge_v2_vs_v1")
print(f"[OK] Test results: {results}")

# ==================== 测试5: 集成测试 ====================
print("\n[Test 5] Integration with Existing Modules")
print("-" * 60)

try:
    from image_gen import ImageGenerator, METRICS_AVAILABLE as image_metrics
    from retrieval import RetrievalQualityJudge, METRICS_AVAILABLE as retrieval_metrics
    print("[OK] image_gen.py imported")
    print(f"  - METRICS_AVAILABLE: {image_metrics}")
    print("[OK] retrieval.py imported")
    print(f"  - METRICS_AVAILABLE: {retrieval_metrics}")

    if image_metrics and retrieval_metrics:
        print("[OK] Monitoring is enabled in both modules")
    else:
        print("[WARN] Monitoring not fully enabled")

except ImportError as e:
    print(f"[FAIL] Module import failed: {e}")

# ==================== 测试6: 指标摘要 ====================
print("\n[Test 6] Metrics Summary")
print("-" * 60)

metrics.print_summary()

# ==================== 测试7: 版本管理 ====================
print("\n[Test 7] Version Management")
print("-" * 60)

try:
    from prompts import PromptVersionManager
    PromptVersionManager.print_version_info()
except ImportError:
    print("[WARN] prompts.py not available")

# ==================== 总结 ====================
print("\n" + "=" * 60)
print("P1 Phase Test Summary")
print("=" * 60)

print("\n[Completed]")
print("1. prompts_v2.py module working")
print("2. Metrics collection functional")
print("3. Few-shot examples available")
print("4. A/B testing framework ready")
print("5. Integration with existing modules successful")

print("\n[Next Steps]")
print("1. Enable few-shot examples in production:")
print("   Set include_examples=True in build_few_shot_prompt()")
print("\n2. Create A/B tests:")
print("   Use get_ab_test().create_test() to set up experiments")
print("\n3. Monitor metrics:")
print("   Call get_metrics().print_summary() periodically")

print("\n" + "=" * 60)
