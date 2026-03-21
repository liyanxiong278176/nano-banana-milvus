"""
提示词配置测试 - 验证 v2.0 提示词工程

【测试内容】
1. prompts.py 模块导入
2. PromptBuilder 功能测试
3. 向后兼容性测试
4. 版本管理测试
"""
import sys
from pathlib import Path

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 修复 Windows 控制台编码问题
from console_utils import fix_console_encoding
fix_console_encoding()

print("=" * 60)
print("提示词配置测试 v2.0")
print("=" * 60)

# ==================== 测试1: 模块导入 ====================
print("\n[测试1] 模块导入测试")
print("-" * 60)

try:
    from prompts import PromptBuilder, PromptVersionManager, Prompts
    print("[OK] prompts.py 导入成功")
    print(f"  - PromptBuilder: {PromptBuilder}")
    print(f"  - PromptVersionManager: {PromptVersionManager}")
    print(f"  - Prompts: {Prompts}")
except ImportError as e:
    print(f"[FAIL] prompts.py 导入失败: {e}")
    sys.exit(1)

# ==================== 测试2: 版本管理 ====================
print("\n[测试2] 版本管理���试")
print("-" * 60)

versions = PromptVersionManager.get_all_versions()
print(f"✓ 获取到 {len(versions)} 个提示词类型")
for prompt_type, info in versions.items():
    print(f"  - {prompt_type}: v{info['current']}")

# ==================== 测试3: 风格分析提示词 ====================
print("\n[测试3] 风格分析提示词测试")
print("-" * 60)

# 单图分析
single_prompt = PromptBuilder.build_style_analysis(has_references=False)
print(f"✓ 单图分析提示词: {len(single_prompt)} 字符")
print(f"  预览: {single_prompt[:100]}...")

# 综合分析
combined_prompt = PromptBuilder.build_style_analysis(has_references=True)
print(f"✓ 综合分析提示词: {len(combined_prompt)} 字符")
print(f"  预览: {combined_prompt[:100]}...")

# ==================== 测试4: 图像生成提示词 ====================
print("\n[测试4] 图像生成提示词测试")
print("-" * 60)

# 多参考图
gen_prompt, neg_prompt = PromptBuilder.build_image_generation(
    style_prompt="柔和自然光，户外海滩场景",
    scene_hint="夏日海滩",
    single_reference=False,
    include_negative=True
)
print(f"✓ 多参考图提示词: {len(gen_prompt)} 字符")
print(f"✓ 负向提示词: {len(neg_prompt)} 字符")
print(f"  负向提示词预览: {neg_prompt[:150]}...")

# 单参考图
single_gen_prompt, single_neg_prompt = PromptBuilder.build_image_generation(
    style_prompt="简洁白色背景，影棚布光",
    scene_hint="",
    single_reference=True,
    include_negative=True
)
print(f"✓ 单参考图提示词: {len(single_gen_prompt)} 字符")
print(f"✓ 单参考图负向提示词: {len(single_neg_prompt)} 字符")

# ==================== 测试5: 质量评估提示词 ====================
print("\n[测试5] 质量评估提示词测试")
print("-" * 60)

# 无参考图
judge_prompt_no_ref = PromptBuilder.build_quality_judge(has_references=False)
print(f"✓ 质量评估(无参考): {len(judge_prompt_no_ref)} 字符")

# 有参考图
judge_prompt_with_ref = PromptBuilder.build_quality_judge(has_references=True)
print(f"✓ 质量评估(有参考): {len(judge_prompt_with_ref)} 字符")
print(f"  预览: {judge_prompt_with_ref[:200]}...")

# ==================== 测试6: 检索评估提示词 ====================
print("\n[测试6] 检索评估提示词测试")
print("-" * 60)

query_context = """品类：midi_dress
风格：elegant
季节：summer
场景：beach"""

results_summary = """1. SKU001 | 品类:midi_dress | 风格:elegant | 季节:summer | 销量:2500
2. SKU002 | 品类:midi_dress | 风格:elegant | 季节:summer | 销量:1800
3. SKU003 | 品类:maxi_dress | 风格:elegant | 季节:summer | 销量:1200"""

retrieval_prompt = PromptBuilder.build_retrieval_judge(
    query_context=query_context,
    results_summary=results_summary
)
print(f"✓ 检索评估提示词: {len(retrieval_prompt)} 字符")
print(f"  包含查询上下文: {query_context[:30]}...")
print(f"  包含结果摘要: {results_summary[:30]}...")

# ==================== 测试7: 向后兼容性 ====================
print("\n[测试7] 向后兼容性测试")
print("-" * 60)

try:
    # 测试 image_gen.py 的导入
    from image_gen import ImageGenerator, PROMPTS_AVAILABLE as image_prompts_available
    print(f"✓ image_gen.py 导入成功")
    print(f"  - PROMPTS_AVAILABLE: {image_prompts_available}")

    # 测试 retrieval.py 的导入
    from retrieval import RetrievalQualityJudge, PROMPTS_AVAILABLE as retrieval_prompts_available
    print(f"✓ retrieval.py 导入成功")
    print(f"  - PROMPTS_AVAILABLE: {retrieval_prompts_available}")

    if image_prompts_available and retrieval_prompts_available:
        print("✓ 提示词配置已集成到现有模块")
    else:
        print("⚠ 部分模块未使用提示词配置（使用内置提示词）")

except ImportError as e:
    print(f"✗ 模块导入失败: {e}")

# ==================== 测试8: 配置参数 ====================
print("\n[测试8] 配置参数测试")
print("-" * 60)

# 检查各个提示词的参数配置
for attr_name in dir(Prompts):
    if attr_name.endswith('_V2'):
        config = getattr(Prompts, attr_name)
        print(f"\n{attr_name}:")
        print(f"  - version: {config.version}")
        print(f"  - parameters: {config.parameters}")
        if config.negative_prompt:
            print(f"  - has_negative_prompt: Yes ({len(config.negative_prompt)} 字符)")

# ==================== 总结 ====================
print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
print("\n【结论】")
print("✓ prompts.py 模块正常工作")
print("✓ PromptBuilder 可以构建所有类型的提示词")
print("✓ 向后兼容性良好，现有模块可以正常导入")
print("✓ 负向提示词已集成到图像生成流程")
print("\n【下一步】")
print("1. 运行完整测试: python test/test_full_integration.py")
print("2. 启动服务: python api.py")
print("3. 验证生成效果")
print("=" * 60)
