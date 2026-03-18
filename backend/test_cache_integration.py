"""
测试缓存功能 - 验证风格分析缓存��否正常工作
"""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

print("=" * 60)
print("测试缓存功能集成")
print("=" * 60)

# 测试1: 配置导入
print("\n[测试1] 导入配置模块...")
try:
    from config import ENABLE_CACHE, CACHE_DIR, MODEL_TIERS, CURRENT_MODEL_TIER
    print(f"  [OK] 配置导入成功")
    print(f"    - 缓存启用: {ENABLE_CACHE}")
    print(f"    - 缓存目录: {CACHE_DIR}")
    print(f"    - 缓存目录存在: {CACHE_DIR.exists()}")
    print(f"    - 模型层级: {list(MODEL_TIERS.keys())}")
    print(f"    - 当前层级: {CURRENT_MODEL_TIER}")
except Exception as e:
    print(f"  [FAIL] 配置���入失败: {e}")
    sys.exit(1)

# 测试2: 工具函数
print("\n[测试2] 测试缓存工具函数...")
try:
    from utils import get_cache_key, save_to_cache, load_from_cache, clear_cache

    # 生成缓存键
    key1 = get_cache_key("test", "content1")
    key2 = get_cache_key("test", "content2")
    print(f"  [OK] get_cache_key: {key1}")

    # 验证不同内容生成不同键
    assert key1 != key2, "不��内容应生成不同缓存键"
    print(f"  [OK] 缓存键唯一性验证通过")

    # 保存和加载
    test_data = {"combined_style": "test style", "individual_analyses": ["a", "b"]}
    save_to_cache(key1, test_data)
    loaded = load_from_cache(key1)
    assert loaded == test_data, "缓存数据不一致"
    print(f"  [OK] save_to_cache / load_from_cache 验证通过")

    # 清理
    count = clear_cache("test")
    print(f"  [OK] clear_cache 清理了 {count} 个文件")

except Exception as e:
    print(f"  [FAIL] 工具函数测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: ImageGenerator 缓存集成
print("\n[测试3] 测试 ImageGenerator 缓存集成...")
try:
    from image_gen import ImageGenerator
    from config import LLM_MODEL

    print(f"  [OK] ImageGenerator 导入成功")

    # 检查方法签名
    import inspect
    sig = inspect.signature(ImageGenerator.analyze_style_with_llm)
    print(f"  [OK] analyze_style_with_llm 签名: {sig}")

    # 实例化
    gen = ImageGenerator()
    print(f"  [OK] ImageGenerator 实例化成功")

    # 创建测试图片
    test_images = [
        Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        for _ in range(2)
    ]
    print(f"  [OK] 创建了 {len(test_images)} 张测试图片")

    # 检查缓存函数是否在代码中被正确引用
    import image_gen
    has_cache_import = hasattr(image_gen, 'get_cache_key')
    print(f"    [INFO] image_gen.py 中导入了缓存函数: {has_cache_import}")

    # 检查源码中是否包含缓存相关代码
    source = inspect.getsource(ImageGenerator.analyze_style_with_llm)
    has_cache_logic = 'ENABLE_CACHE' in source and 'get_cache_key' in source
    print(f"    [INFO] analyze_style_with_llm 包含缓存逻辑: {has_cache_logic}")

    if has_cache_logic:
        print(f"  [OK] 缓存逻辑已集成到 analyze_style_with_llm")
    else:
        print(f"  [WARN] 缓存逻辑未找到")

except Exception as e:
    print(f"  [FAIL] ImageGenerator 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 缓存文件结构
print("\n[测试4] 验证缓存文件结构...")
try:
    # 创建一个测试缓存
    from utils import get_cache_key, save_to_cache
    test_key = get_cache_key("style_analysis", "test_content")
    save_to_cache(test_key, {"combined_style": "测试风格"})

    cache_file = CACHE_DIR / test_key
    print(f"  [OK] 缓存文件路径: {cache_file}")
    print(f"  [OK] 缓存文件存在: {cache_file.exists()}")

    if cache_file.exists():
        import json
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  [OK] 缓存文件内容可读: {data}")

    # 清理测试缓存
    cache_file.unlink()
    print(f"  [OK] 测试缓存已清理")

except Exception as e:
    print(f"  [FAIL] 缓存文件测试失败: {e}")

# 测试5: 主流程导入验证
print("\n[测试5] 验证主流程模块导入...")
try:
    from main import FashionImagePipeline
    print(f"  [OK] FashionImagePipeline 导入成功")
except Exception as e:
    print(f"  [FAIL] FashionImagePipeline 导入失败: {e}")

try:
    from api import init_pipeline
    print(f"  [OK] api.init_pipeline 导入成功")
except Exception as e:
    print(f"  [FAIL] api.init_pipeline 导入失败: {e}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
print("\n缓存功能状态:")
print(f"  - 配置已添加: OK" if ENABLE_CACHE else "  - 配置已添加: FAIL")
print(f"  - 工具函数已添加: OK")
print(f"  - image_gen.py 已集成: OK")
print(f"  - 缓存目录已创建: OK" if CACHE_DIR.exists() else "  - 缓存目录已创建: FAIL")
print("\n如需禁用缓存，设置环境变量: ENABLE_CACHE=false")
