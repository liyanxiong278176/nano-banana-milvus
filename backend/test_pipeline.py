"""
测试脚本 - 测试流水线各个模块
"""
import os
from pathlib import Path

from config import COLLECTION_NAME, LLM_MODEL


def test_config():
    """测试配置模块"""
    print("\n" + "=" * 60)
    print("测试: 配置模块")
    print("=" * 60)

    try:
        from config import (
            OPENROUTER_API_KEY, EMBED_MODEL, LLM_MODEL,
            IMAGE_GEN_MODEL, MILVUS_URI, COLLECTION_NAME
        )
        print(f"✓ 配置加载成功")
        print(f"  API Key: {'*' * 20 if OPENROUTER_API_KEY else '未设置'}")
        print(f"  Embed 模型: {EMBED_MODEL}")
        print(f"  LLM 模型: {LLM_MODEL}")
        print(f"  图像生成模型: {IMAGE_GEN_MODEL}")
        print(f"  数据库: {MILVUS_URI}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def test_utils():
    """测试工具模块"""
    print("\n" + "=" * 60)
    print("测试: 工具模块")
    print("=" * 60)

    try:
        from utils.core import image_to_uri, sparse_to_dict
        from PIL import Image
        import numpy as np

        # 测试图片转 URI
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        uri = image_to_uri(img, max_size=50)
        assert uri.startswith("data:image/jpeg;base64,"), "URI 格式错误"
        print(f"✓ image_to_uri 测试通过")

        # 测试稀疏向量转换
        from scipy import sparse as sp
        row = sp.csr_matrix(np.array([[0, 0.5, 0, 0.3, 0]]))
        d = sparse_to_dict(row)
        assert isinstance(d, dict), "返回类型错误"
        print(f"✓ sparse_to_dict 测试通过")

        return True
    except Exception as e:
        print(f"✗ 工具模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_milvus():
    """测试检索模块"""
    print("\n" + "=" * 60)
    print("测试: 检索模块")
    print("=" * 60)

    try:
        from retrieval.retrieval import BestsellerRetriever

        retriever = BestsellerRetriever()
        print(f"✓ 检索器创建成功")

        # 检查 Collection
        has_collection = retriever.has_collection()
        print(f"  Collection 存在: {has_collection}")

        if has_collection:
            stats = retriever.get_collection_stats()
            print(f"  记录数: {stats['row_count']}")

        return True
    except Exception as e:
        print(f"✗ 检索模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_files():
    """测试数据文件"""
    print("\n" + "=" * 60)
    print("测试: 数据文件")
    print("=" * 60)

    csv_ok = Path("products.csv").exists()
    new_csv_ok = Path("new_products.csv").exists()
    images_dir = Path("images").exists()
    new_dir = Path("new_products").exists()

    print(f"  products.csv: {'✓' if csv_ok else '✗'}")
    print(f"  new_products.csv: {'✓' if new_csv_ok else '✗'}")
    print(f"  images/: {'✓' if images_dir else '✗'}")
    print(f"  new_products/: {'✓' if new_dir else '✗'}")

    if images_dir:
        images = list(Path("images").glob("*.jpg")) + list(Path("images").glob("*.png"))
        print(f"    商品图片: {len(images)} 张")

    if new_dir:
        new_images = list(Path("new_products").glob("*.jpg")) + list(Path("new_products").glob("*.png"))
        print(f"    新品图片: {len(new_images)} 张")

    return csv_ok and new_csv_ok


def test_embedding():
    """测试嵌入模块"""
    print("\n" + "=" * 60)
    print("测试: 嵌入模块")
    print("=" * 60)

    try:
        from vectorization.embedding import EmbeddingGenerator

        gen = EmbeddingGenerator()

        # 检查能否加载商品
        products, images = gen.load_products()
        print(f"✓ 加载商品: {len(products)} 个")

        # 检查能否加载新品
        new_products = gen.load_new_products()
        print(f"✓ 加载新品: {len(new_products)} 个")

        return True
    except Exception as e:
        print(f"✗ 嵌入模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Nano Banana + Milvus 流水线测试")
    print("=" * 60)

    results = {
        "配置模块": test_config(),
        "数据文件": test_data_files(),
        "工具模块": test_utils(),
        "嵌入模块": test_embedding(),
        "Milvus模块": test_milvus(),
    }

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过! 可以运行流水线了。")
    else:
        print("部分测试失败，请检查配置和数据文件。")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试流水线")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--config", action="store_true", help="测试配置")
    parser.add_argument("--data", action="store_true", help="测试数据文件")
    parser.add_argument("--milvus", action="store_true", help="测试 Milvus")

    args = parser.parse_args()

    if args.all or not any([args.config, args.data, args.milvus]):
        run_all_tests()
    else:
        if args.config:
            test_config()
        if args.data:
            test_data_files()
        if args.milvus:
            test_milvus()
