"""
数据库初始化脚本 - 快速构建向量数据库
"""
from pathlib import Path
from milvus_db import FashionMilvusDB
from embedding import EmbeddingGenerator


def init_database(overwrite: bool = False):
    """
    初始化向量数据库

    Args:
        overwrite: 是否覆盖已存在的数据库
    """
    print("=" * 60)
    print("向量数据库初始化")
    print("=" * 60)

    # 检查数据文件
    csv_path = Path("products.csv")
    image_dir = Path("images")

    if not csv_path.exists():
        print(f"错误: 找不到 {csv_path}")
        return False

    if not image_dir.exists():
        print(f"错误: 找不到 {image_dir} 目录")
        return False

    # 统计图片
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"\n找到 {len(image_files)} 张图片")

    # 初始化组件
    milvus_db = FashionMilvusDB()
    embed_gen = EmbeddingGenerator()

    # 创建 Collection
    print("\n创建 Milvus Collection...")
    milvus_db.create_collection(overwrite=overwrite)

    # 生成嵌入向量
    print("\n生成嵌入向量...")
    products, dense_vectors, sparse_vectors, tfidf = \
        embed_gen.process_all_embeddings()

    # 插入数据库
    print("\n插入数据...")
    milvus_db.insert_products(products, dense_vectors, sparse_vectors)

    # 显示统计信息
    stats = milvus_db.get_collection_stats()
    print("\n" + "=" * 60)
    print("初始化完成!")
    print("=" * 60)
    print(f"商品数量: {stats['row_count']}")
    print(f"向量维度: {dense_vectors.shape[1]}")
    from config import MILVUS_URI
    print(f"数据库连接: {MILVUS_URI}")

    return True


def check_database():
    """检查数据库状态"""
    from pathlib import Path

    db_path = Path("milvus_fashion.db")
    csv_path = Path("products.csv")
    image_dir = Path("images")

    print("=" * 60)
    print("数据库状态检查")
    print("=" * 60)

    # 检查数据文件
    print(f"\n数据文件:")
    print(f"  products.csv: {'✓' if csv_path.exists() else '✗'}")
    print(f"  images/: {'✓' if image_dir.exists() else '✗'}")

    if image_dir.exists():
        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        print(f"    图片数量: {len(images)}")

    # 检查数据库
    print(f"\n数据库:")
    print(f"  milvus_fashion.db: {'✓' if db_path.exists() else '✗'}")

    if db_path.exists():
        try:
            from milvus_db import FashionMilvusDB
            db = FashionMilvusDB()
            if db.client.has_collection("fashion_products"):
                stats = db.get_collection_stats()
                print(f"    Collection: fashion_products")
                print(f"    记录数: {stats['row_count']}")
            else:
                print(f"    Collection: 不存在")
        except Exception as e:
            print(f"    错误: {e}")

    print("\n" + "=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="初始化向量数据库")
    parser.add_argument("--init", action="store_true", help="初始化数据库")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的数据库")
    parser.add_argument("--check", action="store_true", help="检查数据库状态")

    args = parser.parse_args()

    if args.check:
        check_database()
    elif args.init:
        init_database(overwrite=args.overwrite)
    else:
        # 默认显示状态
        check_database()
        print("\n使用方法:")
        print("  python init_db.py --check       # 检查状态")
        print("  python init_db.py --init        # 初始化数据库")
        print("  python init_db.py --init --overwrite  # 重建数据库")


if __name__ == "__main__":
    main()
