"""
电商服饰知识图谱 - 使用示例

演示如何使用知识图谱模块进行构建和检索
"""
from pathlib import Path
from PIL import Image

# 导入知识图谱模块
from graph import FashionGraphBuilder, FashionGraphRetriever


# ==================== 示例1: 构建知识图谱 ====================

def example_build_graph():
    """示例1: 从 products.csv 构建知识图谱"""
    print("\n" + "=" * 60)
    print("示例1: 构建知识图谱")
    print("=" * 60)

    # 使用上下文管理器，自动关闭连接
    with FashionGraphBuilder() as builder:
        # 1. 创建 Schema（约束和索引）
        builder.create_schema()

        # 2. 从 CSV 批量构建图谱
        builder.build_from_csv()


# ==================== 示例2: 插入单个商品 ====================

def example_insert_single_product():
    """示例2: 插入单个商品到图谱"""
    print("\n" + "=" * 60)
    print("示例2: 插入单个商品")
    print("=" * 60)

    with FashionGraphBuilder() as builder:
        # 创建 Schema
        builder.create_schema()

        # 准备商品数据
        product_data = {
            "product_id": "TEST001",
            "category": "midi_dress",
            "style": "elegant",
            "season": "summer",
            "color": "blue",
            "sales_count": 2500,
            "price": 45.99,
            "description": "优雅的蓝色中长裙，适合夏季穿着"
        }

        # 加载商品图片（可选）
        img_path = Path("backend/images/TEST001.jpg")
        product_image = None
        if img_path.exists():
            from utils import load_image
            product_image = load_image(str(img_path))

        # 插入商品
        builder.insert_product(product_data, product_image)

        print("✓ 商品已插入图谱")


# ==================== 示例3: 图谱检索 ====================

def example_graph_retrieval():
    """示例3: 基于知识图谱的检索"""
    print("\n" + "=" * 60)
    print("示例3: 知识图谱检索")
    print("=" * 60)

    with FashionGraphRetriever() as retriever:
        # 场景1: 精确匹配检索
        print("\n[场景1] 精确匹配检索:")
        results = retriever.retrieve_by_graph(
            category="midi_dress",
            style="elegant",
            season="summer",
            top_k=3
        )

        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['product_id']} - {r['category']} / {r['style']} (评分: {r['score']:.2f})")

        # 场景2: 场景推理检索
        print("\n[场景2] 场景推理检索:")
        results = retriever.retrieve_by_graph(
            category="midi_dress",
            style="elegant",
            scene_hint="beach party",  # 场景提示
            top_k=3
        )

        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['product_id']} - {r['match_reason']}")

        # 场景3: 同风格关联检索
        print("\n[场景3] 同风格关联检索:")
        results = retriever.retrieve_by_graph(
            style="elegant",
            season="summer",
            top_k=3
        )


# ==================== 示例4: 混合检索（图谱 + 向量） ====================

def example_hybrid_retrieval():
    """示例4: Milvus 向量检索 + Neo4j 图谱检索混合"""
    print("\n" + "=" * 60)
    print("示例4: 混合检索（向量 + 图谱）")
    print("=" * 60)

    # 查询条件
    category = "midi_dress"
    style = "elegant"
    season = "summer"
    scene_hint = "beach"

    # ==================== 向量检索（Milvus） ====================
    print("\n[1/2] Milvus 向量检索...")
    try:
        from retrieval import BestsellerRetriever
        from embedding import EmbeddingGenerator

        # 初始化组件
        vector_retriever = BestsellerRetriever()
        embed_gen = EmbeddingGenerator()

        # 编码查询（示例：使用虚构的查询向量）
        # 实际使用时，需要调用 embed_gen.encode_new_product()
        # 这里简化演示

        print("  向量检索结果: （需要实际商品数据）")

    except Exception as e:
        print(f"  向量检索失败: {e}")
        vector_results = []

    # ==================== 图谱检索（Neo4j） ====================
    print("\n[2/2] Neo4j 图谱检索...")
    try:
        with FashionGraphRetriever() as graph_retriever:
            graph_results = graph_retriever.retrieve_by_graph(
                category=category,
                style=style,
                season=season,
                scene_hint=scene_hint,
                top_k=5
            )
            print(f"  图谱检索结果: {len(graph_results)} 个商品")
    except Exception as e:
        print(f"  图谱检索失败: {e}")
        graph_results = []

    # ==================== 结果融合 ====================
    print("\n[3/3] 结果融合...")

    # 简单融合策略：去重 + 加权
    all_results = {}

    # 添加图谱结果
    for r in graph_results:
        all_results[r['product_id']] = {
            'product_id': r['product_id'],
            'graph_score': r.get('score', 0),
            'vector_score': 0,
            'match_reason': r.get('match_reason', '')
        }

    # 融合评分（可以调整权重）
    for product_id, result in all_results.items():
        # 加权融合: 图谱 60% + 向量 40%
        result['combined_score'] = (
            result['graph_score'] * 0.6 +
            result['vector_score'] * 0.4
        )

    # 排序
    fused_results = sorted(
        all_results.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )

    print(f"\n融合结果 (Top 5):")
    for i, r in enumerate(fused_results[:5], 1):
        print(f"  {i}. {r['product_id']} - 融合评分: {r['combined_score']:.2f}")


# ==================== 示例5: 商品详情查询 ====================

def example_product_details():
    """示例5: 获取商品详细信息"""
    print("\n" + "=" * 60)
    print("示例5: 商品详情查询")
    print("=" * 60)

    with FashionGraphRetriever() as retriever:
        # 获取商品详情
        product_id = "SKU001"  # 替换为实际商品ID
        details = retriever.get_product_details(product_id)

        if details:
            print(f"\n商品: {product_id}")
            print(f"  品类: {details['categories']}")
            print(f"  风格: {details['styles']}")
            print(f"  季节: {details['seasons']}")
            print(f"  颜色: {details['colors']}")
            print(f"  面料: {details['materials']}")
            print(f"  场景: {details['scenes']}")
        else:
            print(f"  未找到商品: {product_id}")


# ==================== 示例6: 相似商品推荐 ====================

def example_similar_products():
    """示例6: 获取相似商品推荐"""
    print("\n" + "=" * 60)
    print("示例6: 相似商品推荐")
    print("=" * 60)

    with FashionGraphRetriever() as retriever:
        product_id = "SKU001"  # 替换为实际商品ID
        similar = retriever.get_similar_products(product_id, top_k=5)

        print(f"\n与 {product_id} 相似的商品:")
        for i, s in enumerate(similar, 1):
            print(f"  {i}. {s['product_id']} - {s['category']} / {s['style']}")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    import argparse

    parser = argparse.ArgumentParser(description="知识图谱使用示例")
    parser.add_argument("--build", action="store_true", help="构建知识图谱")
    parser.add_argument("--insert", action="store_true", help="插入单个商品")
    parser.add_argument("--retrieve", action="store_true", help="执行图谱检索")
    parser.add_argument("--hybrid", action="store_true", help="执行混合检索")
    parser.add_argument("--details", action="store_true", help="查询商品详情")
    parser.add_argument("--similar", action="store_true", help="查找相似商品")

    args = parser.parse_args()

    if args.build:
        example_build_graph()

    if args.insert:
        example_insert_single_product()

    if args.retrieve:
        example_graph_retrieval()

    if args.hybrid:
        example_hybrid_retrieval()

    if args.details:
        example_product_details()

    if args.similar:
        example_similar_products()

    # 如果没有指定任何操作，显示帮助
    if not any([args.build, args.insert, args.retrieve, args.hybrid, args.details, args.similar]):
        parser.print_help()


if __name__ == "__main__":
    main()
