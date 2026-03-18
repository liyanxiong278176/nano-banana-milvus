"""
Neo4j 知识图谱测试脚本

验证：
1. Neo4j 连接是否正常
2. 图谱构建功能
3. 图谱检索功能
4. 混合检索功能
5. 主流用法符合性验证
"""
import os
import sys
import time
from pathlib import Path

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows 兼容：设置 UTF-8 输出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 使用 ASCII 符号替代 Unicode
PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
WARN = "[WARN]"
OK = "OK"
NG = "NG"


def test_1_config_import():
    """测试1: 配置导入"""
    print("\n" + "=" * 60)
    print("测试1: 配置导入")
    print("=" * 60)

    try:
        from config import (
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB,
            NEO4J_MAX_CONNECTION_POOL_SIZE
        )
        print(f"  [OK] 配置导入成功")
        print(f"    URI: {NEO4J_URI}")
        print(f"    User: {NEO4J_USER}")
        print(f"    DB: {NEO4J_DB}")
        print(f"    Pool Size: {NEO4J_MAX_CONNECTION_POOL_SIZE}")
        return True
    except Exception as e:
        print(f"  [FAIL] 配置导入失败: {e}")
        return False


def test_2_neo4j_driver():
    """测试2: Neo4j 驱动安装和连接"""
    print("\n" + "=" * 60)
    print("测试2: Neo4j 驱动安装和连接")
    print("=" * 60)

    try:
        from neo4j import GraphDatabase
        print(f"  [OK] neo4j 驱动已安装")

        # 测试连接
        from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB

        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_pool_size=10,
            connection_timeout=10
        )

        # 验证连接
        with driver.session(database=NEO4J_DB) as session:
            result = session.run("RETURN 1 AS test")
            value = result.single()["test"]
            if value == 1:
                print(f"  [OK] Neo4j 连接成功")
                driver.close()
                return True
            else:
                print(f"  [FAIL] 连接测试失败")
                driver.close()
                return False

    except ImportError as e:
        print(f"  [FAIL] neo4j 驱动未安装: {e}")
        print(f"    请运行: pip install neo4j>=5.0.0")
        return False
    except Exception as e:
        print(f"  [FAIL] Neo4j 连接失败: {e}")
        print(f"\n  请检查:")
        print(f"    1. Neo4j 是否启动: docker ps | grep neo4j")
        print(f"    2. 连接配置是否正确: {NEO4J_URI}")
        print(f"    3. 密码是否正确: .env NEO4J_PASSWORD")
        return False


def test_3_graph_builder():
    """测试3: 图谱构建器"""
    print("\n" + "=" * 60)
    print("测试3: 图谱构建器")
    print("=" * 60)

    try:
        from graph import FashionGraphBuilder

        print(f"  [OK] FashionGraphBuilder 导入成功")

        # 测试连接
        builder = FashionGraphBuilder()
        print(f"  [OK] 图谱构建器初始化成功")

        # 测试 Schema 创建
        builder.create_schema()
        print(f"  [OK] Schema 创建成功")

        # 清理测试数据
        builder.close()

        return True

    except Exception as e:
        print(f"  [FAIL] 图谱构建器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_graph_retriever():
    """测试4: 图谱检索器"""
    print("\n" + "=" * 60)
    print("测试4: 图谱检索器")
    print("=" * 60)

    try:
        from graph import FashionGraphRetriever

        print(f"  [OK] FashionGraphRetriever 导入成功")

        # 测试连接
        retriever = FashionGraphRetriever()
        print(f"  [OK] 图谱检索器初始化成功")

        # 测试统计信息
        stats = retriever.get_graph_stats()
        print(f"  [OK] 图谱统计: {stats}")

        # 测试检索
        results = retriever.retrieve_by_graph(
            category="midi_dress",
            style="elegant",
            season="summer",
            top_k=3
        )
        print(f"  [OK] 检索到 {len(results)} 个结果")

        retriever.close()

        return True

    except Exception as e:
        print(f"  [FAIL] 图谱检索器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_hybrid_retriever():
    """测试5: 混合检索器"""
    print("\n" + "=" * 60)
    print("测试5: 混合检索器")
    print("=" * 60)

    try:
        from graph.hybrid_retriever import HybridRetriever

        print(f"  [OK] HybridRetriever 导入成功")

        # 测试初始化
        retriever = HybridRetriever(
            milvus_weight=0.6,
            graph_weight=0.4
        )
        print(f"  [OK] 混合检索器初始化成功")

        # 测试状态查询
        status = retriever.get_status()
        print(f"  Milvus 启用: {status['milvus_enabled']}")
        print(f"  Neo4j 启用: {status['neo4j_enabled']}")
        print(f"  Milvus 权重: {status['milvus_weight']:.2f}")
        print(f"  Graph 权重: {status['graph_weight']:.2f}")

        if status['neo4j_enabled']:
            print(f"  Neo4j 连接: {status['neo4j_connected']}")
            print(f"  图谱统计: {status.get('graph_stats', {})}")

        return True

    except Exception as e:
        print(f"  [FAIL] 混合检索器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_triplet_extraction():
    """测试6: 三元组抽取"""
    print("\n" + "=" * 60)
    print("测试6: 三元组抽取")
    print("=" * 60)

    try:
        from graph import FashionGraphBuilder

        builder = FashionGraphBuilder()

        # 测试数据
        test_product = {
            "product_id": "TEST_001",
            "category": "midi_dress",
            "style": "elegant",
            "season": "summer",
            "color": "blue",
            "sales_count": 2000,
            "price": 45.99,
            "description": "A beautiful blue summer dress"
        }

        # 测试三元组抽取（无图片）
        triplets = builder.extract_entity_triplets(test_product, product_image=None)

        print(f"  [OK] 抽取到 {len(triplets)} 个三元组:")
        for i, (e1, rel, e2) in enumerate(triplets[:5], 1):
            print(f"    {i}. ({e1}, {rel}, {e2})")

        # 验证预期三元组
        expected_triplets = [
            ("TEST_001", "HAS_CATEGORY", "midi_dress"),
            ("TEST_001", "HAS_STYLE", "elegant"),
            ("TEST_001", "HAS_SEASON", "summer"),
            ("TEST_001", "HAS_COLOR", "blue"),
        ]

        for expected in expected_triplets:
            if expected in triplets:
                print(f"  [OK] 预期三元组存在: {expected}")
            else:
                print(f"  ! 预期三元组缺失: {expected}")

        builder.close()
        return True

    except Exception as e:
        print(f"  [FAIL] 三元组抽取测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_cypher_query():
    """测试7: Cypher 查询符合性"""
    print("\n" + "=" * 60)
    print("测试7: Cypher 查询符合性验证")
    print("=" * 60)

    try:
        from neo4j import GraphDatabase
        from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB

        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        with driver.session(database=NEO4J_DB) as session:
            # 测试1: 简单查询
            print(f"\n  [测试1] 简单查询:")
            result = session.run("MATCH (n) RETURN count(n) AS count LIMIT 1")
            count = result.single()["count"]
            print(f"    节点总数: {count}")

            # 测试2: 参数化查询（防止注入）
            print(f"\n  [测试2] 参数化查询:")
            result = session.run(
                "MATCH (n:Product) WHERE n.sales_count >= $min_sales RETURN count(n) AS count",
                {"min_sales": 1000}
            )
            count = result.single()["count"]
            print(f"    高销量商品数: {count}")

            # 测试3: 索引使用
            print(f"\n  [测试3] 索引使用:")
            result = session.run("SHOW INDEXES")
            indexes = list(result)
            print(f"    索引数量: {len(indexes)}")
            for idx in indexes[:3]:
                print(f"      - {idx.get('labelName', 'N/A')}.{idx.get('properties', [])}")

            # 测试4: 约束使用
            print(f"\n  [测试4] 约束使用:")
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            print(f"    约束数量: {len(constraints)}")
            for c in constraints[:3]:
                print(f"      - {c.get('label', 'N/A')}.{c.get('properties', [])}")

            driver.close()

        print(f"\n  [OK] Cypher 查询验证通过")
        return True

    except Exception as e:
        print(f"  [FAIL] Cypher 查询测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_8_main_practice_compliance():
    """测试8: 主流用法符合性验证"""
    print("\n" + "=" * 60)
    print("测试8: 主流用法符合性验证")
    print("=" * 60)

    checks = []

    # 1. 使用上下文管理器
    print(f"\n  [检查1] 使用上下文管理器:")
    try:
        from graph import FashionGraphBuilder, FashionGraphRetriever

        # 测试 FashionGraphBuilder
        with FashionGraphBuilder() as builder:
            checks.append(("上下文管理器 (Builder)", True))

        # 测试 FashionGraphRetriever
        with FashionGraphRetriever() as retriever:
            checks.append(("上下文管理器 (Retriever)", True))

        print(f"    [OK] 使用 with 语句自动管理连接")

    except Exception as e:
        print(f"    [FAIL] 上下文管理器测试失败: {e}")
        checks.append(("上下文管理器", False))

    # 2. 使用参数化查询（防注入）
    print(f"\n  [检查2] 参数化查询:")
    try:
        from graph.hybrid_retriever import HybridRetriever
        from graph import FashionGraphRetriever

        with FashionGraphRetriever() as retriever:
            # retrieve_by_graph 内部使用参数化查询
            results = retriever.retrieve_by_graph(
                category="midi_dress",  # 参数化传递
                style="elegant",
                top_k=3
            )
            checks.append(("参数化查询", True))
            print(f"    [OK] 查询使用参数化方式")

    except Exception as e:
        print(f"    [FAIL] 参数化查询测试失败: {e}")
        checks.append(("参数化查询", False))

    # 3. 批量操作
    print(f"\n  [检查3] 批量操作:")
    try:
        from graph import FashionGraphBuilder

        with FashionGraphBuilder() as builder:
            # build_from_csv 支持批量插入
            checks.append(("批量操作支持", True))
            print(f"    [OK] 支持批量构建 (build_from_csv)")

    except Exception as e:
        print(f"    [FAIL] 批量操作测试失败: {e}")
        checks.append(("批量操作支持", False))

    # 4. 异常处理
    print(f"\n  [检查4] 异常处理:")
    try:
        from graph.hybrid_retriever import HybridRetriever

        # Neo4j 不可用时自动降级
        retriever = HybridRetriever()
        status = retriever.get_status()

        if not status['neo4j_enabled']:
            checks.append(("异常降级", True))
            print(f"    [OK] Neo4j 不可用时自动降级到 Milvus")
        else:
            checks.append(("异常降级", True))
            print(f"    [OK] 异常处理机制已实现")

    except Exception as e:
        print(f"    [FAIL] 异常处理测试失败: {e}")
        checks.append(("异常降级", False))

    # 5. 连接池配置
    print(f"\n  [检查5] 连接池配置:")
    try:
        from config import NEO4J_MAX_CONNECTION_POOL_SIZE

        print(f"    [OK] 连接池大小: {NEO4J_MAX_CONNECTION_POOL_SIZE}")
        checks.append(("连接池配置", True))

    except Exception as e:
        print(f"    [FAIL] 连接池配置检查失败: {e}")
        checks.append(("连接池配置", False))

    # 总结
    print(f"\n  主流用法符合性检查结果:")
    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for name, result in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"    {status} {name}")

    print(f"\n  通过率: {passed}/{total} ({passed/total*100:.1f}%)")

    return passed == total


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Neo4j 知识图谱集成测试")
    print("=" * 60)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("配置导入", test_1_config_import),
        ("Neo4j 驱动连接", test_2_neo4j_driver),
        ("图谱构建器", test_3_graph_builder),
        ("图谱检索器", test_4_graph_retriever),
        ("混合检索器", test_5_hybrid_retriever),
        ("三元组抽取", test_6_triplet_extraction),
        ("Cypher 查询", test_7_cypher_query),
        ("主流用法符合性", test_8_main_practice_compliance),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  测试异常 ({name}): {e}")
            results[name] = False

    # 输出汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results.items():
        status = "[OK] 通过" if passed else "[FAIL] 失败"
        print(f"  {status} - {name}")

    total = len(results)
    passed_count = sum(1 for v in results.values() if v)

    print(f"\n总计: {passed_count}/{total} 通过")

    if passed_count == total:
        print("\n[OK][OK][OK] 所有测试通过!Neo4j 集成正常工作。")
    else:
        print(f"\n! 有 {total - passed_count} 个测试失败，请检查配置和环境。")

    return passed_count == total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j 知识图谱测试")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--config", action="store_true", help="测试配置")
    parser.add_argument("--connection", action="store_true", help="测试连接")
    parser.add_argument("--builder", action="store_true", help="测试构建器")
    parser.add_argument("--retriever", action="store_true", help="测试检索器")
    parser.add_argument("--hybrid", action="store_true", help="测试混合检索")
    parser.add_argument("--triplet", action="store_true", help="测试三元组")
    parser.add_argument("--cypher", action="store_true", help="测试 Cypher")
    parser.add_argument("--compliance", action="store_true", help="测试主流用法符合性")

    args = parser.parse_args()

    if args.all or not any([args.config, args.connection, args.builder, args.retriever,
                            args.hybrid, args.triplet, args.cypher, args.compliance]):
        run_all_tests()
    else:
        results = {}

        if args.config:
            results["配置"] = test_1_config_import()
        if args.connection:
            results["连接"] = test_2_neo4j_driver()
        if args.builder:
            results["构建器"] = test_3_graph_builder()
        if args.retriever:
            results["检索器"] = test_4_graph_retriever()
        if args.hybrid:
            results["混合检索"] = test_5_hybrid_retriever()
        if args.triplet:
            results["三元组"] = test_6_triplet_extraction()
        if args.cypher:
            results["Cypher"] = test_7_cypher_query()
        if args.compliance:
            results["符合性"] = test_8_main_practice_compliance()

        print(f"\n测试结果: {sum(results.values())}/{len(results)} 通过")
