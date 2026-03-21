"""
测试主入口 - 运行所有评估测试

测试项目：
1. 检索召回率和准确率
2. 成本对比分析
3. 端到端流程测试

使用方法:
    python run_all_tests.py
"""
import sys
from pathlib import Path

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from test.test_retrieval_metrics import RetrievalMetricsTester
from test.test_cost_analysis import CostAnalyzer
from datetime import datetime
import json


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_retrieval_metrics_test():
    """运行检索指标测试"""
    print_header("1. 检索召回率和准确率测试")

    tester = RetrievalMetricsTester()
    results = tester.run_all_tests(top_k=3)

    return results


def run_cost_analysis_test():
    """运行成本分析测试"""
    print_header("2. 生图成本对比分析")

    analyzer = CostAnalyzer()

    # 获取实际新品数量
    from config import NEW_PRODUCT_DIR
    import os
    num_new_products = len([f for f in os.listdir(NEW_PRODUCT_DIR)
                           if f.endswith(('.jpg', '.jpeg', '.png'))])

    print(f"\n检测到 {num_new_products} 个新品图片")

    # 基于实际新品数量进行成本分析
    analyzer.print_comparison_report(num_products=num_new_products)

    # 保存报告
    return analyzer.save_report(num_products=num_new_products)


def save_summary_report(retrieval_results: dict, cost_report_path: str):
    """保存汇总报告"""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"summary_report_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "retrieval_metrics": {
            "total_tests": retrieval_results.get("total_tests", 0),
            "successful_tests": retrieval_results.get("successful_tests", 0),
            "avg_category_match_rate": retrieval_results.get("avg_category_match_rate", 0),
            "avg_ndcg": retrieval_results.get("avg_ndcg", 0),
            "top1_accuracy": retrieval_results.get("top1_accuracy", 0),
            "top3_recall": retrieval_results.get("top3_recall", 0),
        },
        "cost_analysis_report": str(cost_report_path)
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n汇总报告已保存到: {summary_file}")

    return summary_file


def main():
    """主函数"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║        电商 AI 生图流水线 - 评估测试套件                           ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    results = {}

    # 1. 检索指标测试
    try:
        results["retrieval"] = run_retrieval_metrics_test()
    except Exception as e:
        print(f"\n检索测试失败: {e}")
        results["retrieval"] = {"error": str(e)}

    # 2. 成本分析测试
    try:
        results["cost"] = run_cost_analysis_test()
    except Exception as e:
        print(f"\n成本分析失败: {e}")
        results["cost"] = {"error": str(e)}

    # 保存汇总报告
    try:
        save_summary_report(
            results.get("retrieval", {}),
            results.get("cost", "")
        )
    except Exception as e:
        print(f"\n保存汇总报告失败: {e}")

    # 最终总结
    print_header("测试完成")

    retrieval = results.get("retrieval", {})
    if "error" not in retrieval:
        print(f"\n检索测试结果:")
        print(f"  - 成功测试: {retrieval.get('successful_tests', 0)}/{retrieval.get('total_tests', 0)}")
        print(f"  - 平均品类匹配率: {retrieval.get('avg_category_match_rate', 0):.2%}")
        print(f"  - 平均 NDCG: {retrieval.get('avg_ndcg', 0):.4f}")
        print(f"  - Top-1 准确率: {retrieval.get('top1_accuracy', 0):.2%}")

    print("\n所有测试结果保存在 backend/test/results/ 目录下")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
