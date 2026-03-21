"""
生图成本对比分析

对比传统方式与 AI 方式的成本：

传统方式 (请模特 + 搭建场景):
- 模特费用: 每次 500-2000 元
- 摄影师费用: 每次 500-1500 元
- 场景搭建/场地租赁: 每次 200-1000 元
- 服装造型师: 每次 200-500 元
- 后期修图: 每次 100-300 元
- 时间周期: 3-7 天

AI 方式 (本项目):
- API 调用成本 (按 OpenRouter 价格计算)
- 时间周期: 几分钟

这个脚本计算使用本系统处理新品后的成本节省情况。
"""
import sys
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OPENROUTER_API_KEY, EMBED_MODEL, LLM_MODEL, IMAGE_GEN_MODEL


class CostAnalyzer:
    """成本分析器"""

    # OpenRouter API 价格 (美元/1M tokens 或 每张图)
    # 参考: https://openrouter.ai/models
    API_PRICES = {
        # 嵌入模型
        "nvidia/llama-nemotron-embed-vl-1b-v2": {
            "input": 0.0,   # 免费
            "output": 0.0   # 免费
        },

        # LLM 模型 (风格分析)
        "qwen/qwen3-vl-8b-instruct": {
            "input": 0.05,
            "output": 0.15
        },

        # 图像生成模型
        "bytedance-seed/seedream-4.5": {
            "per_image": 0.04  # 每张图
        }
    }

    # 传统摄影成本范围 (元)
    TRADITIONAL_COSTS = {
        "model": {
            "min": 500,
            "max": 2000,
            "avg": 1200,
            "description": "模特费用"
        },
        "photographer": {
            "min": 500,
            "max": 1500,
            "avg": 900,
            "description": "摄影师费用"
        },
        "studio_rental": {
            "min": 200,
            "max": 1000,
            "avg": 500,
            "description": "影棚/场地租赁"
        },
        "stylist": {
            "min": 200,
            "max": 500,
            "avg": 300,
            "description": "造型师费用"
        },
        "post_processing": {
            "min": 100,
            "max": 300,
            "avg": 200,
            "description": "后期修图"
        }
    }

    def __init__(self):
        """初始化分析器"""
        self.exchange_rate = 7.2  # USD 到 CNY 汇率
        self.results = []

    def calculate_traditional_cost(self, scenario: str = "average") -> Dict:
        """
        计算传统摄影成本

        Args:
            scenario: "min", "max", "average"

        Returns:
            成本明细字典
        """
        key = "min" if scenario == "min" else "max" if scenario == "max" else "avg"

        total = 0
        breakdown = {}

        for cost_type, cost_info in self.TRADITIONAL_COSTS.items():
            amount = cost_info[key]
            breakdown[cost_type] = {
                "amount": amount,
                "description": cost_info["description"]
            }
            total += amount

        return {
            "scenario": scenario,
            "breakdown": breakdown,
            "total_cny": total,
            "total_usd": total / self.exchange_rate
        }

    def calculate_ai_cost(self, num_products: int = 1) -> Dict:
        """
        计算 AI 方式成本

        单个新品处理流程:
        1. 图片嵌入: 1 张新品图片 (免费)
        2. 检索: 不需要额外 API 调用
        3. 风格分析: LLM 调用 (~500 tokens)
        4. 图像生成: 生成 1 张宣传图

        Args:
            num_products: 处理的新品数量

        Returns:
            成本明细字典
        """
        # 每个新品的 API 调用
        per_product_calls = {
            "embedding": {
                "model": EMBED_MODEL,
                "num_images": 1,  # 新品图片
                "tokens_estimate": 0,  # 免费模型
                "cost_usd": 0.0
            },
            "style_analysis": {
                "model": LLM_MODEL,
                "input_tokens": 500,   # 估计
                "output_tokens": 200,  # 估计
                "cost_usd": (
                    (500 / 1_000_000) * self.API_PRICES[LLM_MODEL]["input"] +
                    (200 / 1_000_000) * self.API_PRICES[LLM_MODEL]["output"]
                )
            },
            "image_generation": {
                "model": IMAGE_GEN_MODEL,
                "num_images": 1,  # 生成 1 张
                "cost_usd": 1 * self.API_PRICES[IMAGE_GEN_MODEL]["per_image"]
            }
        }

        # 单个新品总成本
        per_product_usd = sum(c["cost_usd"] for c in per_product_calls.values())

        # 批量成本
        total_usd = per_product_usd * num_products
        total_cny = total_usd * self.exchange_rate

        return {
            "num_products": num_products,
            "per_product_breakdown": per_product_calls,
            "per_product_usd": per_product_usd,
            "per_product_cny": per_product_usd * self.exchange_rate,
            "total_usd": total_usd,
            "total_cny": total_cny
        }

    def calculate_savings(self, num_products: int = 1, traditional_scenario: str = "average") -> Dict:
        """
        计算节省成本

        Args:
            num_products: 新品数量
            traditional_scenario: 传统成本场景

        Returns:
            节省分析结果
        """
        traditional = self.calculate_traditional_cost(traditional_scenario)
        ai = self.calculate_ai_cost(num_products)

        # 传统方式: 每个新品都需要完整流程
        traditional_total_cny = traditional["total_cny"] * num_products
        traditional_total_usd = traditional["total_usd"] * num_products

        savings_cny = traditional_total_cny - ai["total_cny"]
        savings_usd = traditional_total_usd - ai["total_usd"]
        savings_percent = (savings_cny / traditional_total_cny) * 100

        return {
            "num_products": num_products,
            "traditional_scenario": traditional_scenario,
            "traditional_cost_cny": traditional_total_cny,
            "traditional_cost_usd": traditional_total_usd,
            "ai_cost_cny": ai["total_cny"],
            "ai_cost_usd": ai["total_usd"],
            "savings_cny": savings_cny,
            "savings_usd": savings_usd,
            "savings_percent": savings_percent,
            "cost_ratio": ai["total_cny"] / traditional_total_cny
        }

    def print_comparison_report(self, num_products: int = 1):
        """
        打印成本对比报告

        Args:
            num_products: 新品数量
        """
        print("\n" + "=" * 70)
        print("电商宣传图生成成本对比分析")
        print("=" * 70)

        # 传统方式详细
        print("\n【传统方式】请模特 + 搭建场景拍摄")
        print("-" * 70)
        traditional = self.calculate_traditional_cost("average")
        for cost_type, info in traditional["breakdown"].items():
            print(f"  {info['description']:15s}: ¥{info['amount']:6,.0f}")
        print(f"  {'总计':15s}: ¥{traditional['total_cny']:6,.0f} (约 ${traditional['total_usd']:.2f})")

        # AI 方式详细
        print("\n【AI 方式】本系统 (基于向量检索 + AIGC)")
        print("-" * 70)
        ai = self.calculate_ai_cost(num_products)
        print(f"  处理新品数量: {num_products}")
        print(f"\n  每个新品 API 调用成本:")
        print(f"    图片嵌入 ({EMBED_MODEL}): ${ai['per_product_breakdown']['embedding']['cost_usd']:.4f} (免费)")
        print(f"    风格分析 ({LLM_MODEL}): ${ai['per_product_breakdown']['style_analysis']['cost_usd']:.4f}")
        print(f"    图像生成 ({IMAGE_GEN_MODEL}, 1张): ${ai['per_product_breakdown']['image_generation']['cost_usd']:.4f}")
        print(f"  每个新品成本: ¥{ai['per_product_cny']:.2f}")
        print(f"  总计: ¥{ai['total_cny']:.2f} (约 ${ai['total_usd']:.2f})")

        # 节省分析
        print("\n【节省分析】")
        print("-" * 70)
        savings = self.calculate_savings(num_products, "average")
        print(f"  传统方式成本: ¥{savings['traditional_cost_cny']:,.0f}")
        print(f"  AI 方式成本:   ¥{savings['ai_cost_cny']:,.0f}")
        print(f"  节省金额:     ¥{savings['savings_cny']:,.0f} (约 ${savings['savings_usd']:,.0f})")
        print(f"  节省比例:     {savings['savings_percent']:.1f}%")
        print(f"  成本比例:     AI/传统 = {savings['cost_ratio']:.1%}")

        # 不同规模场景
        print("\n【不同规模成本对比】")
        print("-" * 70)
        print(f"  {'新品数量':<10s} {'传统方式':<15s} {'AI方式':<15s} {'节省金额':<15s} {'节省比例'}")
        print("-" * 70)
        for n in [1, 10, 50, 100, 500]:
            s = self.calculate_savings(n, "average")
            print(f"  {n:<10d} ¥{s['traditional_cost_cny']:>10,.0f} ¥{s['ai_cost_cny']:>10,.0f} "
                  f"¥{s['savings_cny']:>10,.0f} {s['savings_percent']:>10.1f}%")

        # 时间对比
        print("\n【时间效率对比】")
        print("-" * 70)
        print("  传统方式:")
        print("    - 筹备 (预约模特、场地): 1-3 天")
        print("    - 拍摄 (含布光、调整): 0.5-1 天")
        print("    - 后期修图: 1-3 天")
        print("    - 总计: 3-7 天")
        print("\n  AI 方式:")
        print("    - 图片上传 + 参数设置: 1 分钟")
        print("    - 向量检索: <1 秒")
        print("    - 风格分析: 10-30 秒")
        print("    - 图像生成: 30-60 秒")
        print("    - 总计: 2-5 分钟")
        print("\n  效率提升: 约 1000-2000 倍")

        print("\n" + "=" * 70)

    def save_report(self, num_products: int = 1):
        """
        保存分析报告到 JSON 文件

        Args:
            num_products: 新品数量
        """
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"cost_analysis_{timestamp}.json"

        traditional = self.calculate_traditional_cost("average")
        ai = self.calculate_ai_cost(num_products)
        savings = self.calculate_savings(num_products, "average")

        # 不同规模对比
        scale_comparison = []
        for n in [1, 10, 50, 100, 500]:
            s = self.calculate_savings(n, "average")
            scale_comparison.append({
                "num_products": n,
                "traditional_cost_cny": s["traditional_cost_cny"],
                "ai_cost_cny": s["ai_cost_cny"],
                "savings_cny": s["savings_cny"],
                "savings_percent": s["savings_percent"]
            })

        report = {
            "timestamp": timestamp,
            "exchange_rate": self.exchange_rate,
            "traditional_breakdown": traditional,
            "ai_breakdown": ai,
            "savings_analysis": savings,
            "scale_comparison": scale_comparison,
            "api_prices_reference": self.API_PRICES
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n分析报告已保存到: {report_file}")

        return report_file


def main():
    """主函数"""
    analyzer = CostAnalyzer()

    # 打印对比报告 (默认处理 10 个新品)
    analyzer.print_comparison_report(num_products=10)

    # 保存报告
    analyzer.save_report(num_products=10)


if __name__ == "__main__":
    main()
