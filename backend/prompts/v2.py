"""
提示词效果监控模块 - P1 阶段

【功能】
1. 提示词效果追踪
2. A/B 测试支持
3. 指标收集和分析
"""
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class PromptExecutionRecord:
    """提示词执行记录"""
    timestamp: str
    prompt_type: str
    prompt_version: str
    model: str
    input_summary: str
    output_summary: str
    execution_time: float
    success: bool
    error_message: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        return asdict(self)


class PromptMetrics:
    """提示词指标收集器"""

    def __init__(self, storage_path: str = None):
        """
        初始化指标收集器

        Args:
            storage_path: 指标存储路径
        """
        if storage_path is None:
            storage_path = Path(__file__).parent / "cache" / "prompt_metrics.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.records: List[PromptExecutionRecord] = []
        self._load_records()

    def _load_records(self):
        """加载历史记录"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for record_data in data:
                        record = PromptExecutionRecord(**record_data)
                        self.records.append(record)
            except Exception as e:
                print(f"[PromptMetrics] 加载历史记录失败: {e}")

    def _save_records(self):
        """保存记录"""
        try:
            data = [record.to_dict() for record in self.records]
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[PromptMetrics] 保存记录失败: {e}")

    def record_execution(
        self,
        prompt_type: str,
        prompt_version: str,
        model: str,
        input_summary: str,
        output_summary: str,
        execution_time: float,
        success: bool,
        error_message: str = "",
        metadata: Dict[str, Any] = None
    ):
        """记录一次提示词执行"""
        record = PromptExecutionRecord(
            timestamp=datetime.now().isoformat(),
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            model=model,
            input_summary=input_summary[:200],  # 限制长度
            output_summary=output_summary[:200],
            execution_time=execution_time,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        self.records.append(record)

        # 每 10 条记录保存一次
        if len(self.records) % 10 == 0:
            self._save_records()

    def get_metrics(self, prompt_type: str = None, limit: int = 100) -> Dict[str, Any]:
        """获取指标统计"""
        records = self.records[-limit:]
        if prompt_type:
            records = [r for r in records if r.prompt_type == prompt_type]

        if not records:
            return {"error": "没有记录"}

        total = len(records)
        success_count = sum(1 for r in records if r.success)
        avg_time = sum(r.execution_time for r in records) / total

        return {
            "prompt_type": prompt_type or "all",
            "total_executions": total,
            "success_count": success_count,
            "success_rate": success_count / total,
            "avg_execution_time": avg_time,
            "versions": {},
            "models": {}
        }

    def get_prompt_type_stats(self, prompt_type: str) -> Dict[str, Any]:
        """获取特定提示词类型的统计"""
        records = [r for r in self.records if r.prompt_type == prompt_type]

        if not records:
            return {"error": f"没有 {prompt_type} 的记录"}

        # 按版本分组
        versions: Dict[str, List] = {}
        for r in records:
            if r.prompt_version not in versions:
                versions[r.prompt_version] = []
            versions[r.prompt_version].append(r)

        stats = {
            "prompt_type": prompt_type,
            "total_executions": len(records),
            "versions": {}
        }

        for version, ver_records in versions.items():
            success_count = sum(1 for r in ver_records if r.success)
            avg_time = sum(r.execution_time for r in ver_records) / len(ver_records)
            stats["versions"][version] = {
                "count": len(ver_records),
                "success_count": success_count,
                "success_rate": success_count / len(ver_records),
                "avg_time": avg_time
            }

        return stats

    def print_summary(self):
        """打印指标摘要"""
        print("\n" + "=" * 60)
        print("提示词执行指标摘要")
        print("=" * 60)

        # 按类型统计
        prompt_types = set(r.prompt_type for r in self.records)
        for pt in sorted(prompt_types):
            stats = self.get_prompt_type_stats(pt)
            if "error" not in stats:
                print(f"\n{pt}:")
                print(f"  总执行次数: {stats['total_executions']}")
                for ver, ver_stats in stats['versions'].items():
                    print(f"  v{ver}: {ver_stats['count']}次, "
                          f"成功率{ver_stats['success_rate']:.1%}, "
                          f"平均{ver_stats['avg_time']:.2f}秒")

        print("=" * 60 + "\n")

    def clear_old_records(self, keep_days: int = 7):
        """清理旧记录"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=keep_days)
        cutoff_str = cutoff.isoformat()

        old_count = len(self.records)
        self.records = [r for r in self.records if r.timestamp > cutoff_str]
        new_count = len(self.records)

        if old_count != new_count:
            self._save_records()
            print(f"[PromptMetrics] 清理了 {old_count - new_count} 条旧记录")


# 全局指标收集器实例
_global_metrics: Optional[PromptMetrics] = None


def get_metrics() -> PromptMetrics:
    """获取全局指标收集器"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = PromptMetrics()
    return _global_metrics


def record_prompt_execution(
    prompt_type: str,
    prompt_version: str,
    model: str,
    input_summary: str,
    output_summary: str,
    execution_time: float,
    success: bool,
    error_message: str = "",
    metadata: Dict[str, Any] = None
):
    """记录提示词执行（便捷函数）"""
    metrics = get_metrics()
    metrics.record_execution(
        prompt_type=prompt_type,
        prompt_version=prompt_version,
        model=model,
        input_summary=input_summary,
        output_summary=output_summary,
        execution_time=execution_time,
        success=success,
        error_message=error_message,
        metadata=metadata
    )


# ==================== 少样本示例配置 ====================

FEW_SHOT_EXAMPLES = {
    "quality_judge": [
        {
            "name": "优秀示例",
            "description": "服装完全一致，姿势自然",
            "input": "原图：白色A字连衣裙 | 生成图：白色A字连衣裙，模特站姿自然",
            "output": {
                "clothing_accuracy": 5,
                "pose_naturalness": 5,
                "scene_quality": 5,
                "lighting_quality": 5,
                "commercial_value": 5,
                "reasoning": "服装版型、颜色、材质完全一致，模特姿势自然优雅，场景专业，光线柔和，非常适合电商使用"
            }
        },
        {
            "name": "良好示例",
            "description": "服装基本一致，轻微色差",
            "input": "原图：蓝色修身连衣裙 | 生成图：蓝色修身连衣裙，颜色略深",
            "output": {
                "clothing_accuracy": 4,
                "pose_naturalness": 5,
                "scene_quality": 4,
                "lighting_quality": 4,
                "commercial_value": 4,
                "reasoning": "服装版型和款式一致，但颜色有轻微偏差，整体效果良好，轻微修图即可使用"
            }
        },
        {
            "name": "不合格示例",
            "description": "服装款式变化",
            "input": "原图：红色修身连衣裙 | 生成图：粉色宽松连衣裙",
            "output": {
                "clothing_accuracy": 2,
                "pose_naturalness": 4,
                "scene_quality": 4,
                "lighting_quality": 4,
                "commercial_value": 2,
                "reasoning": "服装款式完全不符（修身变宽松，红色变粉色），虽然姿势和场景不错，但服装准确性太低，不适合使用"
            }
        }
    ],
    "retrieval_judge": [
        {
            "name": "完美匹配",
            "input": "查询: midi_dress, elegant, summer | 结果: midi_dress, elegant, summer, 销量2500",
            "output": {
                "category_match": 10,
                "style_match": 10,
                "scene_match": 10,
                "attribute_match": 10,
                "reasoning": "品类、风格、季节完全匹配，销量高，完美匹配"
            }
        },
        {
            "name": "良好匹配",
            "input": "查询: midi_dress, elegant, summer | 结果: maxi_dress, elegant, summer, 销量1800",
            "output": {
                "category_match": 7,
                "style_match": 10,
                "scene_match": 10,
                "attribute_match": 8,
                "reasoning": "品类相关（同属dress类），风格和季节完全匹配，良好匹配"
            }
        }
    ]
}


def get_few_shot_examples(prompt_type: str) -> List[Dict]:
    """获取少样本示例"""
    return FEW_SHOT_EXAMPLES.get(prompt_type, [])


def build_few_shot_prompt(
    base_prompt: str,
    prompt_type: str,
    include_examples: bool = True,
    max_examples: int = 2
) -> str:
    """
    构建包含少样本示例的提示词

    Args:
        base_prompt: 基础提示词
        prompt_type: 提示词类型
        include_examples: 是否包含示例
        max_examples: 最多包含几个示例

    Returns:
        增强后的提示词
    """
    if not include_examples:
        return base_prompt

    examples = get_few_shot_examples(prompt_type)
    if not examples:
        return base_prompt

    # 构建示例部分
    examples_section = "\n【参考示例】\n"
    for i, example in enumerate(examples[:max_examples], 1):
        examples_section += f"\n示例{i}：{example.get('name', '')}\n"
        if example.get('description'):
            examples_section += f"说明：{example['description']}\n"
        examples_section += f"输入：{example['input']}\n"

        output = example.get('output', {})
        if isinstance(output, dict):
            examples_section += f"输出：{json.dumps(output, ensure_ascii=False)}\n"
        else:
            examples_section += f"输出：{output}\n"

    examples_section += "\n请参考以上示例，评估当前输入：\n"

    # 将示例插入到提示词中
    return examples_section + base_prompt


# ==================== A/B 测试支持 ====================

class ABTestConfig:
    """A/B 测试配置"""

    def __init__(self):
        self.active_tests: Dict[str, Dict] = {}

    def create_test(
        self,
        test_name: str,
        prompt_type: str,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5
    ):
        """
        创建 A/B 测试

        Args:
            test_name: 测试名称
            prompt_type: 提示词类型
            version_a: A 版本
            version_b: B 版本
            traffic_split: A 版本流量比例 (0-1)
        """
        self.active_tests[test_name] = {
            "prompt_type": prompt_type,
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
            "results": {
                "version_a": {"count": 0, "success": 0},
                "version_b": {"count": 0, "success": 0}
            }
        }

    def get_version_for_test(self, test_name: str) -> Optional[str]:
        """获取当前请求应使用的版本"""
        import random

        test = self.active_tests.get(test_name)
        if not test:
            return None

        # 简单随机分流
        if random.random() < test["traffic_split"]:
            version = test["version_a"]
            test["results"]["version_a"]["count"] += 1
        else:
            version = test["version_b"]
            test["results"]["version_b"]["count"] += 1

        return version

    def record_success(self, test_name: str, version: str):
        """记录成功"""
        test = self.active_tests.get(test_name)
        if test:
            key = "version_a" if version == test["version_a"] else "version_b"
            test["results"][key]["success"] += 1

    def get_test_results(self, test_name: str) -> Optional[Dict]:
        """获取测试结果"""
        test = self.active_tests.get(test_name)
        if not test:
            return None

        results_a = test["results"]["version_a"]
        results_b = test["results"]["version_b"]

        return {
            "test_name": test_name,
            "version_a": {
                "version": test["version_a"],
                "count": results_a["count"],
                "success": results_a["success"],
                "success_rate": results_a["success"] / results_a["count"] if results_a["count"] > 0 else 0
            },
            "version_b": {
                "version": test["version_b"],
                "count": results_b["count"],
                "success": results_b["success"],
                "success_rate": results_b["success"] / results_b["count"] if results_b["count"] > 0 else 0
            }
        }


# 全局 A/B 测试实例
_global_ab_test: Optional[ABTestConfig] = None


def get_ab_test() -> ABTestConfig:
    """获取全局 A/B 测试实例"""
    global _global_ab_test
    if _global_ab_test is None:
        _global_ab_test = ABTestConfig()
    return _global_ab_test


__all__ = [
    "PromptExecutionRecord",
    "PromptMetrics",
    "get_metrics",
    "record_prompt_execution",
    "get_few_shot_examples",
    "build_few_shot_prompt",
    "ABTestConfig",
    "get_ab_test"
]
