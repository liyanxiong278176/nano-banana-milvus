"""
用户反馈闭环模块 - P2 阶段

【功能】
1. 收集用户对生成结果的反馈
2. 分析反馈数据
3. 生成提示词优化建议
4. 多语言支持
"""
import json
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


class FeedbackType(Enum):
    """���馈类型"""
    QUALITY = "quality"           # 质量评分
    CLOTHING_MATCH = "clothing"   # 服装匹配度
    STYLE_MATCH = "style"         # 风格匹配度
    OVERALL = "overall"           # 整体满意度
    COMMENT = "comment"           # 文字评论


@dataclass
class UserFeedback:
    """用户反馈记录"""
    feedback_id: str
    task_id: str
    product_id: str
    timestamp: str
    feedback_type: str
    rating: int  # 1-5分或1-10分
    prompt_type: str
    prompt_version: str
    model: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        return asdict(self)


class FeedbackAnalyzer:
    """反馈分析器"""

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = Path(__file__).parent / "cache" / "user_feedback.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.feedbacks: List[UserFeedback] = []
        self._load_feedbacks()

    def _load_feedbacks(self):
        """加载历史反馈"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for fb_data in data:
                        fb = UserFeedback(**fb_data)
                        self.feedbacks.append(fb)
            except Exception as e:
                print(f"[FeedbackAnalyzer] 加载反馈失败: {e}")

    def _save_feedbacks(self):
        """保存反馈"""
        try:
            data = [fb.to_dict() for fb in self.feedbacks]
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[FeedbackAnalyzer] 保存反馈失败: {e}")

    def add_feedback(
        self,
        task_id: str,
        product_id: str,
        feedback_type: str,
        rating: int,
        prompt_type: str,
        prompt_version: str,
        model: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        添加用户反馈

        Args:
            task_id: 任务ID
            product_id: 产品ID
            feedback_type: 反馈类型
            rating: 评分 (1-5 或 1-10)
            prompt_type: 提示词类型
            prompt_version: 提示词版本
            model: 使用的模型
            metadata: 额外元数据

        Returns:
            反馈ID
        """
        import uuid
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"

        feedback = UserFeedback(
            feedback_id=feedback_id,
            task_id=task_id,
            product_id=product_id,
            timestamp=datetime.now().isoformat(),
            feedback_type=feedback_type,
            rating=rating,
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            model=model,
            metadata=metadata or {}
        )

        self.feedbacks.append(feedback)
        self._save_feedbacks()

        return feedback_id

    def get_prompt_performance(self, prompt_type: str, prompt_version: str = None) -> Dict[str, Any]:
        """
        获取提示词性能统计

        Args:
            prompt_type: 提示词类型
            prompt_version: 提示词版本（None表示所有版本）

        Returns:
            性能统计
        """
        filtered = [f for f in self.feedbacks if f.prompt_type == prompt_type]
        if prompt_version:
            filtered = [f for f in filtered if f.prompt_version == prompt_version]

        if not filtered:
            return {"error": "无反馈数据"}

        ratings = [f.rating for f in filtered]
        avg_rating = sum(ratings) / len(ratings)

        # 按评分分组
        rating_distribution = {}
        for r in ratings:
            rating_distribution[r] = rating_distribution.get(r, 0) + 1

        return {
            "prompt_type": prompt_type,
            "prompt_version": prompt_version or "all",
            "total_feedbacks": len(filtered),
            "avg_rating": round(avg_rating, 2),
            "rating_distribution": rating_distribution,
            "satisfaction_rate": sum(1 for r in ratings if r >= 4) / len(ratings) if max(ratings) <= 5 else sum(1 for r in ratings if r >= 7) / len(ratings)
        }

    def compare_versions(self, prompt_type: str) -> Dict[str, Any]:
        """
        比较不同版本的提示词性能

        Args:
            prompt_type: 提示词类型

        Returns:
            版本对比结果
        """
        versions = set(f.prompt_version for f in self.feedbacks if f.prompt_type == prompt_type)

        if len(versions) < 2:
            return {"error": "需要至少两个版本的反馈数据"}

        comparison = {}
        for version in versions:
            stats = self.get_prompt_performance(prompt_type, version)
            if "error" not in stats:
                comparison[version] = {
                    "avg_rating": stats["avg_rating"],
                    "total": stats["total_feedbacks"],
                    "satisfaction": stats["satisfaction_rate"]
                }

        # 找出最佳版本
        best_version = max(comparison.items(), key=lambda x: x[1]["avg_rating"])

        return {
            "prompt_type": prompt_type,
            "comparison": comparison,
            "recommendation": {
                "best_version": best_version[0],
                "best_rating": best_version[1]["avg_rating"]
            }
        }

    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        基于反馈数据生成优化建议

        Returns:
            优化建议列表
        """
        suggestions = []

        # 按提示词类型分析
        prompt_types = set(f.prompt_type for f in self.feedbacks)

        for pt in prompt_types:
            stats = self.get_prompt_performance(pt)

            if "error" in stats:
                continue

            # 检查满意度
            if stats["satisfaction_rate"] < 0.7:
                suggestions.append({
                    "prompt_type": pt,
                    "issue": "satisfaction_low",
                    "current_rate": stats["satisfaction_rate"],
                    "suggestion": "满意度较低，建议检查提示词指令清晰度",
                    "priority": "high"
                })

            # 检查平均评分
            max_rating = 10 if any(f.rating > 5 for f in self.feedbacks if f.prompt_type == pt) else 5
            if stats["avg_rating"] < max_rating * 0.7:
                suggestions.append({
                    "prompt_type": pt,
                    "issue": "rating_low",
                    "current_rating": stats["avg_rating"],
                    "max_rating": max_rating,
                    "suggestion": f"平均评分低于 {max_rating * 0.7}，建议优化提示词内容",
                    "priority": "medium"
                })

            # 检查低分反馈
            low_ratings = [f for f in self.feedbacks if f.prompt_type == pt and f.rating <= 2]
            if len(low_ratings) > len([f for f in self.feedbacks if f.prompt_type == pt]) * 0.2:
                suggestions.append({
                    "prompt_type": pt,
                    "issue": "high_failure_rate",
                    "low_rating_count": len(low_ratings),
                    "suggestion": "低分反馈超过20%，建议增加负向提示词约束",
                    "priority": "high"
                })

        return suggestions

    def print_summary(self):
        """打印反馈摘要"""
        print("\n" + "=" * 60)
        print("用户反馈分析摘要")
        print("=" * 60)

        if not self.feedbacks:
            print("暂无反馈数据")
            return

        # 按类型统计
        prompt_types = set(f.prompt_type for f in self.feedbacks)
        for pt in sorted(prompt_types):
            stats = self.get_prompt_performance(pt)
            if "error" not in stats:
                print(f"\n{pt}:")
                print(f"  反馈数: {stats['total_feedbacks']}")
                print(f"  平均评分: {stats['avg_rating']}")
                print(f"  满意度: {stats['satisfaction_rate']:.1%}")

        # 优化建议
        suggestions = self.get_optimization_suggestions()
        if suggestions:
            print(f"\n优化建议 ({len(suggestions)}条):")
            for i, sg in enumerate(suggestions, 1):
                print(f"  {i}. [{sg['prompt_type']}] {sg['suggestion']} (优先级: {sg['priority']})")

        print("=" * 60 + "\n")


# 全局反馈分析器实例
_global_analyzer: Optional[FeedbackAnalyzer] = None


def get_feedback_analyzer() -> FeedbackAnalyzer:
    """获取全局反馈分析器"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = FeedbackAnalyzer()
    return _global_analyzer


def add_user_feedback(
    task_id: str,
    product_id: str,
    feedback_type: str,
    rating: int,
    prompt_type: str = "quality_judge",
    prompt_version: str = "2.0",
    model: str = "qwen/qwen3-vl-8b-instruct",
    metadata: Dict[str, Any] = None
) -> str:
    """添加用户反馈（便捷函数）"""
    analyzer = get_feedback_analyzer()
    return analyzer.add_feedback(
        task_id=task_id,
        product_id=product_id,
        feedback_type=feedback_type,
        rating=rating,
        prompt_type=prompt_type,
        prompt_version=prompt_version,
        model=model,
        metadata=metadata
    )


# ==================== 多语言支持 ====================

class PromptLocale:
    """提示词语言配置"""

    PROMPTS = {
        "zh": {  # 中文
            "quality_judge": {
                "system": "你是一位专业的电商照片质量评估专家。",
                "rating_scale": "评分（1-5分）",
                "dimensions": {
                    "clothing_accuracy": "服装准确性",
                    "pose_naturalness": "姿势自然度",
                    "scene_quality": "场景质量",
                    "lighting_quality": "布光质量",
                    "commercial_value": "商业价值"
                }
            },
            "style_analysis": {
                "system": "你是一位专业的时尚摄影师和视觉分析师。",
                "instruction": "请分析这张图片的拍摄风格",
                "prohibited": "不要描述服装款式"
            },
            "image_generation": {
                "first_principle": "第一张图片是用户上传的服装，模特身上必须100%还原这件服装！",
                "negative": "【负向约束·严格避免】",
                "requirements": "【正向要求】"
            }
        },
        "en": {  # 英文
            "quality_judge": {
                "system": "You are a professional e-commerce photo quality evaluator.",
                "rating_scale": "Rating (1-5)",
                "dimensions": {
                    "clothing_accuracy": "Clothing Accuracy",
                    "pose_naturalness": "Pose Naturalness",
                    "scene_quality": "Scene Quality",
                    "lighting_quality": "Lighting Quality",
                    "commercial_value": "Commercial Value"
                }
            },
            "style_analysis": {
                "system": "You are a professional fashion photographer and visual analyst.",
                "instruction": "Please analyze the shooting style of this image",
                "prohibited": "Do not describe clothing style"
            },
            "image_generation": {
                "first_principle": "The first image is the user's uploaded clothing. The model must wear this clothing 100% accurately!",
                "negative": "[Negative Constraints - Strictly Avoid]",
                "requirements": "[Positive Requirements]"
            }
        }
    }

    @classmethod
    def get_prompt(cls, prompt_type: str, locale: str = "zh", key: str = "system") -> str:
        """
        获取本地化提示词

        Args:
            prompt_type: 提示词类型
            locale: 语言代码 (zh/en)
            key: 提示词部分

        Returns:
            本地化的提示词文本
        """
        if locale not in cls.PROMPTS:
            locale = "zh"  # 默认中文

        return cls.PROMPTS[locale].get(prompt_type, {}).get(key, "")

    @classmethod
    def translate_prompt(cls, base_prompt: str, locale: str = "zh") -> str:
        """
        翻译提示词中的关键术语

        Args:
            base_prompt: 基础提示词
            locale: 目标语言

        Returns:
            本地化后的提示词
        """
        # 简化版实现：替换关键术语
        if locale == "en":
            translations = {
                "服装准确性": "Clothing Accuracy",
                "姿势自然度": "Pose Naturalness",
                "场景质量": "Scene Quality",
                "布光质量": "Lighting Quality",
                "商业价值": "Commercial Value",
                "第一性原则": "First Principle",
                "负向约束": "Negative Constraints"
            }
            for zh, en in translations.items():
                base_prompt = base_prompt.replace(zh, en)

        return base_prompt

    @classmethod
    def get_supported_locales(cls) -> List[str]:
        """获取支持的语言列表"""
        return list(cls.PROMPTS.keys())


__all__ = [
    "FeedbackType",
    "UserFeedback",
    "FeedbackAnalyzer",
    "get_feedback_analyzer",
    "add_user_feedback",
    "PromptLocale"
]
