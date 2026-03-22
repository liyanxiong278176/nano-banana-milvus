"""
提示词模板管理 - 基于 LangChain
"""
from typing import Dict, Tuple
from pathlib import Path

# LangChain 导入
from langchain_core.prompts import ChatPromptTemplate

# 项目导入
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== 提示词模板 ====================

class StyleAnalysisPrompts:
    """风格分析提示词"""

    SINGLE = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的时尚摄影师和视觉分析师。"),
        ("human", """这是一张时尚产品宣传照片。

【核心原则】你只需要分析**摄影拍摄风格**，绝对不要描述模特身上的服装款式！

【分析维度】
1. 场景/背景设置
2. 光线和色调
3. 模特姿势和构图
4. 整体氛围和美学风格

【严格禁止】不要描述服装的款式、颜色、材质、图案！

请用一句话（50字以内）总结这张图片的**拍摄风格**。
只输出拍摄风格描述，不要其他内容。""")
    ])

    COMBINED = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的时尚摄影师和视觉分析师。"),
        ("human", """这些是我们店铺最畅销的时尚产品宣传照片。

【核心原则】你只需要分析**摄影拍摄风格**，绝对不要描述服装款式！

【分析维度】
1. 场景/背景设置
2. 光线和色调
3. 模特姿势和构图
4. 整体氛围和美学风格

【严格禁止】不要描述服装的款式、颜色、材质、图案！

基于以上分析，写一段简洁的图像生成提示词（150字以内），
描述**拍摄场景、光线、构图风格**（不涉及具体服装款式）。

只输出提示词，不要输出其他内容。""")
    ])


class QualityJudgePrompts:
    """质量评估提示词"""

    BASE = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的电商照片质量评估专家。"),
        ("human", """**图片说明**：
- 图片1：原始产品平铺图
- 图片2：AI 生成的宣传图

请评估生成的宣传图（图片2）的质量，对以下维度进行评分（1-10分）：

1. **clothing_accuracy**（服装准确性）：服装与原图的匹配度
2. **pose_naturalness**（姿势自然度）：模特姿势和服装合身度
3. **scene_quality**（场景质量）：背景/场景专业度
4. **lighting_quality**（布光质量）：光线质量
5. **commercial_value**（商业价值）：是否适合电商使用

只输出 JSON 格式：
{{"clothing_accuracy": 8, "pose_naturalness": 7, "scene_quality": 9, "lighting_quality": 8, "commercial_value": 8}}""")
    ])

    WITH_REF = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的电商照片质量评估专家。"),
        ("human", """**图片说明**：
- 图片1：原始产品平铺图
- 图片2：AI 生成的宣传图
- 图片3及以后：参考爆款图

请评估生成的宣传图（图片2）的质量，对以下维度进行评分（1-10分）：

1. **clothing_accuracy**（服装准确性）：服装与原图的匹配度
2. **pose_naturalness**（姿势自然度）：模特姿势和服装合身度
3. **scene_quality**（场景质量）：背景/场景专业度
4. **lighting_quality**（布光质量）：光线质量
5. **commercial_value**（商业价值）：是否适合电商使用

只输出 JSON 格式：
{{"clothing_accuracy": 8, "pose_naturalness": 7, "scene_quality": 9, "lighting_quality": 8, "commercial_value": 8}}""")
    ])


# ==================== 提示词管理器 ====================

class PromptTemplateManager:
    """提示词模板管理器"""

    def __init__(self):
        pass

    # 风格分析
    def get_style_analysis_single(self) -> str:
        """获取单图风格分析提示词"""
        return StyleAnalysisPrompts.SINGLE.messages[1].prompt.template

    def get_style_analysis_combined(self) -> str:
        """获取多图综合风格分析提示词"""
        return StyleAnalysisPrompts.COMBINED.messages[1].prompt.template

    def get_style_analysis_prompt(self, has_references: bool = False) -> str:
        """获取风格分析提示词"""
        if has_references:
            return self.get_style_analysis_combined()
        return self.get_style_analysis_single()

    # 质量评估
    def get_quality_judge_base(self) -> str:
        """获取基础质量评估提示词"""
        return QualityJudgePrompts.BASE.messages[1].prompt.template

    def get_quality_judge_with_ref(self) -> str:
        """获取带参考图的质量评估提示词"""
        return QualityJudgePrompts.WITH_REF.messages[1].prompt.template

    def get_quality_judge_prompt(self, has_references: bool = False) -> str:
        """获取质量评估提示词"""
        if has_references:
            return self.get_quality_judge_with_ref()
        return self.get_quality_judge_base()


# 单例
_prompt_manager = None


def get_prompt_manager() -> PromptTemplateManager:
    """获取提示词管理器实例"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptTemplateManager()
    return _prompt_manager
