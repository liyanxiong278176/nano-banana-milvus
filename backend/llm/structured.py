"""
结构化输出模块 - Pydantic Schema 定义
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class QualityScoreSchema(BaseModel):
    """图片质量评分结果"""
    clothing_accuracy: int = Field(description="服装准确性 (1-10)", ge=1, le=10)
    pose_naturalness: int = Field(description="姿势自然度 (1-10)", ge=1, le=10)
    scene_quality: int = Field(description="场景质量 (1-10)", ge=1, le=10)
    lighting_quality: int = Field(description="布光质量 (1-10)", ge=1, le=10)
    commercial_value: int = Field(description="商业价值 (1-10)", ge=1, le=10)

    @property
    def average(self) -> float:
        return round(sum([
            self.clothing_accuracy,
            self.pose_naturalness,
            self.scene_quality,
            self.lighting_quality,
            self.commercial_value
        ]) / 5, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clothing_accuracy": self.clothing_accuracy,
            "pose_naturalness": self.pose_naturalness,
            "scene_quality": self.scene_quality,
            "lighting_quality": self.lighting_quality,
            "commercial_value": self.commercial_value,
            "average": self.average
        }
