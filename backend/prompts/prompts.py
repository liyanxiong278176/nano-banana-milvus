"""
提示词配置模块 - 企业级提示词管理

【设计原则】
1. 版本管理：每个提示词都有版本号
2. 结构化输出：统一使用 JSON 格式
3. 可配置性：支持参数替换
4. 向后兼容：保持原有 API 接口不变
"""
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class PromptConfig:
    """提示词配置"""
    version: str
    prompt: str
    system_prompt: str = ""
    parameters: Dict[str, Any] = None
    examples: List[Dict] = None
    negative_prompt: str = ""

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.examples is None:
            self.examples = []


# ==================== 提示词版本配置 ====================

class Prompts:
    """提示词配置中心"""

    # ==================== 风格分析提示词 ====================

    STYLE_ANALYSIS_V2 = PromptConfig(
        version="2.0",
        system_prompt="你是一位专业的时尚摄影师和视觉分析师，擅长分析商业照片的拍摄风格。",
        prompt="""这是我们的时尚产品宣传照片。

【核心原则】
你只需要分析**摄影拍摄风格**，绝对不要描述模特身上的服装款式！

【分析维度】
请按以下结构分析这张图片的**拍摄风格**：

1. **场景设置**
   - 背景类型：[纯色背景/自然户外/室内场景/影棚等]
   - 场景元素：[具体描述，如植物、家具、天空等]

2. **光线特点**
   - 光线类型：[自然光/影棚光/混合光]
   - 主光方向：[顺光/侧光/逆光/顶光]
   - 光线质感：[柔和/强烈/漫反射]
   - 色调倾向：[暖色/冷色/中性]

3. **构图风格**
   - 景别：[全身/半身/特写/七分身]
   - 拍摄角度：[平视/俯视/仰视]
   - 构图方式：[居中构图/三分法/对角线/留白]

4. **整体氛围**
   - 风格关键词：[3-5个词，如极简、商务、清新、复古等]
   - 画面情绪：[专业/轻松/优雅/动感等]

【严格禁止】
- 不要描述服装的款式、颜色、材质、图案！
- 不要描述模特的外貌特征！
- 不要评价服装的美丑！

【输出格式】
只输出 JSON 格式：
```json
{{
  "scene": {{
    "background_type": "背景类型",
    "elements": "场景元素描述"
  }},
  "lighting": {{
    "type": "光线类型",
    "direction": "主光方向",
    "quality": "光线质感",
    "tone": "色调倾向"
  }},
  "composition": {{
    "shot_type": "景别",
    "angle": "拍摄角度",
    "method": "构图方式"
  }},
  "atmosphere": {{
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "mood": "画面情绪"
  }},
  "summary": "一句话总结拍摄风格（50字以内）"
}}
```
""",
        parameters={
            "temperature": 0.7,
            "max_tokens": 500
        }
    )

    # 综合风格分析（多张参考图）
    STYLE_ANALYSIS_COMBINED_V2 = PromptConfig(
        version="2.0",
        system_prompt="你是一位专业的时尚摄影师和视觉分析师。",
        prompt="""这些是我们店铺最畅销的时尚产品宣传照片。

【核心原则】
你只需要分析**摄影拍摄风格**，绝对不要描述服装款式！

【分析维度】
1. 场景/背景设置
2. 光线和色调
3. 模特姿势和构图
4. 整体氛围和美学风格

【严格禁止】
- 不要描述服装的款式、颜色、材质、图案！

【输出要求】
基于以上分析，写一段简洁的图像生成提示词（150字以内），
描述**拍摄场景、光线、构图风格**（不涉及具体服装款式）。

提示词应包含：
- 场景描述
- 光线特点
- 构图方式
- 整体氛围

【输出格式】
只输出提示词文本，不要输出其他内容。
""",
        parameters={
            "temperature": 0.7,
            "max_tokens": 512
        }
    )

    # ==================== 图像生成提示词 ====================

    IMAGE_GEN_V2 = PromptConfig(
        version="2.0",
        prompt="""【第一性原则·最高优先级】第一张图片是用户上传的服装，模特身上必须100%还原这件服装！
版型、材质、颜色、款式、细节完全一致，绝对不允许更改服装款式！

后续图片（图片2及以后）是参考图，仅用于参考**拍摄风格**（场景、光线、构图），
绝对不能模仿参考图里的服装款式、颜色、材质！

请生成一张专业的电商宣传照片，展示一位女性模特穿着**第一张图片中的服装**。

拍摄风格参考（不含服装）：{style_prompt}

【正向要求】：
1. 服装必须与第一张用户上传的原图完全一致
2. 版型、颜色、材质、细节100%还原
3. 全身照，照片级真实感，8K高清
4. 服装与模特身体贴合自然，无变形、无穿模
5. 无可见标签或吊牌，细节清晰
6. 专业商业布光，电商宣传照标准

【场景提示】：{scene_hint}

{negative_prompt_section}
""",
        negative_prompt="""【负向约束·严格避免】：
- 服装款式与原图不一致
- 服装颜色发生改变或偏差
- 服装材质表现错误
- 模特姿势不自然僵硬
- 服装与身体不贴合或有穿模
- 可见的标签、吊牌、logo
- 背景杂乱干扰主体
- 光线过曝或过暗
- 面部扭曲或变形
- 手指数量错误
- 服装细节丢失或模糊""",
        parameters={
            "temperature": 0.8,
            "aspect_ratio": "3:4",
            "image_size": "2K"
        }
    )

    # 单参考图版本
    IMAGE_GEN_SINGLE_V2 = PromptConfig(
        version="2.0",
        prompt="""【第一性原则】图片1是用户上传的服装平铺图，必须100%还原这件服装！
图片2是参考图，仅用于参考**拍摄风格**（场景、光线、构图），不能模仿它的服装款式！

请生成一张专业的电商宣传照片，展示一位女性模特穿着图片1中的服装。

拍摄风格参考（不含服装）：{style_prompt}

【正向要求】：
1. 服装与原图完全一致
2. 全身照，照片级真实感
3. 服装贴合自然，无变形
4. 无可见标签或吊牌
5. 专业商业布光

【场景提示】：{scene_hint}

{negative_prompt_section}
""",
        negative_prompt="""【避免】：
- 服装款式变化
- 颜色偏差
- 姿势不自然
- 背景杂乱
- 光线问题""",
        parameters={
            "temperature": 0.8,
            "aspect_ratio": "3:4",
            "image_size": "2K"
        }
    )

    # ==================== 质量评估提示词 ====================

    QUALITY_JUDGE_V2 = PromptConfig(
        version="2.0",
        system_prompt="你是一位专业的电商照片质量评估专家。",
        prompt="""**图片说明**：
- 图片1：原始产品平铺图（用户上传的服装）
- 图片2：AI 生成的宣传图
{ref_text}

【评估任务】
请评估生成的宣传图（图片2）的质量，对以下维度进行评分（1-10分）：

**评分维度**：

1. **clothing_accuracy**（服装准确性）
   - 9-10分：服装完全一致，版型、颜色、材质、细节完美还原
   - 7-8分：服装基本一致，有细微差异
   - 5-6分：服装大致相似，有明显差异
   - 3-4分：服装部分不一致
   - 1-2分：服装完全不同

2. **pose_naturalness**（姿势自然度）
   - 9-10分：姿势自然优雅，服装贴合完美
   - 7-8分：姿势自然，服装贴合良好
   - 5-6分：姿势基本自然，服装有小问题
   - 3-4分：姿势略显僵硬，服装贴合不佳
   - 1-2分：姿势不自然，服装严重不合身

3. **scene_quality**（场景质量）
   - 9-10分：场景专业，背景简洁美观
   - 7-8分：场景良好，背景适当
   - 5-6分：场景一般，背景可接受
   - 3-4分：场景较差，背景杂乱
   - 1-2分：场景很差，影响主体

4. **lighting_quality**（布光质量）
   - 9-10分：光线专业，层次丰富
   - 7-8分：光线良好，布光合理
   - 5-6分：光线一般，基本可用
   - 3-4分：光线较差，有过曝或过暗
   - 1-2分：光线很差，严重影响观感

5. **commercial_value**（商业价值）
   - 9-10分：非常适合电商使用，可直接上架
   - 7-8分：适合电商使用，轻微修图可用
   - 5-6分：基本可用，需要较多修图
   - 3-4分：不太适合，需要重新拍摄
   - 1-2分：不适合使用

【输出格式】
只输出 JSON 格式：
```json
{{
  "clothing_accuracy": 8,
  "pose_naturalness": 7,
  "scene_quality": 9,
  "lighting_quality": 8,
  "commercial_value": 8
}}
```
""",
        parameters={
            "temperature": 0.3,
            "max_tokens": 500
        }
    )

    # ==================== 检索质量评估提示词 ====================

    RETRIEVAL_QUALITY_JUDGE_V2 = PromptConfig(
        version="2.0",
        system_prompt="你是一位专业的电商检索质量评估专家。",
        prompt="""【查询需求】
{query_context}

【检索结果】
{results_summary}

【评估任务】
请评估检索结果与查询需求的相关性，对以下维度进行评分（0-10分）：

**评分维度**：

1. **category_match**（品类匹配度）
   - 检索结果的品类是否与查询品类相关
   - 完全相同品类：10分
   - 父子品类关系：7-9分
   - 相关品类：4-6分
   - 不相关：0-3分

2. **style_match**（风格匹配度）
   - 风格标签（如优雅、休闲）是否匹配
   - 完全匹配：9-10分
   - 基本匹配：7-8分
   - 部分匹配：5-6分
   - 不匹配：0-4分

3. **scene_match**（场景匹配度）
   - 整体视觉风格是否适合该季节/场景
   - 非常适合：9-10分
   - 比较适合：7-8分
   - 一般适合：5-6分
   - 不太适合：3-4分
   - 完全不适合：0-2分

4. **attribute_match**（属性匹配度）
   - 季节、价格区间等属性的综合匹配度
   - 综合评分，考虑多个属性因素

【评分标准】
- 9-10分：完美匹配，高度相关
- 7-8分：良好匹配，基本符合需求
- 5-6分：一般匹配，部分相关
- 3-4分：匹配度较低，相关性弱
- 0-2分：完全不匹配，无相关性

【输出格式】
只输出 JSON 格式：
```json
{{
  "category_match": 8,
  "style_match": 7,
  "scene_match": 6,
  "attribute_match": 7
}}
```
""",
        parameters={
            "temperature": 0.3,
            "max_tokens": 300
        }
    )


# ==================== 提示词工具函数 ====================

class PromptBuilder:
    """提示词构建器"""

    @staticmethod
    def build_style_analysis(has_references: bool = False) -> str:
        """
        构建风格分析提示词

        Args:
            has_references: 是否有参考图
        """
        config = Prompts.STYLE_ANALYSIS_V2
        if has_references:
            config = Prompts.STYLE_ANALYSIS_COMBINED_V2
        return config.prompt

    @staticmethod
    def build_image_generation(
        style_prompt: str,
        scene_hint: str = "",
        single_reference: bool = False,
        include_negative: bool = True
    ) -> tuple:
        """
        构建图像生成提示词

        Args:
            style_prompt: 风格描述
            scene_hint: 场景提示
            single_reference: 是否单参考图
            include_negative: 是否包含负向提示词

        Returns:
            (prompt, negative_prompt)
        """
        if single_reference:
            config = Prompts.IMAGE_GEN_SINGLE_V2
        else:
            config = Prompts.IMAGE_GEN_V2

        negative_section = ""
        negative_prompt = ""
        if include_negative:
            negative_section = config.negative_prompt
            negative_prompt = config.negative_prompt

        prompt = config.prompt.format(
            style_prompt=style_prompt,
            scene_hint=scene_hint or "无特殊场景要求",
            negative_prompt_section=negative_section
        )

        return prompt, negative_prompt

    @staticmethod
    def build_quality_judge(has_references: bool = False) -> str:
        """构建质量评估提示词"""
        config = Prompts.QUALITY_JUDGE_V2

        ref_text = ""
        if has_references:
            ref_text = "- 图片3及以后：参考爆款图"

        return config.prompt.format(ref_text=ref_text)

    @staticmethod
    def build_retrieval_judge(
        query_context: str,
        results_summary: str
    ) -> str:
        """构建检索质量评估提示词"""
        config = Prompts.RETRIEVAL_QUALITY_JUDGE_V2
        return config.prompt.format(
            query_context=query_context,
            results_summary=results_summary
        )


# ==================== 版本管理 ====================

class PromptVersionManager:
    """提示词版本管理器"""

    _versions = {
        "style_analysis": {
            "current": "2.0",
            "available": {
                "1.0": "原始版本",
                "2.0": "结构化输出版本"
            }
        },
        "image_generation": {
            "current": "2.0",
            "available": {
                "1.0": "原始版本",
                "2.0": "增加负向提示词版本"
            }
        },
        "quality_judge": {
            "current": "2.0",
            "available": {
                "1.0": "原始版本",
                "2.0": "详细评分标准版本"
            }
        },
        "retrieval_judge": {
            "current": "2.0",
            "available": {
                "1.0": "原始版本",
                "2.0": "详细评分标准版本"
            }
        }
    }

    @classmethod
    def get_version(cls, prompt_type: str) -> str:
        """获取当前版本号"""
        return cls._versions.get(prompt_type, {}).get("current", "1.0")

    @classmethod
    def get_all_versions(cls) -> Dict:
        """获取所有版本信息"""
        return cls._versions

    @classmethod
    def print_version_info(cls):
        """打印版本信息"""
        print("\n" + "=" * 60)
        print("提示词版本信息")
        print("=" * 60)
        for prompt_type, info in cls._versions.items():
            print(f"\n{prompt_type}:")
            print(f"  当前版本: {info['current']}")
            print(f"  可用版本:")
            for ver, desc in info['available'].items():
                marker = " [当前]" if ver == info['current'] else ""
                print(f"    - {ver}: {desc}{marker}")
        print("=" * 60 + "\n")


# ==================== 导出接口 ====================

__all__ = [
    "PromptConfig",
    "Prompts",
    "PromptBuilder",
    "PromptVersionManager"
]


if __name__ == "__main__":
    PromptVersionManager.print_version_info()

    # 测试构建器
    print("\n=== 测试提示词构建 ===\n")

    print("1. 风格分析提示词:")
    style_prompt = PromptBuilder.build_style_analysis(has_references=True)
    print(f"长度: {len(style_prompt)} 字符")
    print(f"预览: {style_prompt[:200]}...\n")

    print("2. 图像生成提示词:")
    gen_prompt, neg_prompt = PromptBuilder.build_image_generation(
        style_prompt="柔和自然光，户外场景",
        scene_hint="海滩",
        single_reference=False
    )
    print(f"正向提示词长度: {len(gen_prompt)} 字符")
    print(f"负向提示词长度: {len(neg_prompt)} 字符")
    print(f"负向提示词预览: {neg_prompt[:200]}...\n")
