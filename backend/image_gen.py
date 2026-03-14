"""
图像生成模块 - 使用 Qwen3.5 + Nano Banana 2
"""
import re
import json
from typing import List, Dict, Tuple
from PIL import Image

from openai import OpenAI

from config import OPENROUTER_API_KEY, LLM_MODEL, IMAGE_GEN_MODEL
from config import DEFAULT_ASPECT_RATIO, DEFAULT_IMAGE_SIZE
from utils import image_to_uri, extract_images


class ImageQualityJudge:
    """图片质量裁判官 - 使用多模态 LLM 评分"""

    def __init__(self, model: str = None):
        """
        初始化裁判官

        Args:
            model: 指定裁判模型，None 则使用配置的 LLM_MODEL
        """
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model or LLM_MODEL

    def score_image_quality(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        reference_images: List[Image.Image] = None
    ) -> Dict[str, any]:
        """
        对生成的图片进行多维度评分

        Args:
            generated_image: 生成的宣传图
            original_image: 原始平铺图
            reference_images: 参考爆款图（可选）

        Returns:
            评分结果字典，包含各维度得分和总分
        """
        content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(original_image)}},
            {"type": "image_url", "image_url": {"url": image_to_uri(generated_image)}},
        ]

        # 如果有参考图，也加入评估
        if reference_images:
            for ref in reference_images[:2]:  # 最多2张参考图
                content.append({"type": "image_url", "image_url": {"url": image_to_uri(ref)}})

        score_prompt = self._build_score_prompt(len(reference_images) if reference_images else 0)
        content.append({"type": "text", "text": score_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=500,
                temperature=0.3,  # 低温度保证评分稳定
            )

            content_text = response.choices[0].message.content

            # 提取 JSON
            json_match = re.search(r'\{[^}]+\}', content_text)
            if json_match:
                try:
                    scores = json.loads(json_match.group())
                    # 计算平均分
                    if isinstance(scores, dict):
                        avg_score = sum(scores.values()) / len(scores)
                        scores['average'] = round(avg_score, 2)
                    return scores
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            print(f"评分失败: {e}")

        # 默认分数
        return {
            "clothing_accuracy": 3,
            "pose_naturalness": 3,
            "scene_quality": 3,
            "lighting_quality": 3,
            "commercial_value": 3,
            "average": 3.0
        }

    def _build_score_prompt(self, has_references: int = 0) -> str:
        """构建评分提示词"""
        ref_text = ""
        if has_references > 0:
            ref_text = f"图片3及以后是参考爆款图。\n"

        return f"""你是一位专业的电商照片质量评估专家。

**图片说明**：
- 图片1：原始产品平铺图
- 图片2：AI 生成的宣传图
- {ref_text}

请评估生成的宣传图（图片2）的质量，对以下维度进行评分（1-5分）：

1. **clothing_accuracy**（服装准确性）：图片2中的服装与图片1中的原始产品匹配度
2. **pose_naturalness**（姿势自然度）：模特的姿势和服装合身度是否自然
3. **scene_quality**（场景质量）：背景/场景是否专业
4. **lighting_quality**（布光质量）：光线质量如何
5. **commercial_value**（商业价值）：总体来说，这张图片适合电商使用吗

**评分标准**：
- 5分：优秀，完全符合要求
- 4分：良好，基本符合要求
- 3分：一般，有可改进之处
- 2分：较差，需要明显改进
- 1分：很差，不符合要求

只输出JSON格式，例如：
{{"clothing_accuracy": 4, "pose_naturalness": 3, "scene_quality": 5, "lighting_quality": 4, "commercial_value": 4}}
"""

    def should_regenerate(
        self,
        scores: Dict[str, any],
        threshold: float = 3.5
    ) -> Tuple[bool, str]:
        """
        判断是否需要重新生成

        Args:
            scores: 评分结果
            threshold: 最低及格分数

        Returns:
            (是否需要重新生成, 原因说明)
        """
        avg_score = scores.get('average', 0)

        if avg_score < threshold:
            # 找出得分最低的维度
            lowest_dim = min(
                [(k, v) for k, v in scores.items() if k != 'average'],
                key=lambda x: x[1]
            )
            dim_names = {
                'clothing_accuracy': '服装准确性',
                'pose_naturalness': '姿势自然度',
                'scene_quality': '场景质量',
                'lighting_quality': '布光质量',
                'commercial_value': '商业价值'
            }
            return True, f"{dim_names.get(lowest_dim[0], lowest_dim[0])}得分过低({lowest_dim[1]})，建议重新生成"

        # 检查关键维度
        if scores.get('clothing_accuracy', 5) < 4:
            return True, f"服装准确性不足({scores['clothing_accuracy']})，服装与原图不匹配"

        return False, "质量合格"

    def batch_score(
        self,
        generated_images: List[Image.Image],
        original_image: Image.Image,
        reference_images: List[Image.Image] = None
    ) -> List[Dict[str, any]]:
        """
        批量评分多张图片

        Returns:
            评分结果列表，按得分从高到低排序
        """
        results = []
        for i, img in enumerate(generated_images):
            scores = self.score_image_quality(img, original_image, reference_images)
            scores['index'] = i
            results.append(scores)

        # 按平均分排序
        results.sort(key=lambda x: x.get('average', 0), reverse=True)
        return results


class ImageGenerator:
    """AI 图像生成器"""

    def __init__(self):
        """初始化生成器"""
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    def analyze_style_with_llm(
        self,
        reference_images: List[Image.Image]
    ) -> str:
        """
        使用 Qwen3.5 分析爆款图片风格

        Args:
            reference_images: 参考爆款图片列表

        Returns:
            风格描述 prompt
        """
        print("\n使用 Qwen3.5 分析爆款风格...")

        # 构建多模态输入
        content = [
            {
                "type": "image_url",
                "image_url": {"url": image_to_uri(img)}
            }
            for img in reference_images
        ]

        content.append({
            "type": "text",
            "text": (
                "这些是我们店铺最畅销的时尚产品宣传照片。\n\n"
                "请分析它们的共同视觉风格，从以下维度：\n"
                "1. 场景/背景设置\n"
                "2. 光线和色调\n"
                "3. 模特姿势和构图\n"
                "4. 整体氛围和美学风格\n\n"
                "基于以上分析，请写一段简洁的图像生成提示词（100字以内），"
                "描述模特穿着新服装的场景。\n"
                "只输出提示词，不要输出其他内容。"
            ),
        })

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.7,
        )

        style_prompt = response.choices[0].message.content.strip()
        print(f"\n风格分析结果:\n{style_prompt}\n")

        return style_prompt

    def _get_model_modalities(self, model: str) -> List[str]:
        """
        根据模型类型返回正确的 modalities

        Args:
            model: 模型 ID

        Returns:
            modalities 列表
        """
        # Gemini 模型支持文字+图像输出
        if "google" in model or "gemini" in model.lower():
            return ["text", "image"]
        # Flux, Sourceful, ByteDance 等仅输出图像
        else:
            return ["image"]

    def generate_promotional_photo(
        self,
        new_product_image: Image.Image,
        reference_image: Image.Image,
        style_prompt: str,
        scene_hint: str = "",
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        image_size: str = DEFAULT_IMAGE_SIZE
    ) -> List[Image.Image]:
        """
        使用图像生成模型生成宣传图

        Args:
            new_product_image: 新品平铺图
            reference_image: 参考爆款图
            style_prompt: LLM 分析的风格描述
            scene_hint: 场景提示
            aspect_ratio: 宽高比 (3:4, 1:1, 4:1 等)
            image_size: 分辨率 (512px, 1K, 2K, 4K)

        Returns:
            生成的图片列表
        """
        print(f"\n使用 {IMAGE_GEN_MODEL} 生成宣传图...")
        print(f"  宽高比: {aspect_ratio}")
        print(f"  分辨率: {image_size}")

        gen_prompt = (
            f"图片1是一件新的服装产品（平铺照片），图片2是我们畅销目录中的参考宣传照。\n\n"
            f"请生成一张专业的电商宣传照片，展示一位女性模特穿着图片1中的服装。\n\n"
            f"风格指导：{style_prompt}\n\n"
        )

        if scene_hint:
            gen_prompt += f"场景提示：{scene_hint}\n\n"

        gen_prompt += (
            f"要求：\n"
            f"- 全身照，照片级真实感，高质量\n"
            f"- 服装必须与图片1完全一致\n"
            f"- 照片风格和氛围应与图片2匹配\n"
            f"- 服装与模特身体的贴合自然\n"
            f"- 无可见标签或吊牌\n"
            f"- 细节清晰，专业布光"
        )

        gen_content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(new_product_image)}},
            {"type": "image_url", "image_url": {"url": image_to_uri(reference_image)}},
            {"type": "text", "text": gen_prompt},
        ]

        # 根据模型类型使用正确的 modalities
        modalities = self._get_model_modalities(IMAGE_GEN_MODEL)

        gen_response = self.client.chat.completions.create(
            model=IMAGE_GEN_MODEL,
            messages=[{"role": "user", "content": gen_content}],
            extra_body={
                "modalities": modalities,
                "image_config": {
                    "aspect_ratio": aspect_ratio,
                    "image_size": image_size
                },
            },
        )

        # 提取生成的图片
        generated_images = extract_images(gen_response)

        # 检查是否有文本响应
        text_content = gen_response.choices[0].message.content
        if text_content:
            print(f"\n模型响应: {text_content[:200]}...")

        return generated_images

    def process_single_product(
        self,
        new_product_image: Image.Image,
        reference_images: List[Image.Image],
        scene_hint: str = ""
    ) -> tuple:
        """
        完整处理单个新品: 风格分析 + 图像生成

        Args:
            new_product_image: 新品图片
            reference_images: 参考爆款图片列表
            scene_hint: 场景提示

        Returns:
            (风格描述, 生成图片列表)
        """
        # 1. 分析风格
        style_prompt = self.analyze_style_with_llm(reference_images)

        # 2. 生成图片
        generated = self.generate_promotional_photo(
            new_product_image=new_product_image,
            reference_image=reference_images[0],
            style_prompt=style_prompt,
            scene_hint=scene_hint
        )

        return style_prompt, generated

    def process_with_quality_check(
        self,
        new_product_image: Image.Image,
        reference_images: List[Image.Image],
        scene_hint: str = "",
        min_score: float = 3.5,
        judge_model: str = None
    ) -> Dict:
        """
        生成图片并进行质量评估

        Args:
            new_product_image: 新品图片
            reference_images: 参考爆款图片列表
            scene_hint: 场景提示
            min_score: 最低及格分数
            judge_model: 裁判模型，None 则使用默认 LLM_MODEL

        Returns:
            包含风格描述、图片和评分的字典
        """
        print(f"\n=== 启用质量评估模式 ===")
        print(f"将生成 1 张图片并进行 AI 质量评分")

        # 1. 分析风格
        style_prompt = self.analyze_style_with_llm(reference_images)

        # 2. 生成单张图片
        print(f"\n生成宣传图...")
        generated = self.generate_promotional_photo(
            new_product_image=new_product_image,
            reference_image=reference_images[0],
            style_prompt=style_prompt,
            scene_hint=scene_hint
        )

        if not generated or not generated[0]:
            return {
                'style_prompt': style_prompt,
                'best_image': None,
                'all_images': [],
                'scores': [],
                'error': '图片生成失败'
            }

        generated_image = generated[0]

        # 3. 使用裁判官评分
        judge = ImageQualityJudge(model=judge_model)
        print(f"\n使用裁判官 {judge.model} 评估图片质量...")

        score_result = judge.score_image_quality(
            generated_image=generated_image,
            original_image=new_product_image,
            reference_images=reference_images
        )

        # 4. 输出评分结果
        print("\n=== 质量评分结果 ===")
        print(f"平均分: {score_result['average']:.2f}/5")
        print(f"  服装准确性: {score_result['clothing_accuracy']}/5")
        print(f"  姿势自然度: {score_result['pose_naturalness']}/5")
        print(f"  场景质量: {score_result['scene_quality']}/5")
        print(f"  布光质量: {score_result['lighting_quality']}/5")
        print(f"  商业价值: {score_result['commercial_value']}/5")

        # 5. 判断质量是否合格
        should_regenerate, reason = judge.should_regenerate(score_result, threshold=min_score)

        if should_regenerate:
            print(f"\n⚠️ 警告: {reason}")
        else:
            print(f"\n✅ 质量合格")

        return {
            'style_prompt': style_prompt,
            'best_image': generated_image,
            'all_images': [generated_image],
            'best_score': score_result,
            'all_scores': [score_result],
            'should_regenerate': should_regenerate,
            'regenerate_reason': reason if should_regenerate else None
        }

    def process_single_product_with_judge(
        self,
        new_product_image: Image.Image,
        reference_images: List[Image.Image],
        scene_hint: str = "",
        enable_judge: bool = True,
        judge_model: str = None
    ) -> tuple:
        """
        处理单个新品（带可选的质量评估）

        Args:
            new_product_image: 新品图片
            reference_images: 参考爆款图片列表
            scene_hint: 场景提示
            enable_judge: 是否启用质量评估
            judge_model: 裁判模型

        Returns:
            (风格描述, 生成图片列表, 评分结果字典)
        """
        # 1. 分析风格
        style_prompt = self.analyze_style_with_llm(reference_images)

        # 2. 生成图片
        generated = self.generate_promotional_photo(
            new_product_image=new_product_image,
            reference_image=reference_images[0],
            style_prompt=style_prompt,
            scene_hint=scene_hint
        )

        # 3. 可选的质量评估
        judge_result = None
        if enable_judge and generated:
            judge = ImageQualityJudge(model=judge_model)
            judge_result = judge.score_image_quality(
                generated_image=generated[0] if generated else None,
                original_image=new_product_image,
                reference_images=reference_images
            )
            print(f"\n质量评分: {judge_result.get('average', 0):.2f}/5")

        return style_prompt, generated, judge_result


if __name__ == "__main__":
    print("图像生成模块")
    print("测试需要准备图片数据")
