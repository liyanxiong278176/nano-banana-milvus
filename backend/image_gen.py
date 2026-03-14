"""
图像生成模块 - 使用 Qwen3.5 + Nano Banana 2
"""
from typing import List, Dict, Tuple
from PIL import Image

from openai import OpenAI

from config import OPENROUTER_API_KEY, LLM_MODEL, IMAGE_GEN_MODEL
from config import DEFAULT_ASPECT_RATIO, DEFAULT_IMAGE_SIZE
from utils import image_to_uri, extract_images


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


if __name__ == "__main__":
    print("图像生成模块")
    print("测试需要准备图片数据")
