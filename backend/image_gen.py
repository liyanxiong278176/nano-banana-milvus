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
                "These are our top-selling fashion product photos.\n\n"
                "Analyze their common visual style in these dimensions:\n"
                "1. Scene / background setting\n"
                "2. Lighting and color tone\n"
                "3. Model pose and framing\n"
                "4. Overall mood and aesthetic\n\n"
                "Then, based on this analysis, write ONE concise image generation prompt "
                "(under 100 words) that captures this style. The prompt should describe "
                "a scene for a model wearing a new clothing item. "
                "Output ONLY the prompt, nothing else."
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
            f"I have a new clothing product (Image 1: flat-lay photo) and a reference "
            f"promotional photo from our bestselling catalog (Image 2).\n\n"
            f"Generate a professional e-commerce promotional photograph of a female model "
            f"wearing the clothing from Image 1.\n\n"
            f"Style guidance: {style_prompt}\n\n"
        )

        if scene_hint:
            gen_prompt += f"Scene hint: {scene_hint}\n\n"

        gen_prompt += (
            f"Requirements:\n"
            f"- Full body shot, photorealistic, high quality\n"
            f"- The clothing should match Image 1 exactly\n"
            f"- The photo style and mood should match Image 2\n"
            f"- Natural fit between clothing and model body\n"
            f"- No visible tags or labels\n"
            f"- Sharp details, professional lighting"
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
