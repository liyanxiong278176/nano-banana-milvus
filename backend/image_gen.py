"""
图像生成模块 - 使用 Qwen3.5 + Nano Banana 2
"""
import re
import json
from typing import List, Dict, Tuple
from PIL import Image

from openai import OpenAI
import httpx

from config import (
    OPENROUTER_API_KEY, LLM_MODEL, IMAGE_GEN_MODEL,
    DEFAULT_ASPECT_RATIO, DEFAULT_IMAGE_SIZE, ENABLE_CACHE
)
from utils import image_to_uri, extract_images, get_cache_key, save_to_cache, load_from_cache

# 超时配置 (秒)
API_TIMEOUT = 300  # 5 分钟 - LLM 和图片生成请求超时时间


# 常量定义
MAX_REFERENCE_IMAGES_FOR_SCORING = 2  # 质量评估时最多使用的参考图数量


class ImageQualityJudge:
    """图片质量裁判官 - 使用多模态 LLM 评分"""

    def __init__(self, model: str = None):
        """
        初始化裁判官

        Args:
            model: 指定裁判模��，None 则使用配置的 LLM_MODEL
        """
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=httpx.Timeout(API_TIMEOUT, connect=60)
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
            for ref in reference_images[:MAX_REFERENCE_IMAGES_FOR_SCORING]:
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

            # 尝试多种方式解析 JSON
            scores = None

            # 方法1: 直接解析整个响应
            try:
                scores = json.loads(content_text.strip())
            except json.JSONDecodeError:
                pass

            # 方法2: 使用正则匹配 JSON 对象
            if scores is None:
                # 匹配可能嵌套的 JSON
                json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', content_text, re.DOTALL)
                if json_match:
                    try:
                        scores = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            # 成功解析，计算平均分
            if scores and isinstance(scores, dict):
                avg_score = sum(scores.values()) / len(scores)
                scores['average'] = round(avg_score, 2)
                scores['is_fallback'] = False
                return scores

        except Exception as e:
            print(f"评分失败: {e}")

        # 默认分数（标识为后备值）
        return {
            "clothing_accuracy": 3,
            "pose_naturalness": 3,
            "scene_quality": 3,
            "lighting_quality": 3,
            "commercial_value": 3,
            "average": 3.0,
            "is_fallback": True  # 标识这是默认值
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
            base_url="https://openrouter.ai/api/v1",
            timeout=httpx.Timeout(API_TIMEOUT, connect=60)
        )

    def analyze_style_with_llm(
        self,
        reference_images: List[Image.Image]
    ) -> str:
        """
        使用 LLM 分析爆款图片风格（综合分析）

        【第一性原则】只提取拍摄风格（场景、光线、构图），绝不描述服装款式

        【缓存机制】根据参考图片内容生成缓存键，相同图片组合直接返回缓存结果

        Args:
            reference_images: 参考爆款图片列表

        Returns:
            风格描述 prompt（仅包含拍摄风格，不包含服装款式）
        """
        print(f"\n使用 {LLM_MODEL} 分析爆款拍摄风格...")
        print(f"  传入参考图数量: {len(reference_images)}")

        if not reference_images:
            print("  警告: 没有参考图！")
            return "专业电商宣传照，简洁背景，柔和光线"

        # ==================== 【新增】缓存逻辑 ====================
        if ENABLE_CACHE:
            # 生成缓存键：基于所有参考图的 base64 拼接
            cache_content = "".join([image_to_uri(img) for img in reference_images])
            cache_key = get_cache_key("style_analysis", cache_content)

            # 尝试从缓存加载
            cached_result = load_from_cache(cache_key)
            if cached_result is not None:
                print("  ✓ 使用缓存的风格分析结果")
                return cached_result

        # 对每张图单独分析风格
        individual_analyses = []
        for i, ref_img in enumerate(reference_images):
            print(f"  分析第 {i+1}/{len(reference_images)} 张参考图...")

            content = [
                {"type": "image_url", "image_url": {"url": image_to_uri(ref_img)}},
                {"type": "text", "text": (
                    "这是一张时尚产品宣传照片。\n\n"
                    "【核心原则】你只需要分析**摄影拍摄风格**，绝对不要描述模特身上的服装款式！\n\n"
                    "请仅从以下维度分析这张图片的**拍摄风格**：\n"
                    "1. 场景/背景设置（如：纯白背景、自然户外场景、极简室内等）\n"
                    "2. 光线和色调（如：柔和自然光、侧光、暖色调、冷色调等）\n"
                    "3. 模特姿势和构图（如：全身站姿、坐姿、动态抓拍等）\n"
                    "4. 整体氛围和美学风格（如：极简主义、商务专业、清新自然等）\n\n"
                    "【严格禁止】不要描述服装的款式、颜色、材质、图案！\n\n"
                    "请用一句话（50字以内）总结这张图片的**拍摄风格**。\n"
                    "只输出拍摄风格描述，不要其他内容。"
                )},
            ]

            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=200,
                    temperature=0.7,
                )
                analysis = response.choices[0].message.content.strip()
                individual_analyses.append(analysis)
                print(f"    图{i+1}拍摄风格: {analysis}")
            except Exception as e:
                print(f"    图{i+1}分析失败: {e}")
                individual_analyses.append(f"参考图{i+1}")

        # 基于所有图片的综合分析（生成最终的风格 prompt）
        print(f"\n基于 {len(reference_images)} 张参考图进行综合分析...")

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
                "【核心原则】你只需要分析**摄影拍摄风格**，绝对不要描述服装款式！\n\n"
                "请仅从以下维度分析它们的**拍摄风格**：\n"
                "1. 场景/背景设置\n"
                "2. 光线和色调\n"
                "3. 模特姿势和构图\n"
                "4. 整体氛围和美学风格\n\n"
                "【严格禁止】不要描述服装的款式、颜色、材质、图案！\n\n"
                "基于以上分析，请写一段简洁的图像生成提示词（100字以内），"
                "描述**拍摄场景、光线、构图风格**（不涉及具体服装款式）。\n"
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
        print(f"\n综合拍摄风格分析结果:\n{style_prompt}\n")

        # 构建返回结果
        result = {
            "combined_style": style_prompt,
            "individual_analyses": individual_analyses
        }

        # ==================== 【新增】保存到缓存 ====================
        if ENABLE_CACHE:
            save_to_cache(cache_key, result)

        # 返回综合风格和每张图的分析
        return result

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
        reference_images: List[Image.Image],
        style_prompt: str,
        scene_hint: str = "",
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        image_size: str = DEFAULT_IMAGE_SIZE
    ) -> List[Image.Image]:
        """
        使用图像生成模型生成宣传图

        【第一性原则】模特身上穿的必须是用户上传的服装，参考图只提供拍摄风格

        Args:
            new_product_image: 新品平铺图（必须100%还原的服装）
            reference_images: 参考爆款图列表（仅用于提取拍摄风格）
            style_prompt: LLM 分析的拍摄风格描述
            scene_hint: 场景提示
            aspect_ratio: 宽高比 (3:4, 1:1, 4:1 等)
            image_size: 分辨率 (512px, 1K, 2K, 4K)

        Returns:
            生成的图片列表
        """
        ref_count = len(reference_images)
        print(f"\n使用 {IMAGE_GEN_MODEL} 生成宣传图...")
        print(f"  参考图数量: {ref_count}")
        print(f"  宽高比: {aspect_ratio}")
        print(f"  分辨率: {image_size}")

        # 构建 prompt，根据参考图数量调整描述
        if ref_count == 1:
            gen_prompt = (
                f"【第一性原则】图片1是用户上传的服装平铺图，必须100%还原这件服装！\n"
                f"图片2是参考图，仅用于参考**拍摄风格**（场景、光线、构图），不能模仿它的服装款式！\n\n"
                f"请生成一张专业的电商宣传照片，展示一位女性模特穿着图片1中的服装。\n\n"
                f"拍摄风格参考（不含服装）：{style_prompt}\n\n"
            )
        else:
            gen_prompt = f"""
【第一性原则·最高优先级】第一张图片是用户上传的服装，模特身上必须100%还原这件服装！
版型、材质、颜色、款式、细节完全一致，绝对不允许更改服装款式！

后续图片（图片2及以后）是参考图，仅用于参考**拍摄风格**（场景、光线、构图），
绝对不能模仿参考图里的服装款式、颜色、材质！

请生成一张专业的电商宣传照片，展示一位女性模特穿着**第一张图片中的服装**。

拍摄风格参考（不含服装）：{style_prompt}

【硬性规则·违反即失败】：
1. 服装必须与第一张用户上传的原图完全一致，版型、颜色、材质、细节100%还原
2. 后续参考图仅提供拍摄风格、光线、场景构图参考，绝对不能使用参考图里的服装款式
3. 全身照，照片级真实感，8K高清
4. 服装与模特身体贴合自然，无变形、无穿模
5. 无可见标签或吊牌，细节清晰，专业商业布光
"""

        if scene_hint:
            gen_prompt += f"\n场景提示：{scene_hint}\n"

        # 【第一性原则】只传用户上传的服装图，不传参考图！
        # 参考图的拍摄风格已经通过 LLM 分析提取到 style_prompt 中了
        # 如果传参考图给生成模型，模型会同时参考服装款式，导致生成的服装变化
        gen_content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(new_product_image)}},
            {"type": "text", "text": gen_prompt}
        ]

        # 根据模型类型使用正确的 modalities
        modalities = self._get_model_modalities(IMAGE_GEN_MODEL)

        # ==================== 【调试】详细日志 ====================
        print(f"\n[图像生成] 开始调用 API...")
        print(f"  模型: {IMAGE_GEN_MODEL}")
        print(f"  Modalities: {modalities}")
        print(f"  宽高比: {aspect_ratio}")
        print(f"  分辨率: {image_size}")
        print(f"  Prompt 长度: {len(gen_prompt)} 字符")
        print(f"  API Key: {OPENROUTER_API_KEY[:10]}...{OPENROUTER_API_KEY[-4:]}")
        print(f"  超时设置: {API_TIMEOUT} 秒")

        import time
        start_time = time.time()

        try:
            print(f"\n[图像生成] 发送请求到 OpenRouter...")
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
            elapsed = time.time() - start_time
            print(f"\n[图像生成] API 响应成功! 耗时: {elapsed:.2f} 秒")
            print(f"  响应类型: {type(gen_response)}")
            print(f"  Choices 数量: {len(gen_response.choices)}")

            # 检查响应内容
            if gen_response.choices:
                choice = gen_response.choices[0]
                print(f"  Choice[0] role: {choice.message.role}")
                print(f"  Choice[0] content 类型: {type(choice.message.content)}")
                print(f"  Choice[0] content 长度: {len(str(choice.message.content))}")

                # 检查是否有额外字段
                if hasattr(choice.message, '__dict__'):
                    print(f"  Choice[0] 所有字段: {list(choice.message.__dict__.keys())}")

            # 提取生成的图片
            print(f"\n[图像生成] 提取生成的图片...")
            generated_images = extract_images(gen_response)
            print(f"  提取到 {len(generated_images)} 张图片")

            # 检查是否有文本响应
            text_content = gen_response.choices[0].message.content
            if text_content:
                print(f"\n  模型文本响应: {text_content[:200]}...")

            print(f"\n[图像生成] 完成!")
            return generated_images

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[图像生成] API 调用失败! 耗时: {elapsed:.2f} 秒")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误信息: {str(e)}")

            # 详细错误信息
            import traceback
            print(f"\n[详细堆栈]:")
            traceback.print_exc()

            # 重新抛出异常
            raise

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
        style_analysis = self.analyze_style_with_llm(reference_images)
        style_prompt = style_analysis["combined_style"]

        # 2. 生成图片
        generated = self.generate_promotional_photo(
            new_product_image=new_product_image,
            reference_images=reference_images,
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
        style_analysis = self.analyze_style_with_llm(reference_images)
        style_prompt = style_analysis["combined_style"]

        # 2. 生成单张图片
        print(f"\n生成宣传图...")
        generated = self.generate_promotional_photo(
            new_product_image=new_product_image,
            reference_images=reference_images,
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
            print(f"\n[WARNING] {reason}")
        else:
            print(f"\n[OK] 质量合格")

        return {
            'style_prompt': style_prompt,
            'style_analysis': style_analysis,
            'individual_analyses': style_analysis.get('individual_analyses', []),
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
        style_analysis = self.analyze_style_with_llm(reference_images)
        style_prompt = style_analysis["combined_style"]

        # 2. 生成图片
        generated = self.generate_promotional_photo(
            new_product_image=new_product_image,
            reference_images=reference_images,
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
