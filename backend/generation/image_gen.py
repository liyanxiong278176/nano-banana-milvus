"""
图像生成模块 - 使用 LangChain 改进版

【改进点】
1. 使用 LangChain LLMClient 统一 LLM 调用
2. 使用 with_structured_output 进行质量评估
3. 使用 PromptTemplateManager 管理提示词
4. 保留向后兼容性
"""
import time
import logging
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image

# 配置日志
logger = logging.getLogger(__name__)

# 图像生成仍需直接使用 OpenAI（特殊参数）
from openai import OpenAI
import httpx

from config import (
    OPENROUTER_API_KEY, LLM_MODEL, IMAGE_GEN_MODEL,
    DEFAULT_ASPECT_RATIO, DEFAULT_IMAGE_SIZE, ENABLE_CACHE,
    MAX_REFERENCE_IMAGES_FOR_SCORING
)
from utils.core import image_to_uri, extract_images, get_cache_key, save_to_cache, load_from_cache

# ==================== LangChain 集成 ====================
try:
    from llm import (
        get_llm_client,
        get_light_client,
        get_prompt_manager,
        QualityScoreSchema
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logging.warning("LangChain 模块不可用")
    LANGCHAIN_AVAILABLE = False

# ==================== 常量定义 ====================

# 超时配置
API_TIMEOUT = 300

# 默认分数（LLM 调用失败时的 fallback 值）
FALLBACK_SCORE = 6

# 模型配置
class ModelConfig:
    """模型配置集中管理"""

    @staticmethod
    def get_modalities(model: str) -> List[str]:
        """根据模型类型返回 modalities"""
        if "google" in model or "gemini" in model.lower():
            return ["text", "image"]
        return ["image"]


# ==================== 图片质量评估（LangChain 版本）====================

class ImageQualityJudge:
    """
    图片质量裁判官 - 使用 LangChain 结构化输出

    【职责】
    - 对生成的图片进行多维度质量评分
    - 判断是否需要重新生成
    - 支持批量评分
    """

    # 类常量
    FALLBACK_SCORE = 6
    DEFAULT_THRESHOLD = 7.0

    def __init__(self, model: str = None):
        """
        初始化裁判官

        Args:
            model: 指定裁判模型，None 则使用轻量级模型
        """
        if LANGCHAIN_AVAILABLE:
            self.client = get_light_client() if model is None else get_llm_client(model)
            self.prompt_manager = get_prompt_manager()
            self.use_langchain = True
            self.model = self.client.model
        else:
            # 向后兼容
            self.client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                timeout=httpx.Timeout(API_TIMEOUT, connect=60)
            )
            self.model = model or LLM_MODEL
            self.use_langchain = False

        self._logger = logging.getLogger(f"{__name__}.ImageQualityJudge")

    def score_image_quality(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        reference_images: Optional[List[Image.Image]] = None
    ) -> Dict[str, Any]:
        """
        对生成的图片进行多维度评分

        Args:
            generated_image: 生成的宣传图
            original_image: 原始平铺图
            reference_images: 参考爆款图（可选）

        Returns:
            评分结果字典，包含各维度得分和总分
        """
        # 输入验证
        self._validate_images(generated_image, original_image)

        start_time = time.time()
        ref_count = len(reference_images) if reference_images else 0

        if self.use_langchain:
            return self._score_with_langchain(
                generated_image, original_image, reference_images, ref_count, start_time
            )
        else:
            return self._score_with_openai(
                generated_image, original_image, reference_images, ref_count, start_time
            )

    def _validate_images(self, generated_image: Image.Image, original_image: Image.Image):
        """验证输入图片"""
        if generated_image is None:
            raise ValueError("generated_image 不能为 None")
        if original_image is None:
            raise ValueError("original_image 不能为 None")
        if not isinstance(generated_image, Image.Image):
            raise TypeError(f"generated_image 必须是 PIL.Image，实际类型: {type(generated_image)}")
        if not isinstance(original_image, Image.Image):
            raise TypeError(f"original_image 必须是 PIL.Image，实际类型: {type(original_image)}")

    def _get_fallback_scores(self, error: str = None) -> Dict[str, Any]:
        """获取默认分数"""
        return {
            "clothing_accuracy": self.FALLBACK_SCORE,
            "pose_naturalness": self.FALLBACK_SCORE,
            "scene_quality": self.FALLBACK_SCORE,
            "lighting_quality": self.FALLBACK_SCORE,
            "commercial_value": self.FALLBACK_SCORE,
            "average": float(self.FALLBACK_SCORE),
            "is_fallback": True,
            "error": error
        }

    def _score_with_langchain(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        reference_images: List[Image.Image],
        ref_count: int,
        start_time: float
    ) -> Dict[str, Any]:
        """使用 LangChain 结构化输出评分"""
        images = [original_image, generated_image]
        if reference_images:
            images.extend(reference_images[:MAX_REFERENCE_IMAGES_FOR_SCORING])

        prompt = self.prompt_manager.get_quality_judge_prompt(
            has_references=bool(reference_images)
        )

        try:
            scores_dict = self.client.invoke_structured(
                text=prompt,
                schema=QualityScoreSchema,
                images=images,
                temperature=0.3
            )

            # model_dump() 不包含 @property，需要手动计算 average
            scores_dict['average'] = round(sum([
                scores_dict.get('clothing_accuracy', 0),
                scores_dict.get('pose_naturalness', 0),
                scores_dict.get('scene_quality', 0),
                scores_dict.get('lighting_quality', 0),
                scores_dict.get('commercial_value', 0)
            ]) / 5, 2)

            execution_time = time.time() - start_time
            scores_dict['execution_time'] = execution_time
            scores_dict['is_fallback'] = False

            self._logger.info(
                f"质量评估完成: 平均分 {scores_dict['average']}/10, "
                f"耗时 {execution_time:.2f}s"
            )

            return scores_dict

        except (ValueError, TypeError, AttributeError) as e:
            # 结构化输出相关的错误
            execution_time = time.time() - start_time
            self._logger.error(f"结构化输��解析失败: {e}，耗时 {execution_time:.2f}s")
            return self._get_fallback_scores(error=str(e))

        except Exception as e:
            # 其他未预期的错误
            execution_time = time.time() - start_time
            self._logger.error(f"评分失败: {e}，耗时 {execution_time:.2f}s", exc_info=True)
            return self._get_fallback_scores(error=str(e))

    def _score_with_openai(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        reference_images: List[Image.Image],
        ref_count: int,
        start_time: float
    ) -> Dict[str, Any]:
        """向后兼容：使用 OpenAI 客户端评分"""
        import json
        import re

        content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(original_image)}},
            {"type": "image_url", "image_url": {"url": image_to_uri(generated_image)}},
        ]

        if reference_images:
            for ref in reference_images[:MAX_REFERENCE_IMAGES_FOR_SCORING]:
                content.append({"type": "image_url", "image_url": {"url": image_to_uri(ref)}})

        prompt = self._get_builtin_prompt(ref_count)
        content.append({"type": "text", "text": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=500,
                temperature=0.3,
            )

            content_text = response.choices[0].message.content

            # 尝试解析 JSON
            scores = None
            try:
                scores = json.loads(content_text.strip())
            except json.JSONDecodeError:
                # 使用正则匹配 JSON 对象
                json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', content_text, re.DOTALL)
                if json_match:
                    try:
                        scores = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            if scores and isinstance(scores, dict):
                avg_score = sum(scores.values()) / len(scores)
                scores['average'] = round(avg_score, 2)
                scores['is_fallback'] = False
                execution_time = time.time() - start_time
                self._logger.info(f"质量评估完成: 平均分 {scores['average']}/10")
                return scores

        except (openai.APIError, httpx.HTTPError) as e:
            execution_time = time.time() - start_time
            self._logger.error(f"API 调用失败: {e}，耗时 {execution_time:.2f}s")

        except (json.JSONDecodeError, ValueError) as e:
            execution_time = time.time() - start_time
            self._logger.error(f"JSON 解析失败: {e}，耗时 {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time
            self._logger.error(f"未预期的错误: {e}，耗时 {execution_time:.2f}s", exc_info=True)

        # 返回默认分数
        return self._get_fallback_scores()

    def _get_builtin_prompt(self, ref_count: int) -> str:
        """内置提示词（向后兼容）"""
        ref_text = ""
        if ref_count > 0:
            ref_text = "图片3及以后是参考爆款图。\n"

        return f"""你是一位专业的电商照片质量评估专家。

**图片说明**：
- 图片1：原始产品平铺图
- 图片2：AI 生成的宣传图
- {ref_text}

请评估生成的宣传图（图片2）的质量，对以下维度进行评分（1-10分）：

1. **clothing_accuracy**（服装准确性）
2. **pose_naturalness**（姿势自然度）
3. **scene_quality**（场景质量）
4. **lighting_quality**（布光质量）
5. **commercial_value**（商业价值）

只输出 JSON 格式：
{{"clothing_accuracy": 8, "pose_naturalness": 7, "scene_quality": 9, "lighting_quality": 8, "commercial_value": 8}}
"""

    def should_regenerate(
        self,
        scores: Dict[str, Any],
        threshold: float = None
    ) -> Tuple[bool, str]:
        """
        判断是否需要重新生成

        Args:
            scores: 评分结果
            threshold: 最低及格分数，None 则使用默认值

        Returns:
            (是否需要重新生成, 原因说明)
        """
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLD

        avg_score = scores.get('average', 0)

        if avg_score < threshold:
            # 找出得分最低的维度
            valid_items = [
                (k, v) for k, v in scores.items()
                if k not in ['average', 'is_fallback', 'error', 'index', 'execution_time']
            ]
            if not valid_items:
                return True, f"平均分过低({avg_score:.1f})，建议重新生成"

            lowest_dim = min(valid_items, key=lambda x: x[1])
            dim_names = {
                'clothing_accuracy': '服装准确性',
                'pose_naturalness': '姿势自然度',
                'scene_quality': '场景质量',
                'lighting_quality': '布光质量',
                'commercial_value': '商业价值'
            }
            return True, f"{dim_names.get(lowest_dim[0], lowest_dim[0])}得分过低({lowest_dim[1]})，建议重新生成"

        # 检查关键维度
        clothing_score = scores.get('clothing_accuracy', 10)
        if clothing_score < 8:
            return True, f"服装准确性不足({clothing_score})"

        return False, "质量合格"

    def batch_score(
        self,
        generated_images: List[Image.Image],
        original_image: Image.Image,
        reference_images: Optional[List[Image.Image]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量评分多张图片

        Args:
            generated_images: 生成的图片列表
            original_image: 原始平铺图
            reference_images: 参考爆款图（可选）

        Returns:
            评分结果列表，按得分从高到低排序
        """
        if not generated_images:
            self._logger.warning("batch_score 收到空列表")
            return []

        results = []
        for i, img in enumerate(generated_images):
            scores = self.score_image_quality(img, original_image, reference_images)
            scores['index'] = i
            results.append(scores)  # 修复: 原来是 results.append(results)
        results.sort(key=lambda x: x.get('average', 0), reverse=True)
        return results


# ==================== 图像生成器（LangChain 改进版）====================

class ImageGenerator:
    """AI 图像生成器"""

    # 类常量
    DEFAULT_STYLE = "专业电商宣传照，简洁背景，柔和光线"

    def __init__(self):
        """初始化生成器"""
        # 图像生成需要直接使用 OpenAI（特殊参数）
        self.gen_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=httpx.Timeout(API_TIMEOUT, connect=60)
        )

        # LangChain 客户端用于风格分析
        if LANGCHAIN_AVAILABLE:
            self.llm_client = get_llm_client()
            self.prompt_manager = get_prompt_manager()
        else:
            self.llm_client = None
            self.prompt_manager = None

        self._logger = logging.getLogger(f"{__name__}.ImageGenerator")

    def analyze_style_with_llm(
        self,
        reference_images: List[Image.Image]
    ) -> Dict[str, Any]:
        """
        使用 LLM 分析爆款图片风格

        Args:
            reference_images: 参考爆款图片列表

        Returns:
            风格分析结果
        """
        # 输入验证
        if not reference_images:
            self._logger.warning("没有参考图，使用默认风格模板")
            return {
                "combined_style": self.DEFAULT_STYLE,
                "individual_analyses": []
            }

        self._logger.info(
            f"使用 {LLM_MODEL} 分析爆款拍摄风格，"
            f"参考图数量: {len(reference_images)}, "
            f"缓存: {'启用' if ENABLE_CACHE else '禁用'}"
        )

        # 缓存逻辑（对图片排序以确保缓存键一致）
        cache_key = None
        if ENABLE_CACHE:
            sorted_images = sorted(reference_images, key=lambda img: (img.size, img.mode))
            cache_content = "".join([image_to_uri(img) for img in sorted_images])
            cache_key = get_cache_key("style_analysis", cache_content)

            cached_result = load_from_cache(cache_key)
            if cached_result is not None:
                self._logger.info("缓存命中，跳过 LLM 分析")
                return cached_result
            else:
                self._logger.debug("缓存未命中，执行 LLM 风格分析")

        # 风格分析
        if self.llm_client and self.prompt_manager:
            return self._analyze_with_langchain(reference_images, cache_key)
        else:
            return self._analyze_with_openai(reference_images, cache_key)

    def _analyze_with_langchain(
        self,
        reference_images: List[Image.Image],
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """使用 LangChain 分析风格"""
        individual_analyses = []

        # 单图分析
        single_prompt = self.prompt_manager.get_style_analysis_prompt(has_references=False)

        for i, ref_img in enumerate(reference_images):
            self._logger.debug(f"分析第 {i+1}/{len(reference_images)} 张参考图...")
            try:
                analysis = self.llm_client.invoke_with_images(
                    text=single_prompt,
                    images=[ref_img],
                    max_tokens=200,
                    temperature=0.7
                )
                individual_analyses.append(analysis)
                self._logger.debug(f"图{i+1}拍摄风格: {analysis[:50]}...")
            except Exception as e:
                self._logger.warning(f"图{i+1}分析失败: {e}")
                individual_analyses.append(f"参考图{i+1}")

        # 综合分析
        combined_prompt = self.prompt_manager.get_style_analysis_prompt(has_references=True)

        try:
            style_prompt = self.llm_client.invoke_with_images(
                text=combined_prompt,
                images=reference_images,
                max_tokens=512,
                temperature=0.7
            )
            self._logger.info(f"综合拍摄风格分析结果: {style_prompt[:100]}...")
        except Exception as e:
            self._logger.error(f"综合分析失败: {e}，使用默认风格")
            style_prompt = self.DEFAULT_STYLE

        result = {
            "combined_style": style_prompt,
            "individual_analyses": individual_analyses
        }

        # 保存缓存
        if cache_key and ENABLE_CACHE:
            save_to_cache(cache_key, result)
            self._logger.debug("缓存已保存")

        return result

    def _analyze_with_openai(
        self,
        reference_images: List[Image.Image],
        cache_key: str = None
    ) -> Dict[str, any]:
        """向后兼容：使用 OpenAI 客户端分析"""
        individual_analyses = []

        single_prompt = (
            "这是一张时尚产品宣传照片。\n\n"
            "【核心原则】你只需要分析**摄影拍摄风格**，绝对不要描述模特身上的服装款式！\n\n"
            "请用一句话（50字以内）总结这张图片的**拍摄风格**。\n"
            "只输出拍摄风格描述，不要其他内容。"
        )

        for i, ref_img in enumerate(reference_images):
            self._logger.debug(f"分析第 {i+1}/{len(reference_images)} 张参考图...")
            try:
                content = [
                    {"type": "image_url", "image_url": {"url": image_to_uri(ref_img)}},
                    {"type": "text", "text": single_prompt},
                ]
                response = self.gen_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=200,
                    temperature=0.7,
                )
                analysis = response.choices[0].message.content.strip()
                individual_analyses.append(analysis)
                self._logger.debug(f"图{i+1}拍摄风格: {analysis[:50]}...")
            except Exception as e:
                self._logger.warning(f"图{i+1}分析失败: {e}")
                individual_analyses.append(f"参考图{i+1}")

        # 综合分析
        combined_prompt = (
            "这些是我们店铺最畅销的时尚产品宣传照片。\n\n"
            "【核心原则】你只需要分析**摄影拍摄风格**，绝对不要描述服装款式！\n\n"
            "基于以上分析，请写一段简洁的图像生成提示词（150字以内），"
            "描述**拍摄场景、光线、构图风格**（不涉及具体服装款式）。\n"
            "只输出提示词，不要输出其他内容。"
        )

        try:
            content = [
                {"type": "image_url", "image_url": {"url": image_to_uri(img)}}
                for img in reference_images
            ]
            content.append({"type": "text", "text": combined_prompt})

            response = self.gen_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.7,
            )
            style_prompt = response.choices[0].message.content.strip()
            self._logger.info(f"综合拍摄风格分析结果: {style_prompt[:100]}...")
        except Exception as e:
            self._logger.error(f"综合分析失败: {e}，使用默认风格")
            style_prompt = "专业电商宣传照，简洁背景，柔和光线"

        result = {
            "combined_style": style_prompt,
            "individual_analyses": individual_analyses
        }

        if cache_key and ENABLE_CACHE:
            save_to_cache(cache_key, result)

        return result

    def _get_model_modalities(self, model: str) -> List[str]:
        """根据模型类型返回 modalities"""
        return ModelConfig.get_modalities(model)

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
        生成宣传图

        Args:
            new_product_image: 新品平铺图
            reference_images: 参考爆款图列表
            style_prompt: LLM 分析的拍摄风格描述
            scene_hint: 场景提示
            aspect_ratio: 宽高比
            image_size: 分辨率

        Returns:
            生成的图片列表
        """
        # 输入验证
        if new_product_image is None:
            raise ValueError("new_product_image 不能为 None")
        if not isinstance(new_product_image, Image.Image):
            raise TypeError(f"new_product_image 必须是 PIL.Image")

        ref_count = len(reference_images)
        self._logger.info(
            f"使用 {IMAGE_GEN_MODEL} 生成宣传图，"
            f"参考图: {ref_count}, 宽高比: {aspect_ratio}, 分辨率: {image_size}"
        )

        # 构建提示词
        gen_prompt = self._build_gen_prompt(style_prompt, scene_hint, ref_count)

        gen_content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(new_product_image)}},
            {"type": "text", "text": gen_prompt}
        ]

        modalities = self._get_model_modalities(IMAGE_GEN_MODEL)

        start_time = time.time()

        try:
            gen_response = self.gen_client.chat.completions.create(
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
            self._logger.info(f"图像生成 API 响应成功! 耗时: {elapsed:.2f} 秒")

            generated_images = extract_images(gen_response)
            self._logger.info(f"提取到 {len(generated_images)} 张图片")

            return generated_images

        except Exception as e:
            elapsed = time.time() - start_time
            self._logger.error(f"图像生成 API 调用失败! 耗时: {elapsed:.2f} 秒", exc_info=True)
            raise

    def _build_gen_prompt(self, style_prompt: str, scene_hint: str, ref_count: int) -> str:
        """构建生成提示词"""
        prompt = f"""【第一性原则·最高优先级】第一张图片是用户上传的服装，模特身上必须100%还原这件服装！
版型、材质、颜色、款式、细节完全一致，绝对不允许更改服装款式！

请生成一张专业的电商宣传照片，展示一位女性模特穿着**第一张图片中的服装**。

拍摄风格参考（不含服装）：{style_prompt}

【正向要求】：
1. 服装必须与第一张用户上传的原图完全一致
2. 版型、颜色、材质、细节100%还原
3. 全身照，照片级真实感，8K高清
4. 服装与模特身体贴合自然，无变形、无穿模
5. 无可见标签或吊牌，细节清晰
6. 专业商业布光，电商宣传照标准
"""
        if scene_hint:
            prompt += f"\n【场景提示】：{scene_hint}\n"

        return prompt

    def process_single_product(
        self,
        new_product_image: Image.Image,
        reference_images: List[Image.Image],
        scene_hint: str = ""
    ) -> tuple:
        """完整处理单个新品"""
        style_analysis = self.analyze_style_with_llm(reference_images)
        style_prompt = style_analysis["combined_style"]

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
        min_score: float = 7.0,
        judge_model: str = None
    ) -> Dict:
        """生成图片并进行质量评估"""
        self._logger.info("=== 启用质量评估模式 ===")

        style_analysis = self.analyze_style_with_llm(reference_images)
        style_prompt = style_analysis["combined_style"]

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
                'error': '图片生成失败'
            }

        generated_image = generated[0]

        judge = ImageQualityJudge(model=judge_model)
        score_result = judge.score_image_quality(
            generated_image=generated_image,
            original_image=new_product_image,
            reference_images=reference_images
        )

        self._logger.info("=== 质量评分结果 ===")
        self._logger.info(f"平均分: {score_result['average']:.2f}/10")

        should_regenerate, reason = judge.should_regenerate(score_result, threshold=min_score)

        return {
            'style_prompt': style_prompt,
            'style_analysis': style_analysis,
            'best_image': generated_image,
            'all_images': [generated_image],
            'best_score': score_result,
            'all_scores': [score_result],
            'should_regenerate': should_regenerate,
            'regenerate_reason': reason if should_regenerate else None
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("图像生成模块 - LangChain 改进版")
    logger.info(f"LangChain 可用: {LANGCHAIN_AVAILABLE}")
