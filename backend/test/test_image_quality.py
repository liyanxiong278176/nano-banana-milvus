"""
生图质量对比测试

测试目标：
对比 "本方案" (带检索参考图) 与 "基线方案" (仅新品图+prompt) 的生成图片质量差异

测试方法：
1. 对同一新品，分别用两种方案生成图片
2. 使用 VLM (视觉语言模型) 对生成图片进行多维度评分
3. 统计对比两种方案的得分差异

评分维度：
- 服装一致性 (1-5分): 是否正确穿上新品服装
- 人体自然度 (1-5分): 姿势、合身度是否自然
- 场景质量 (1-5分): 背景、场景是否专业
- 光影效果 (1-5分): 光线是否自然、专业
- 整体商业价值 (1-5分): 是否适合电商使用
"""
import sys
import os
import io
from pathlib import Path
from typing import List, Dict, Tuple
import statistics
import json
from datetime import datetime
from PIL import Image

# 设置编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OPENROUTER_API_KEY, NEW_PRODUCT_DIR, IMAGE_DIR, LLM_MODEL, IMAGE_GEN_MODEL
from utils import load_image, image_to_uri, save_image, extract_images
from openai import OpenAI


class ImageQualityComparator:
    """图片质量对比器"""

    # 模拟普通电商卖家的���单 prompt（基线方案使用）
    BASELINE_PROMPT = (
        "请生成一张模特穿着这件服装的宣传照片。"
        "要专业、适合电商使用。"
    )

    def __init__(self):
        """初始化"""
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        self.results = []

    def load_new_products(self, limit: int = 3) -> List[Dict]:
        """加载新品图片用于测试"""
        products = []

        for img_file in sorted(os.listdir(NEW_PRODUCT_DIR))[:limit]:
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue

            new_id = img_file.rsplit('.', 1)[0]
            img_path = NEW_PRODUCT_DIR / img_file

            try:
                img = load_image(str(img_path))
                products.append({
                    'new_id': new_id,
                    'image': img,
                    'path': img_path
                })
            except FileNotFoundError:
                continue

        return products

    def _load_reference_images(self, num_refs: int = 3) -> List[Image.Image]:
        """从 images 目录加载爆款参考图片"""
        images = []
        for img_file in sorted(os.listdir(IMAGE_DIR))[:num_refs]:
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = IMAGE_DIR / img_file
            try:
                img = load_image(str(img_path))
                images.append(img)
            except FileNotFoundError:
                continue
        return images

    def generate_baseline_image(self, new_product_image: Image.Image) -> Image.Image:
        """
        基线方案：仅使用新品图 + 通用 prompt

        Args:
            new_product_image: 新品图片

        Returns:
            生成的图片
        """
        print("    [基线方案] 生成中...")

        gen_content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(new_product_image)}},
            {"type": "text", "text": self.BASELINE_PROMPT},
        ]

        response = self.client.chat.completions.create(
            model=IMAGE_GEN_MODEL,
            messages=[{"role": "user", "content": gen_content}],
            extra_body={
                "modalities": ["image"],
                "image_config": {
                    "aspect_ratio": "3:4",
                    "image_size": "2K"
                },
            },
        )

        images = extract_images(response)
        return images[0] if images else None

    def generate_with_reference(
        self,
        new_product_image: Image.Image,
        reference_images: List[Image.Image]
    ) -> Tuple[Image.Image, str]:
        """
        本方案：使用新品图 + 参考图 + LLM 风格分析

        Args:
            new_product_image: 新品图片
            reference_images: 参考爆款图片

        Returns:
            (生成图片, 风格prompt)
        """
        print("    [本方案] 风格分析中...")

        # 分析风格
        style_content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(img)}}
            for img in reference_images
        ]
        style_content.append({
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

        style_response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": style_content}],
            max_tokens=512,
            temperature=0.7,
        )

        style_prompt = style_response.choices[0].message.content.strip()
        print(f"    [本方案] 风格: {style_prompt[:50]}...")

        # 生成图片
        print("    [本方案] 生成中...")

        gen_prompt = (
            f"图片1是一件新的服装产品（平铺照片），图片2是我们畅销目录中的参考宣传照。\n\n"
            f"请生成一张专业的电商宣传照片，展示一位女性模特穿着图片1中的服装。\n\n"
            f"风格指导：{style_prompt}\n\n"
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
            {"type": "image_url", "image_url": {"url": image_to_uri(reference_images[0])}},
            {"type": "text", "text": gen_prompt},
        ]

        gen_response = self.client.chat.completions.create(
            model=IMAGE_GEN_MODEL,
            messages=[{"role": "user", "content": gen_content}],
            extra_body={
                "modalities": ["image"],
                "image_config": {
                    "aspect_ratio": "3:4",
                    "image_size": "2K"
                },
            },
        )

        images = extract_images(gen_response)
        generated = images[0] if images else None

        return generated, style_prompt

    def score_image_with_vlm(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        method_name: str
    ) -> Dict[str, int]:
        """
        使用 VLM 对生成图片评分

        Args:
            generated_image: 生成的图片
            original_image: 原始新品图片
            method_name: 方法名称（用于 prompt）

        Returns:
            评分字典 {维度: 分数}
        """
        print(f"    [VLM 评分] 评估 {method_name}...")

        score_prompt = """
你是一位专业的电商照片质量评估专家。请评估生成的宣传照片（图片2）与原始产品平铺图（图片1）的对比。

请对以下维度进行评分（1-5分）：
1. clothing_accuracy：图片2中的服装与图片1中的原始产品匹配度如何？
2. pose_naturalness：模特的姿势和合身度自然吗？
3. scene_quality：背景/场景专业吗？
4. lighting_quality：布光质量如何？
5. commercial_value：总体来说，这张图片适合电商使用吗？

只输出JSON格式，例如：
{"clothing_accuracy": 4, "pose_naturalness": 3, "scene_quality": 5, "lighting_quality": 4, "commercial_value": 4}
"""

        content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(original_image)}},
            {"type": "image_url", "image_url": {"url": image_to_uri(generated_image)}},
            {"type": "text", "text": score_prompt},
        ]

        response = self.client.chat.completions.create(
            model=LLM_MODEL,  # 使用支持视觉的模型
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
            temperature=0.3,
        )

        import re
        content_text = response.choices[0].message.content

        # 提取 JSON
        json_match = re.search(r'\{[^}]+\}', content_text)
        if json_match:
            try:
                scores = json.loads(json_match.group())
                return scores
            except json.JSONDecodeError:
                pass

        # 如果解析失败，返回默认分数
        return {
            "clothing_accuracy": 3,
            "pose_naturalness": 3,
            "scene_quality": 3,
            "lighting_quality": 3,
            "commercial_value": 3
        }

    def run_comparison_test(
        self,
        limit: int = 3,
        use_reference_images: List[Image.Image] = None
    ) -> Dict:
        """
        运行对比测试

        Args:
            limit: 测试新品数量
            use_reference_images: 指定参考图列表（如果为 None，从 images 目录加载爆款图片）

        Returns:
            测试结果
        """
        print("\n" + "=" * 70)
        print("图片质量对比测试")
        print("=" * 70)

        products = self.load_new_products(limit)
        print(f"\n加载了 {len(products)} 个新品用于测试")

        # 如果没有提供参考图，从 images 目录加载爆款图片
        if use_reference_images is None:
            use_reference_images = self._load_reference_images()
            print(f"加载了 {len(use_reference_images)} 张爆款参考图")

        comparison_results = []

        for i, product in enumerate(products):
            print(f"\n[{i+1}/{len(products)}] 测试新品: {product['new_id']}")
            print("-" * 50)

            result = {
                'new_id': product['new_id'],
                'baseline': {},
                'with_reference': {}
            }

            # 创建输出目录
            output_dir = Path(__file__).parent / "results" / "quality_comparison"
            output_dir.mkdir(exist_ok=True)

            # 1. 基线方案
            try:
                baseline_img = self.generate_baseline_image(product['image'])
                if baseline_img:
                    # 保存图片
                    baseline_path = output_dir / f"{product['new_id']}_baseline.png"
                    save_image(baseline_img, str(baseline_path))

                    # VLM 评分
                    baseline_scores = self.score_image_with_vlm(
                        baseline_img, product['image'], "基线方案"
                    )

                    result['baseline'] = {
                        'success': True,
                        'path': str(baseline_path),
                        'scores': baseline_scores,
                        'avg_score': statistics.mean(baseline_scores.values())
                    }
                    print(f"    [基线方案] 平均分: {result['baseline']['avg_score']:.2f}")
                else:
                    result['baseline'] = {'success': False, 'error': 'generation_failed'}
            except Exception as e:
                result['baseline'] = {'success': False, 'error': str(e)}
                print(f"    [基线方案] 失败: {e}")

            # 2. 本方案（带参考图）
            try:
                with_ref_img, style_prompt = self.generate_with_reference(
                    product['image'], use_reference_images
                )
                if with_ref_img:
                    # 保存图片
                    ref_path = output_dir / f"{product['new_id']}_with_reference.png"
                    save_image(with_ref_img, str(ref_path))

                    # VLM 评分
                    ref_scores = self.score_image_with_vlm(
                        with_ref_img, product['image'], "本方案"
                    )

                    result['with_reference'] = {
                        'success': True,
                        'path': str(ref_path),
                        'scores': ref_scores,
                        'avg_score': statistics.mean(ref_scores.values()),
                        'style_prompt': style_prompt
                    }
                    print(f"    [本方案] 平均分: {result['with_reference']['avg_score']:.2f}")
                else:
                    result['with_reference'] = {'success': False, 'error': 'generation_failed'}
            except Exception as e:
                result['with_reference'] = {'success': False, 'error': str(e)}
                print(f"    [本方案] 失败: {e}")

            # 计算差异
            if result['baseline'].get('success') and result['with_reference'].get('success'):
                improvement = (
                    result['with_reference']['avg_score'] - result['baseline']['avg_score']
                )
                result['improvement'] = improvement
                print(f"    [差异] 本方案提升: {improvement:+.2f} 分")

            comparison_results.append(result)

        # 汇总统计
        return self._summarize_results(comparison_results)

    def _summarize_results(self, results: List[Dict]) -> Dict:
        """汇总测试结果"""
        successful = [r for r in results
                     if r.get('baseline', {}).get('success')
                     and r.get('with_reference', {}).get('success')]

        if not successful:
            print("\n警告: 没有成功的对比测试")
            return {'error': 'no_successful_tests'}

        # 计算各维度平均分
        dimensions = ['clothing_accuracy', 'pose_naturalness', 'scene_quality',
                     'lighting_quality', 'commercial_value']

        baseline_avg = {}
        with_ref_avg = {}

        for dim in dimensions:
            baseline_avg[dim] = statistics.mean(
                [r['baseline']['scores'][dim] for r in successful]
            )
            with_ref_avg[dim] = statistics.mean(
                [r['with_reference']['scores'][dim] for r in successful]
            )

        # 总体平均分
        overall_baseline = statistics.mean([r['baseline']['avg_score'] for r in successful])
        overall_with_ref = statistics.mean([r['with_reference']['avg_score'] for r in successful])

        print("\n" + "=" * 70)
        print("测试结果汇总")
        print("=" * 70)

        print(f"\n成功对比: {len(successful)} 组")
        print(f"\n{'维度':<20s} {'基线方案':<12s} {'本方案':<12s} {'提升'}")
        print("-" * 55)

        for dim in dimensions:
            improvement = with_ref_avg[dim] - baseline_avg[dim]
            print(f"{dim:<20s} {baseline_avg[dim]:<12.2f} {with_ref_avg[dim]:<12.2f} {improvement:+.2f}")

        print("-" * 55)
        overall_improvement = overall_with_ref - overall_baseline
        print(f"{'总体平均':<20s} {overall_baseline:<12.2f} {overall_with_ref:<12.2f} {overall_improvement:+.2f}")
        print("=" * 70)

        # 计算提升百分比
        improvement_percent = (overall_improvement / overall_baseline) * 100

        return {
            'total_tests': len(results),
            'successful_tests': len(successful),
            'baseline_avg': baseline_avg,
            'with_reference_avg': with_ref_avg,
            'overall_baseline': overall_baseline,
            'overall_with_reference': overall_with_ref,
            'overall_improvement': overall_improvement,
            'improvement_percent': improvement_percent,
            'detailed_results': results
        }

    def save_report(self, results: Dict):
        """保存测试报告"""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"quality_comparison_{timestamp}.json"

        # 移除无法序列化的内容
        serializable_results = []
        for r in results.get('detailed_results', []):
            serializable = {
                'new_id': r['new_id'],
                'baseline': {
                    'success': r['baseline'].get('success', False),
                    'path': r['baseline'].get('path', ''),
                    'scores': r['baseline'].get('scores', {}),
                    'avg_score': r['baseline'].get('avg_score', 0)
                },
                'with_reference': {
                    'success': r['with_reference'].get('success', False),
                    'path': r['with_reference'].get('path', ''),
                    'scores': r['with_reference'].get('scores', {}),
                    'avg_score': r['with_reference'].get('avg_score', 0)
                }
            }
            if 'improvement' in r:
                serializable['improvement'] = r['improvement']
            serializable_results.append(serializable)

        report = {
            'timestamp': timestamp,
            'summary': {
                'total_tests': results.get('total_tests'),
                'successful_tests': results.get('successful_tests'),
                'overall_baseline': results.get('overall_baseline'),
                'overall_with_reference': results.get('overall_with_reference'),
                'overall_improvement': results.get('overall_improvement'),
                'improvement_percent': results.get('improvement_percent')
            },
            'dimension_comparison': {
                'baseline': results.get('baseline_avg', {}),
                'with_reference': results.get('with_reference_avg', {})
            },
            'detailed_results': serializable_results
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n报告已保存到: {report_file}")
        return report_file


def main():
    """主函数"""
    comparator = ImageQualityComparator()

    # 运行测试（测试 3 个新品，平衡成本和准确性）
    results = comparator.run_comparison_test(limit=3)

    # 保存报告
    if results.get('successful_tests', 0) > 0:
        comparator.save_report(results)


if __name__ == "__main__":
    main()
