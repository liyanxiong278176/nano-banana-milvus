"""
生图质量对比测试 - 改进版

对比五种方案:
   - 基线方案: 新品图 + 简单 prompt
   - 原方案: 新品图 + 固定爆款图 + LLM 风格分析
   - 改进方案A: 新品图 + 多张固定爆款图（无 LLM 分析）
   - 改进方案B: 新品图 + 单张固定爆款图（无 LLM 分析）
   - 改进方案C: 新品图 + **检索相似爆款图** + LLM 风格分析（新增）
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

# 导入检索相关模块
from retrieval import BestsellerRetriever
from embedding import EmbeddingGenerator
from utils import sparse_to_dict
from sklearn.feature_extraction.text import TfidfVectorizer


class ImprovedImageQualityComparator:
    """改进版图片质量对比器"""

    # 基线 prompt
    BASELINE_PROMPT = (
        "请生成一张模特穿着这件服装的宣传照片。"
        "要专业、适合电商使用。"
    )

    # 改进方案 prompt (多参考图，无LLM分析)
    MULTI_REF_PROMPT = (
        "图片1是一件新的服装产品（平铺照片）。\n"
        "图片2-4是我们畅销的宣传照片。\n\n"
        "请生成一张专业的电商宣传照片，展示一位女性模特穿着图片1中的服装。\n\n"
        "要求：\n"
        "- 风格和氛围与图片2-4匹配\n"
        "- 全身照，照片级真实感，高质量\n"
        "- 服装必须与图片1完全一致\n"
        "- 服装与模特身体的贴合自然\n"
        "- 专业布光，细节清晰"
    )

    def __init__(self):
        """初始化"""
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        self.results = []

        # 初始化检索器和 TF-IDF（用于检索相似爆款图）
        self.use_retrieval = False
        self.retriever = None
        self.tfidf = None

        try:
            self.retriever = BestsellerRetriever()
            if self.retriever.has_collection():
                # 加载一个产品来构建 TF-IDF
                import csv
                products = []
                product_csv = Path(__file__).parent.parent / "products.csv"
                if product_csv.exists():
                    with open(product_csv, newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            products.append(row)

                    if products:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        descriptions = [p.get("description", "") for p in products]
                        self.tfidf = TfidfVectorizer(stop_words="english", max_features=500)
                        self.tfidf.fit(descriptions)
                        self.use_retrieval = True
                        print("检索系统初始化成功")
            else:
                print("Collection 不存在，将使用固定爆款图")
        except Exception as e:
            print(f"警告: 检索系统初始化失败 ({e})，将使用固定爆款图")

    def load_new_products(self, limit: int = 3) -> List[Dict]:
        """加载新品图片用于测试"""
        products = []
        product_info = {}  # 存储从 CSV 读取的品类信息

        # 从 new_products.csv 读取品类信息
        import csv
        csv_path = Path(__file__).parent.parent / "new_products.csv"
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    new_id = row.get('new_id', '').rsplit('.', 1)[0]  # 去掉 .jpg 后缀
                    product_info[new_id] = {
                        'category': row.get('category', 'dress'),
                        'style': row.get('style', ''),
                        'season': row.get('season', ''),
                    }

        for img_file in sorted(os.listdir(NEW_PRODUCT_DIR))[:limit]:
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue

            new_id = img_file.rsplit('.', 1)[0]
            img_path = NEW_PRODUCT_DIR / img_file

            try:
                img = load_image(str(img_path))
                # 添加品类信息
                info = product_info.get(new_id, {})
                products.append({
                    'new_id': new_id,
                    'image': img,
                    'path': img_path,
                    'category': info.get('category', 'dress'),
                    'style': info.get('style', ''),
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

    def retrieve_similar_products(
        self,
        new_product_image: Image.Image,
        category: str = "dress",
        top_k: int = 3
    ) -> List[Tuple[Image.Image, Dict]]:
        """
        检索与新品相似的爆款产品

        Returns:
            List of (image, product_info) tuples
        """
        if not self.use_retrieval or not self.retriever:
            return [(img, {}) for img in self._load_reference_images(top_k)]

        try:
            from utils import get_image_embeddings
            import numpy as np

            # 获取新品的向量
            dense_vectors = get_image_embeddings([new_product_image], batch_size=1)
            if len(dense_vectors) == 0:
                raise RuntimeError("图片编码失败")
            query_dense = dense_vectors[0].tolist()

            # 简单的稀疏向量（基于品类）
            query_text = f"{category} fashion clothing"
            query_sparse = sparse_to_dict(self.tfidf.transform([query_text])[0])

            # 检索相似产品
            results = self.retriever._hybrid_search(
                query_dense=query_dense,
                query_sparse=query_sparse,
                filter_expr=f'sales_count > 0',  # 只要有销量的都可以
                top_k=top_k
            )

            # 加载检索到的图片
            retrieved = []
            for hit in results[:top_k]:
                entity = hit.get("entity", {})
                product_id = entity.get("product_id")
                if product_id:
                    img_path = IMAGE_DIR / f"{product_id}.jpg"
                    try:
                        img = load_image(str(img_path))
                        retrieved.append((img, entity))
                    except FileNotFoundError:
                        continue

            return retrieved if retrieved else [(img, {}) for img in self._load_reference_images(top_k)]

        except Exception as e:
            print(f"    检索失败，使用固定爆款图: {e}")
            return [(img, {}) for img in self._load_reference_images(top_k)]

    def generate_baseline(self, new_product_image: Image.Image) -> Image.Image:
        """基线方案：仅使用新品图 + 通用 prompt"""
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

    def generate_original_scheme(
        self,
        new_product_image: Image.Image,
        reference_images: List[Image.Image]
    ) -> Tuple[Image.Image, str]:
        """原方案：新品图 + 固定爆款图 + LLM 风格分析"""
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

        # 生成图片（仅用第一张参考图）
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

    def generate_improved_multi_ref(
        self,
        new_product_image: Image.Image,
        reference_images: List[Image.Image]
    ) -> Image.Image:
        """改进方案A：新品图 + 多张固定爆款图（无 LLM 分析）"""
        gen_content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(new_product_image)}},
        ]
        # 添加多张参考图
        for ref_img in reference_images:
            gen_content.append({"type": "image_url", "image_url": {"url": image_to_uri(ref_img)}})
        gen_content.append({"type": "text", "text": self.MULTI_REF_PROMPT})

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

    def generate_improved_single_ref(
        self,
        new_product_image: Image.Image,
        reference_images: List[Image.Image]
    ) -> Image.Image:
        """改进方案B：新品图 + 单张固定爆款图（无 LLM 分析）"""
        gen_content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(new_product_image)}},
            {"type": "image_url", "image_url": {"url": image_to_uri(reference_images[0])}},
            {"type": "text", "text": (
                "图片1：新的服装产品（平铺照片）\n"
                "图片2：参考宣传照片\n\n"
                "请生成一张专业的电商照片，展示模特穿着图片1中的服装，"
                "风格与图片2匹配。"
            )},
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

    def generate_retrieval_scheme(
        self,
        new_product_image: Image.Image,
        category: str = "dress"
    ) -> Tuple[Image.Image, str, List[str]]:
        """
        改进方案C：新品图 + 检索相似爆款图 + LLM 风格分析

        Returns:
            (生成图片, 风格prompt, 检索到的产品ID列表)
        """
        # 1. 检索相似爆款图
        print("    [方案C] 检索相似爆款图中...")
        retrieved_products = self.retrieve_similar_products(new_product_image, category, top_k=3)

        retrieved_images = [img for img, _ in retrieved_products]
        retrieved_ids = [info.get('product_id', 'unknown') for _, info in retrieved_products]
        print(f"    [方案C] 检索到: {', '.join(retrieved_ids)}")

        # 2. 分析风格
        print("    [方案C] 分析风格...")
        style_content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(img)}}
            for img in retrieved_images
        ]
        style_content.append({
            "type": "text",
            "text": (
                "这些是我们与新品相似的畅销时尚产品宣传照片。\n\n"
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
        print(f"    [方案C] 风格: {style_prompt[:50]}...")

        # 3. 生成图片（使用检索到的参考图）
        print("    [方案C] 生成图片...")
        gen_prompt = (
            f"图片1是一件新的服装产品���平铺照片）。\n"
            f"图片2及以后是我们目录中与新品相似的畅销宣传照片。\n\n"
            f"请生成一张专业的电商宣传照片，展示一位女性模特穿着图片1中的服装。\n\n"
            f"风格指导：{style_prompt}\n\n"
            f"要求：\n"
            f"- 全身照，照片级真实感，高质量\n"
            f"- 服装必须与图片1完全一致\n"
            f"- 照片风格应与参考照片匹配\n"
            f"- 服装与模特身体的贴合自然\n"
            f"- 无可见标签或吊牌\n"
            f"- 细节清晰，专业布光"
        )

        gen_content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(new_product_image)}},
        ]
        # 添加检索到的参考图
        for ref_img in retrieved_images:
            gen_content.append({"type": "image_url", "image_url": {"url": image_to_uri(ref_img)}})
        gen_content.append({"type": "text", "text": gen_prompt})

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

        return generated, style_prompt, retrieved_ids

    def score_image_with_vlm(
        self,
        generated_image: Image.Image,
        original_image: Image.Image,
        method_name: str
    ) -> Dict[str, int]:
        """使用 VLM 对生成图片评分"""
        score_prompt = """
你是一位专业的电商照片质量评估专家。请评估生成的宣传照片（图片2）与原始产品平铺图（图片1）的对比。

请对以下维度进行评分（1-5分）：
1. clothing_accuracy：图片2中的服装与图片1中的原始产品匹配度如何？
2. pose_naturalness：模特的姿势���合身度自然吗？
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
            model=LLM_MODEL,
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

        return {
            "clothing_accuracy": 3,
            "pose_naturalness": 3,
            "scene_quality": 3,
            "lighting_quality": 3,
            "commercial_value": 3
        }

    def run_comparison_test(self, limit: int = 2) -> Dict:
        """运行对比测试（测试5种方案）"""
        print("\n" + "=" * 70)
        print("图片质量对比测试 - 改进版 (5种方案对比)")
        print("=" * 70)

        products = self.load_new_products(limit)
        print(f"\n加载了 {len(products)} 个新品用于测试")

        # 固定爆款图（用于原方案和改进方案A、B）
        fixed_reference_images = self._load_reference_images(3)
        print(f"加载了 {len(fixed_reference_images)} 张固定爆款参考图")

        comparison_results = []
        output_dir = Path(__file__).parent / "results" / "quality_comparison_improved"
        output_dir.mkdir(exist_ok=True)

        for i, product in enumerate(products):
            print(f"\n[{i+1}/{len(products)}] 测试新品: {product['new_id']}")
            print("-" * 50)

            result = {
                'new_id': product['new_id'],
                'schemes': {}
            }

            # 测试5种方案
            schemes = [
                ('baseline', '基线方案', lambda: self.generate_baseline(product['image'])),
                ('original', '原方案(固定爆款+LLM)', lambda: self.generate_original_scheme(product['image'], fixed_reference_images)),
                ('multi_ref', '改进A(多固定爆���无LLM)', lambda: self.generate_improved_multi_ref(product['image'], fixed_reference_images)),
                ('single_ref', '改进B(单固定爆款无LLM)', lambda: self.generate_improved_single_ref(product['image'], fixed_reference_images)),
                ('retrieval', '改进C(检索相似爆款+LLM)', lambda: self.generate_retrieval_scheme(product['image'], product.get('category', 'dress'))),
            ]

            for scheme_key, scheme_name, generator in schemes:
                print(f"  [{scheme_name}] 生成中...")
                try:
                    if scheme_key in ['original', 'retrieval']:
                        generated, style_prompt, *extra = generator()
                        retrieved_ids = extra[0] if scheme_key == 'retrieval' and extra else None
                    else:
                        generated = generator()
                        style_prompt = None
                        retrieved_ids = None

                    if generated:
                        # 保存图片
                        img_path = output_dir / f"{product['new_id']}_{scheme_key}.png"
                        save_image(generated, str(img_path))

                        # VLM 评分
                        scores = self.score_image_with_vlm(generated, product['image'], scheme_name)
                        avg_score = statistics.mean(scores.values())

                        scheme_result = {
                            'success': True,
                            'path': str(img_path),
                            'scores': scores,
                            'avg_score': avg_score,
                        }
                        if style_prompt:
                            scheme_result['style_prompt'] = style_prompt
                        if retrieved_ids:
                            scheme_result['retrieved_ids'] = retrieved_ids

                        result['schemes'][scheme_key] = scheme_result
                        print(f"    平均分: {avg_score:.2f}")
                    else:
                        result['schemes'][scheme_key] = {'success': False, 'error': 'generation_failed'}
                except Exception as e:
                    result['schemes'][scheme_key] = {'success': False, 'error': str(e)}
                    print(f"    失败: {e}")

            comparison_results.append(result)

        return self._summarize_results(comparison_results)

    def _summarize_results(self, results: List[Dict]) -> Dict:
        """汇总测试结果"""
        print("\n" + "=" * 70)
        print("测试结果汇总")
        print("=" * 70)

        schemes = ['baseline', 'original', 'multi_ref', 'single_ref', 'retrieval']
        scheme_names = {
            'baseline': '基线方案',
            'original': '原方案(固定爆款+LLM)',
            'multi_ref': '改进A(多固定爆款无LLM)',
            'single_ref': '改进B(单固定爆款无LLM)',
            'retrieval': '改进C(检索相似爆款+LLM)'
        }

        # 计算各方案平均���
        scheme_avgs = {}
        for scheme in schemes:
            scores = []
            for r in results:
                if r['schemes'].get(scheme, {}).get('success'):
                    scores.append(r['schemes'][scheme]['avg_score'])
            if scores:
                scheme_avgs[scheme] = statistics.mean(scores)

        print(f"\n{'方案':<25s} {'平均分':<10s} {'对比基线'}")
        print("-" * 45)

        baseline_avg = scheme_avgs.get('baseline', 0)
        for scheme in schemes:
            avg = scheme_avgs.get(scheme, 0)
            diff = avg - baseline_avg if scheme != 'baseline' else 0
            print(f"{scheme_names[scheme]:<25s} {avg:<10.2f} {diff:+.2f}")

        print("=" * 70)

        # 找出最佳方案
        if scheme_avgs:
            best_scheme = max(scheme_avgs.items(), key=lambda x: x[1])
            print(f"\n最佳方案: {scheme_names[best_scheme[0]]} ({best_scheme[1]:.2f}分)")

        return {
            'scheme_averages': scheme_avgs,
            'best_scheme': max(scheme_avgs.items(), key=lambda x: x[1])[0] if scheme_avgs else None,
            'detailed_results': results
        }

    def save_report(self, results: Dict):
        """保存测试报告"""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"quality_comparison_improved_{timestamp}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n报告已保存到: {report_file}")
        return report_file


def main():
    """主函数"""
    comparator = ImprovedImageQualityComparator()

    # 运行测试（2个新品以节省成本）
    results = comparator.run_comparison_test(limit=2)

    # 保存报告
    if results.get('scheme_averages'):
        comparator.save_report(results)


if __name__ == "__main__":
    main()
