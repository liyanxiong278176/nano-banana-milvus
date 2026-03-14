"""
检索召回率和准确率测试

测试目标：
1. 使用 new_products 中的新品作为查询
2. 对于每个新品，检查检索到的相似爆款是否与新品类别/风格匹配
3. 计算召回率(Recall)和准确率(Precision)

工作原理：
- 由于我们没有人工标注的"正确答案"，我们使用品类和风格作为评估基准
- 如果检索结果的类别与查询新品类别相同，则认为相关
- 通过计算 Top-K 检索结果中相关商品的比例来评估准确率
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import statistics

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import IMAGE_DIR, NEW_PRODUCT_DIR, NEW_PRODUCT_CSV
from retrieval import BestsellerRetriever
from embedding import EmbeddingGenerator
from utils import load_image


class RetrievalMetricsTester:
    """检索指标测试器"""

    def __init__(self):
        """初始化测试器"""
        print("初始化检索指标测试器...")
        self.retriever = BestsellerRetriever()
        self.embed_gen = EmbeddingGenerator()

        # 加载 TF-IDF 向量化器
        products, _ = self.embed_gen.load_products()
        self.tfidf = self.embed_gen.build_tfidf_vectorizer(products)

        # 加载新品数据
        self.new_products = self._load_new_products_with_images()

        print(f"加载了 {len(self.new_products)} 个新品")

    def _load_new_products_with_images(self) -> List[Dict]:
        """
        加载新品数据和图片

        Returns:
            新品列表，包含元数据和图片
        """
        new_products = []

        # 遍历 new_products 目录
        for img_file in os.listdir(NEW_PRODUCT_DIR):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue

            # 从文件名提取 new_id
            new_id = img_file.rsplit('.', 1)[0]

            # 加载图片
            img_path = NEW_PRODUCT_DIR / img_file
            try:
                img = load_image(str(img_path))
            except FileNotFoundError:
                print(f"警告: 图片不存在 {img_path}")
                continue

            # 从 CSV 获取元数据
            import csv
            with open(NEW_PRODUCT_CSV, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['new_id'] == new_id:
                        new_products.append({
                            'new_id': new_id,
                            'image': img,
                            'category': row.get('category', ''),
                            'style': row.get('style', ''),
                            'season': row.get('season', ''),
                            'prompt_hint': row.get('prompt_hint', '')
                        })
                        break
                else:
                    # 如果 CSV 中没有，使用默认值
                    new_products.append({
                        'new_id': new_id,
                        'image': img,
                        'category': 'unknown',
                        'style': 'unknown',
                        'season': 'all_season',
                        'prompt_hint': ''
                    })

        return new_products

    def encode_query(self, new_product: Dict) -> Tuple:
        """
        编码新品查询

        Args:
            new_product: 新品数据

        Returns:
            (稠密向量, 稀疏向量)
        """
        # Dense 向量: 图片编码
        from utils import get_image_embeddings
        dense_vectors = get_image_embeddings([new_product['image']], batch_size=1)
        dense = dense_vectors[0].tolist() if len(dense_vectors) > 0 else None

        # Sparse 向量: 文本查询编码
        from utils import sparse_to_dict
        query_text = (
            f"{new_product.get('category', '')} "
            f"{new_product.get('style', '')} "
            f"{new_product.get('season', '')} "
            f"{new_product.get('prompt_hint', '')}"
        )
        sparse = sparse_to_dict(self.tfidf.transform([query_text])[0])

        return dense, sparse

    def calculate_category_match(self, retrieved: List[Dict], query_category: str) -> float:
        """
        计算品类匹配率

        Args:
            retrieved: 检索结果列表
            query_category: 查询新品类别

        Returns:
            匹配率 (0-1)
        """
        if not retrieved:
            return 0.0

        matches = sum(1 for r in retrieved if r.get('category') == query_category)
        return matches / len(retrieved)

    def calculate_ndcg(self, retrieved: List[Dict], query_category: str, k: int = 3) -> float:
        """
        计算 NDCG@K (Normalized Discounted Cumulative Gain)

        Args:
            retrieved: 检索结果列表
            query_category: 查询新品类别
            k: Top-K

        Returns:
            NDCG 分数 (0-1)
        """
        import math

        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for i, item in enumerate(retrieved[:k]):
            # 相关性: 类别匹配为 1，不匹配为 0
            relevance = 1 if item.get('category') == query_category else 0
            # DCG = relevance / log2(i+2)
            dcg += relevance / math.log2(i + 2)

        # 理想 DCG: 所有结果都相关
        ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(retrieved))))

        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def run_single_test(self, new_product: Dict, top_k: int = 3) -> Dict:
        """
        对单个新品运行测试

        Args:
            new_product: 新品数据
            top_k: 检索数量

        Returns:
            测试结果字典
        """
        print(f"\n测试新品: {new_product['new_id']}")
        print(f"  类别: {new_product['category']}, 风格: {new_product['style']}")

        # 编码查询
        dense, sparse = self.encode_query(new_product)

        if dense is None:
            print(f"  错误: 编码失败")
            return {
                'new_id': new_product['new_id'],
                'success': False,
                'error': 'encoding_failed'
            }

        # 执行检索
        retrieved = self.retriever._hybrid_search(
            query_dense=dense,
            query_sparse=sparse,
            filter_expr=f'category == "{new_product["category"]}"',
            top_k=top_k
        )

        # 计算指标
        category_match = self.calculate_category_match(retrieved, new_product['category'])
        ndcg = self.calculate_ndcg(retrieved, new_product['category'], top_k)

        print(f"  检索到 {len(retrieved)} 个结果")
        print(f"  品类匹配率: {category_match:.2%}")
        print(f"  NDCG@{top_k}: {ndcg:.4f}")

        # 显示检索结果
        for i, hit in enumerate(retrieved, 1):
            entity = hit.get("entity", {})
            print(f"    {i}. {entity.get('product_id', 'N/A')} | "
                  f"{entity.get('category', 'N/A')} | {entity.get('style', 'N/A')}")

        return {
            'new_id': new_product['new_id'],
            'query_category': new_product['category'],
            'query_style': new_product['style'],
            'success': True,
            'retrieved_count': len(retrieved),
            'category_match_rate': category_match,
            'ndcg': ndcg,
            'retrieved_products': [
                {
                    'product_id': hit.get('entity', {}).get('product_id'),
                    'category': hit.get('entity', {}).get('category'),
                    'style': hit.get('entity', {}).get('style'),
                    'score': hit.get('distance')
                }
                for hit in retrieved
            ]
        }

    def run_all_tests(self, top_k: int = 3) -> Dict:
        """
        运行所有新品测试

        Args:
            top_k: 检索数量

        Returns:
            汇总统计结果
        """
        print("\n" + "=" * 60)
        print("开始检索指标测试")
        print("=" * 60)

        results = []
        for new_product in self.new_products:
            result = self.run_single_test(new_product, top_k)
            results.append(result)

        # 计算汇总统计
        successful_results = [r for r in results if r.get('success', False)]

        if not successful_results:
            print("\n警告: 所有测试都失败了")
            return {'error': 'all_tests_failed'}

        avg_category_match = statistics.mean(
            [r['category_match_rate'] for r in successful_results]
        )
        avg_ndcg = statistics.mean(
            [r['ndcg'] for r in successful_results]
        )

        # 按 Top-K 准确率统计
        top1_accuracy = sum(
            1 for r in successful_results
            if len(r.get('retrieved_products', [])) > 0
            and r['retrieved_products'][0].get('category') == r['query_category']
        ) / len(successful_results)

        # Top-3 召回率
        top3_recall = sum(
            1 for r in successful_results
            if any(
                p.get('category') == r['query_category']
                for p in r.get('retrieved_products', [])
            )
        ) / len(successful_results)

        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print(f"测试样本数: {len(successful_results)}")
        print(f"平均品类匹配率: {avg_category_match:.2%}")
        print(f"平均 NDCG@{top_k}: {avg_ndcg:.4f}")
        print(f"Top-1 准确率: {top1_accuracy:.2%}")
        print(f"Top-3 召回率: {top3_recall:.2%}")
        print("=" * 60)

        return {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'avg_category_match_rate': avg_category_match,
            'avg_ndcg': avg_ndcg,
            'top1_accuracy': top1_accuracy,
            'top3_recall': top3_recall,
            'detailed_results': results
        }


def main():
    """主函数"""
    tester = RetrievalMetricsTester()
    results = tester.run_all_tests(top_k=3)

    # 保存结果到文件
    import json
    from datetime import datetime

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"retrieval_metrics_{timestamp}.json"

    with open(result_file, 'w', encoding='utf-8') as f:
        # 将不能序列化的对象转换为字符串
        serializable_results = {
            k: v if not isinstance(v, list) or not v else
               [{**item, 'retrieved_products': item.get('retrieved_products', [])}
                for item in v] if isinstance(v, list) else v
            for k, v in results.items()
        }
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
