"""
批量处理模块 - 批量生成新品宣传图
"""
import csv
import time
from pathlib import Path
from typing import List, Dict
from PIL import Image

from config import OUTPUT_DIR, TOP_K_RETRIEVAL, EMBED_BATCH_SIZE
from milvus_db import FashionMilvusDB
from embedding import EmbeddingGenerator
from retrieval import BestsellerRetriever
from image_gen import ImageGenerator
from utils import save_image


class BatchProcessor:
    """批量处理器"""

    def __init__(self):
        """初始化"""
        self.milvus_db = FashionMilvusDB()
        self.embed_gen = EmbeddingGenerator()
        self.retriever = BestsellerRetriever(self.milvus_db)
        self.image_gen = ImageGenerator()

        # 加载 TF-IDF 向量化器
        products, _ = self.embed_gen.load_products()
        self.tfidf = self.embed_gen.build_tfidf_vectorizer(products)

    def process_batch(
        self,
        new_product_ids: List[str] = None,
        save_output: bool = True
    ) -> List[Dict]:
        """
        批量处理新品

        Args:
            new_product_ids: 新品 ID 列表，None 则处理全部
            save_output: 是否保存输出

        Returns:
            处理结果列表
        """
        # 加载新品数据
        new_products = self.embed_gen.load_new_products()

        if new_product_ids is None:
            new_product_ids = [p["new_id"] for p in new_products]

        results = []
        total = len(new_product_ids)

        print(f"\n{'='*60}")
        print(f"批量处理: {total} 个新品")
        print(f"{'='*60}\n")

        for i, new_id in enumerate(new_product_ids, 1):
            print(f"\n[{i}/{total}] 处理: {new_id}")
            print("-" * 60)

            # 查找新品数据
            new_product = next(
                (p for p in new_products if p["new_id"] == new_id),
                None
            )

            if not new_product:
                print(f"  跳过: 找不到新品数据")
                continue

            result = self._process_single(
                new_product,
                save_output=save_output
            )

            if result:
                results.append(result)

            # 礼貌延迟
            if i < total:
                time.sleep(1)

        # 生成汇总报告
        self._generate_summary(results, save_output)

        return results

    def _process_single(self, new_product: Dict, save_output: bool) -> Dict:
        """处理单个新品"""
        try:
            # 编码
            query_dense, query_sparse, new_img = self.embed_gen.encode_new_product(
                new_product, self.tfidf
            )

            # 检索
            retrieved = self.retriever.retrieve_similar_bestsellers(
                query_dense=query_dense.tolist(),
                query_sparse=query_sparse,
                category=new_product["category"],
                top_k=TOP_K_RETRIEVAL
            )

            if not retrieved:
                print("  跳过: 未找到相似爆款")
                return None

            # 生成
            ref_images = [r["image"] for r in retrieved if r["image"]]

            style_prompt, generated = self.image_gen.process_single_product(
                new_product_image=new_img,
                reference_images=ref_images,
                scene_hint=new_product.get("prompt_hint", "")
            )

            result = {
                "new_id": new_product["new_id"],
                "category": new_product["category"],
                "style": new_product["style"],
                "retrieved_count": len(retrieved),
                "generated_count": len(generated) if generated else 0,
                "style_prompt": style_prompt,
                "success": len(generated) > 0 if generated else False
            }

            # 保存
            if save_output and generated:
                self._save_output(new_product, new_img, retrieved, generated, style_prompt)

            print(f"  完成: 生成 {len(generated)} 张图片")
            return result

        except Exception as e:
            print(f"  错误: {e}")
            return {
                "new_id": new_product["new_id"],
                "success": False,
                "error": str(e)
            }

    def _save_output(
        self,
        new_product: Dict,
        new_img: Image.Image,
        retrieved: List[Dict],
        generated: List[Image.Image],
        style_prompt: str
    ):
        """保存输出文件"""
        new_id = new_product["new_id"]
        output_dir = OUTPUT_DIR / new_id
        output_dir.mkdir(exist_ok=True)

        # 保存原图
        save_image(new_img, str(output_dir / f"{new_id}_original.png"))

        # 保存参考图
        for i, ref in enumerate(retrieved):
            if ref["image"]:
                save_image(
                    ref["image"],
                    str(output_dir / f"{new_id}_reference_{i+1}.png")
                )

        # 保存生成图
        for i, gen_img in enumerate(generated):
            save_image(
                gen_img,
                str(output_dir / f"{new_id}_generated_{i+1}.png")
            )

        # 保存 prompt
        with open(output_dir / f"{new_id}_style_prompt.txt", "w", encoding="utf-8") as f:
            f.write(style_prompt)

    def _generate_summary(self, results: List[Dict], save: bool = True):
        """生成汇总报告"""
        print(f"\n{'='*60}")
        print("批量处理汇总")
        print(f"{'='*60}\n")

        success_count = sum(1 for r in results if r.get("success"))
        fail_count = len(results) - success_count

        print(f"总计: {len(results)} 个新品")
        print(f"成功: {success_count} 个")
        print(f"失败: {fail_count} 个")

        print("\n详情:")
        print("-" * 60)
        for r in results:
            status = "[OK]" if r.get("success") else "[FAIL]"
            print(f"  {status} {r['new_id']} | {r.get('category', 'N/A')} | "
                  f"生成: {r.get('generated_count', 0)} 张")

        # 保存 CSV 报告
        if save:
            report_path = OUTPUT_DIR / "batch_report.csv"
            with open(report_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "new_id", "category", "style", "retrieved_count",
                    "generated_count", "success", "error", "style_prompt"
                ], extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)

            print(f"\n报告已保存: {report_path}")


def main():
    """批量处理入口"""
    import argparse

    parser = argparse.ArgumentParser(description="批量生成新品宣传图")
    parser.add_argument("--ids", nargs="+", help="指定新品 ID 列表")
    parser.add_argument("--all", action="store_true", help="处理所有新品")

    args = parser.parse_args()

    processor = BatchProcessor()

    if args.all:
        processor.process_batch()
    elif args.ids:
        processor.process_batch(args.ids)
    else:
        print("使用方法:")
        print("  python batch_process.py --all")
        print("  python batch_process.py --ids NEW001 NEW002")


if __name__ == "__main__":
    main()
