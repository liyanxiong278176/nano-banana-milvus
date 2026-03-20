"""
主程序 - 电商 AI 生图流水线
完整流程: 初始化数据库 → 检索爆款 → 分析风格 → 生成宣传图
"""
import csv
import time
from pathlib import Path
from typing import List, Dict
from PIL import Image

from config import OUTPUT_DIR, TOP_K_RETRIEVAL, COLLECTION_NAME
from embedding import EmbeddingGenerator
from retrieval import BestsellerRetriever
from retrieval_wrapper import RetrievalWrapper, create_retrieval_wrapper
from image_gen import ImageGenerator
from utils import save_image


class FashionImagePipeline:
    """时尚产品图像生成流水线"""

    def __init__(self):
        """初始化流水线"""
        print("=" * 60)
        print("Nano Banana + Milvus 电商生图流水线")
        print("=" * 60)

        # 使用检索包装器（Milvus 向量检索 + 循环检索状态机）
        self.retriever = create_retrieval_wrapper()
        self.embed_gen = EmbeddingGenerator()
        self.image_gen = ImageGenerator()
        self.tfidf = None

    def _check_database_ready(self) -> bool:
        """检查数据库是否已初始化"""
        try:
            # 混合检索器内部有 BestsellerRetriever
            milvus_retriever = self.retriever.milvus_retriever
            if not milvus_retriever.has_collection():
                return False
            stats = milvus_retriever.get_collection_stats()
            return stats.get('row_count', 0) > 0
        except Exception:
            return False

    def _ensure_database_ready(self):
        """确保数据库已初始化（只初始化一次）"""
        if self._check_database_ready():
            milvus_retriever = self.retriever.milvus_retriever
            stats = milvus_retriever.get_collection_stats()
            print(f"\n[OK] 数据库已就绪，包含 {stats['row_count']} 条记录")
            # 加载 TF-IDF 向量化器
            products, _ = self.embed_gen.load_products()
            self.tfidf = self.embed_gen.build_tfidf_vectorizer(products)
            return

        print("\n数据库未初始化，开始初始化...")
        self._init_database()

    def _init_database(self, overwrite: bool = False):
        """初始化向量数据库"""
        print("\n" + "=" * 60)
        print("初始化向量数据库")
        print("=" * 60)

        # 检查数据文件
        csv_path = Path("products.csv")
        image_dir = Path("images")

        if not csv_path.exists():
            raise FileNotFoundError(f"找不到 {csv_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"找不到 {image_dir} 目录")

        # 统计图片
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        print(f"找到 {len(image_files)} 张商品图片")

        # 创建 Collection（通过 Milvus 检索器）
        milvus_retriever = self.retriever.milvus_retriever
        milvus_retriever.create_collection(overwrite=overwrite)

        # 生成嵌入向量
        products, dense_vectors, sparse_vectors, tfidf = \
            self.embed_gen.process_all_embeddings()

        # 插入数据库
        milvus_retriever.insert_products(products, dense_vectors, sparse_vectors)

        # 保存 TF-IDF
        self.tfidf = tfidf

        stats = milvus_retriever.get_collection_stats()
        print(f"\n[OK] 数据库初始化完成! 共 {stats['row_count']} 条记录")

    def process_new_product(
        self,
        new_product_id: str,
        save_output: bool = True
    ) -> Dict:
        """
        处理单个新品: 检索 → 分析 → 生成

        Args:
            new_product_id: 新品 ID (如 "NEW001")
            save_output: 是否保存输出图片

        Returns:
            处理结果字典
        """
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "电商 AI 生图流水线" + " " * 25 + "║")
        print("║" + "═" * 58 + "║")
        print(f"║  新品ID: {new_product_id}                              ║")
        print("╚" + "═" * 58 + "╝")

        # 加载新品数据
        print("\n[1/5] 加载新品数据...")
        new_products = self.embed_gen.load_new_products()
        new_product = next(
            (p for p in new_products if p["new_id"] == new_product_id),
            None
        )

        if not new_product:
            print(f"  ✗ 找不到新品 {new_product_id}")
            return {"new_id": new_product_id, "success": False, "error": "未找到"}

        print(f"  ✓ 品类: {new_product['category']}")
        print(f"  ✓ 风格: {new_product['style']}")
        print(f"  ✓ 季节: {new_product.get('season', 'N/A')}")
        print(f"  ✓ 场景: {new_product.get('prompt_hint', 'N/A')}")

        try:
            # 编码新品
            print("\n[2/5] 编码新品向量...")
            query_dense, query_sparse, new_img = self.embed_gen.encode_new_product(
                new_product, self.tfidf
            )
            print(f"  ✓ Dense向量: {len(query_dense)}维")
            print(f"  ✓ Sparse向量: {len(query_sparse)}个非零项")

            # 检索相似爆款（循环检索 + 查询重写 + 质量评估）
            print("\n[3/5] 检索相似爆款...")
            retrieved = self.retriever.retrieve_similar_bestsellers(
                query_dense=query_dense.tolist(),
                query_sparse=query_sparse,
                category=new_product["category"],
                top_k=TOP_K_RETRIEVAL,
                enable_cycle=True,                      # 启用循环检索状态机
                query_category=new_product.get("category", ""),  # 用于质量评估
                query_style=new_product.get("style", ""),        # 用于质量评估
                query_season=new_product.get("season", ""),      # 用于质量评估
                query_scene_hint=new_product.get("prompt_hint", "")  # 用于质量评估
            )

            if not retrieved:
                print("  ✗ 未找到相似爆款")
                return {
                    "new_id": new_product_id,
                    "success": False,
                    "error": "未找到相似爆款"
                }

            # 提取参考图片
            ref_images = [r["image"] for r in retrieved if r["image"]]
            print(f"  ✓ 检索完成: 获得 {len(retrieved)} 个参考商品")

            # 分析风格并生成图片
            print("\n[4/5] 分析风格并生成宣传图...")
            print(f"  模型: FLUX.2 Klein")
            style_prompt, generated = self.image_gen.process_single_product(
                new_product_image=new_img,
                reference_images=ref_images,
                scene_hint=new_product.get("prompt_hint", "")
            )

            if not generated:
                print("  ✗ 图片生成失败")
                return {
                    "new_id": new_product_id,
                    "success": False,
                    "error": "图片生成失败"
                }

            # 保存结果
            if save_output:
                print("\n[5/5] 保存结果文件...")
                self._save_output(
                    new_product_id, new_img, retrieved,
                    generated, style_prompt
                )

            print(f"\n  ✓ 完成: 生成 {len(generated)} 张图片")

            print("\n" + "╔" + "═" * 58 + "╗")
            print("║" + " " * 20 + "任务完成!" + " " * 24 + "║")
            print("╚" + "═" * 58 + "╝\n")

            return {
                "new_id": new_product_id,
                "category": new_product['category'],
                "style": new_product['style'],
                "retrieved_count": len(retrieved),
                "generated_count": len(generated),
                "style_prompt": style_prompt,
                "success": True
            }

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            return {
                "new_id": new_product_id,
                "success": False,
                "error": str(e)
            }

    def _save_output(
        self,
        new_id: str,
        new_img: Image.Image,
        retrieved: List[Dict],
        generated: List[Image.Image],
        style_prompt: str
    ):
        """保存输出文件"""
        output_dir = OUTPUT_DIR / new_id
        output_dir.mkdir(exist_ok=True)

        # 保存原图
        save_image(new_img, str(output_dir / f"{new_id}_original.png"))

        # 保存参考图
        for i, ref in enumerate(retrieved):
            if ref.get("image"):
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
        print(f"{'='*60}")

        success_count = sum(1 for r in results if r.get("success"))
        fail_count = len(results) - success_count

        print(f"  总计: {len(results)} 个")
        print(f"  成功: {success_count} 个")
        print(f"  失败: {fail_count} 个")

        if save:
            report_path = OUTPUT_DIR / "batch_report.csv"
            with open(report_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "new_id", "category", "style", "retrieved_count",
                    "generated_count", "success", "error", "style_prompt"
                ], extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  报告已保存: {report_path}")

    def run(
        self,
        new_product_ids: List[str] = None,
        save_report: bool = True
    ) -> List[Dict]:
        """
        运行流水线

        Args:
            new_product_ids: 新品 ID 列表，None 则处理全部
            save_report: 是否保存汇总报告

        Returns:
            处理结果列表
        """
        # 确保数据库已初始化
        self._ensure_database_ready()

        # 获取要处理的新品列表
        if new_product_ids is None:
            new_products = self.embed_gen.load_new_products()
            new_product_ids = [p["new_id"] for p in new_products]

        if not new_product_ids:
            print("\n没有需要处理的新品")
            return []

        # 批量处理
        results = []
        total = len(new_product_ids)

        print(f"\n开始处理 {total} 个新品...")

        for i, pid in enumerate(new_product_ids, 1):
            print(f"\n[{i}/{total}] {pid}")
            result = self.process_new_product(
                pid,
                save_output=True
            )
            if result:
                results.append(result)

            # 礼貌延迟
            if i < total:
                time.sleep(1)

        # 生成汇总报告
        if len(results) > 1 and save_report:
            self._generate_summary(results)

        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Nano Banana + Milvus 电商 AI 生图流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 处理所有新品
  python main.py --process NEW001   # 处理单个新品
  python main.py --ids NEW001 NEW002 # 处理指定新品
  python main.py --reinit           # 强制重新初始化数据库
        """
    )

    parser.add_argument("--process", type=str, help="处理指定新品 ID")
    parser.add_argument("--ids", nargs="+", help="处理指定新品 ID 列表")
    parser.add_argument("--reinit", action="store_true", help="强制重新初始化数据库")

    args = parser.parse_args()

    pipeline = FashionImagePipeline()

    # 强制重新初始化
    if args.reinit:
        pipeline._init_database(overwrite=True)

    # 处理单个新品
    if args.process:
        pipeline._ensure_database_ready()
        pipeline.process_new_product(args.process)

    # 处理指定列表
    elif args.ids:
        pipeline.run(new_product_ids=args.ids)

    # 处理所有新品（默认）
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
