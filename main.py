"""
主程序 - 电商 AI 生图流水线
完整流程: 加载数据 → 构建向量库 → 检索爆款 → 分析风格 → 生成宣传图
"""
import os
from pathlib import Path
from PIL import Image

from config import OUTPUT_DIR, TOP_K_RETRIEVAL
from milvus_db import FashionMilvusDB
from embedding import EmbeddingGenerator
from retrieval import BestsellerRetriever
from image_gen import ImageGenerator
from utils import save_image, display_comparison


class FashionImagePipeline:
    """时尚产品图像生成流水线"""

    def __init__(self):
        """初始化流水线"""
        print("=" * 60)
        print("Nano Banana + Milvus 电商生图流水线")
        print("=" * 60)

        self.milvus_db = FashionMilvusDB()
        self.embed_gen = EmbeddingGenerator()
        self.retriever = BestsellerRetriever(self.milvus_db)
        self.image_gen = ImageGenerator()

    def build_vector_database(self, overwrite: bool = False):
        """
        构建向量数据库

        Args:
            overwrite: 是否覆盖已存在的数据库
        """
        print("\n" + "=" * 60)
        print("构建向量数据库")
        print("=" * 60)

        # 创建 Collection
        self.milvus_db.create_collection(overwrite=overwrite)

        # 生成所有嵌入向量
        products, dense_vectors, sparse_vectors, tfidf = \
            self.embed_gen.process_all_embeddings()

        # 插入数据库
        self.milvus_db.insert_products(products, dense_vectors, sparse_vectors)

        # 保存 TF-IDF 向量化器供后续使用
        self.tfidf = tfidf

        print("\n向量数据库构建完成!")

    def process_new_product(
        self,
        new_product_id: str,
        save_output: bool = True,
        display: bool = True
    ) -> dict:
        """
        处理单个新品: 检索 → 分析 → 生成

        Args:
            new_product_id: 新品 ID (如 "NEW001")
            save_output: 是否保存输出图片
            display: 是否显示结果

        Returns:
            处理结果字典
        """
        print("\n" + "=" * 60)
        print(f"处理新品: {new_product_id}")
        print("=" * 60)

        # 加载新品数据
        new_products = self.embed_gen.load_new_products()
        new_product = next(
            (p for p in new_products if p["new_id"] == new_product_id),
            None
        )

        if not new_product:
            print(f"错误: 找不到新品 {new_product_id}")
            return None

        print(f"品类: {new_product['category']}")
        print(f"风格: {new_product['style']}")
        print(f"季节: {new_product['season']}")
        print(f"场景提示: {new_product['prompt_hint']}")

        # 编码新品
        query_dense, query_sparse, new_img = self.embed_gen.encode_new_product(
            new_product, self.tfidf
        )

        # 检索相似爆款
        retrieved = self.retriever.retrieve_similar_bestsellers(
            query_dense=query_dense.tolist(),
            query_sparse=query_sparse,
            category=new_product["category"],
            top_k=TOP_K_RETRIEVAL
        )

        if not retrieved:
            print("警告: 未找到相似爆款")
            return None

        # 提取参考图片
        ref_images = [r["image"] for r in retrieved if r["image"]]

        # 分析风格并生成图片
        style_prompt, generated = self.image_gen.process_single_product(
            new_product_image=new_img,
            reference_images=ref_images,
            scene_hint=new_product.get("prompt_hint", "")
        )

        if not generated:
            print("警告: 图片生成失败")
            return {
                "new_product": new_product,
                "new_image": new_img,
                "retrieved": retrieved,
                "style_prompt": style_prompt,
                "generated_images": []
            }

        # 保存结果
        result = {
            "new_product": new_product,
            "new_image": new_img,
            "retrieved": retrieved,
            "style_prompt": style_prompt,
            "generated_images": generated
        }

        if save_output:
            output_dir = OUTPUT_DIR / new_product_id
            output_dir.mkdir(exist_ok=True)

            # 保存新品原图
            save_image(new_img, str(output_dir / f"{new_product_id}_original.png"))

            # 保存参考图
            for i, ref in enumerate(retrieved):
                if ref["image"]:
                    save_image(
                        ref["image"],
                        str(output_dir / f"{new_product_id}_reference_{i+1}.png")
                    )

            # 保存生成图
            for i, gen_img in enumerate(generated):
                save_image(
                    gen_img,
                    str(output_dir / f"{new_product_id}_generated_{i+1}.png")
                )

            # 保存风格描述
            with open(output_dir / f"{new_product_id}_style_prompt.txt", "w", encoding="utf-8") as f:
                f.write(style_prompt)

            print(f"\n所有结果已保存到: {output_dir}")

        if display:
            self._display_results(result)

        return result

    def _display_results(self, result: dict):
        """显示处理结果"""
        print("\n" + "=" * 60)
        print("结果展示")
        print("=" * 60)

        print("\n[风格描述]")
        print(result["style_prompt"])

        print(f"\n[生成图片] 共 {len(result['generated_images'])} 张")

    def run_full_pipeline(self, new_product_ids: list = None):
        """
        运行完整流水线

        Args:
            new_product_ids: 要处理的新品 ID 列表，如果为 None 则处理所有
        """
        # 构建向量数据库
        self.build_vector_database(overwrite=False)

        # 获取所有新品
        new_products = self.embed_gen.load_new_products()

        if new_product_ids is None:
            new_product_ids = [p["new_id"] for p in new_products]

        # 处理每个新品
        results = []
        for pid in new_product_ids:
            result = self.process_new_product(pid)
            if result:
                results.append(result)

        print("\n" + "=" * 60)
        print(f"流水线完成! 共处理 {len(results)} 个新品")
        print("=" * 60)

        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="电商 AI 生图流水线")
    parser.add_argument("--build-db", action="store_true", help="构建向量数据库")
    parser.add_argument("--process", type=str, help="处理指定新品 ID")
    parser.add_argument("--batch", action="store_true", help="批量处理所有新品")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的数据库")

    args = parser.parse_args()

    pipeline = FashionImagePipeline()

    if args.build_db or args.batch:
        # 确保数据库已构建
        if not Path(pipeline.milvus_db.client._uri).exists() or args.overwrite:
            pipeline.build_vector_database(overwrite=args.overwrite)
        else:
            # 加载现有的 TF-IDF
            from embedding import EmbeddingGenerator
            products, _ = pipeline.embed_gen.load_products()
            pipeline.tfidf = pipeline.embed_gen.build_tfidf_vectorizer(products)

    if args.process:
        # 处理单个新品
        pipeline.process_new_product(args.process)

    elif args.batch:
        # 批量处理
        new_products = pipeline.embed_gen.load_new_products()
        product_ids = [p["new_id"] for p in new_products]
        pipeline.run_full_pipeline(product_ids)

    else:
        print("\n使用方法:")
        print("  python main.py --build-db          # 构建向量数据库")
        print("  python main.py --process NEW001    # 处理单个新品")
        print("  python main.py --batch             # 批量处理所有新品")
        print("  python main.py --build-db --overwrite  # 重建数据库")


if __name__ == "__main__":
    main()
