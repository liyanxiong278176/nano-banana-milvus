"""
爆款检索模块 - Milvus 数据库 + 混合检索 + 循环检索 + 质量评估
"""
import sys
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from openai import OpenAI
import httpx

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from config import (
    MILVUS_URI, COLLECTION_NAME, EMBED_DIM,
    MIN_SALES_COUNT, IMAGE_DIR, OPENROUTER_API_KEY, LIGHT_LLM_MODEL, LLM_MODEL
)
from utils import load_image, image_to_uri

# 超时配置 (秒)
API_TIMEOUT = 300  # 5 分钟 - LLM 质量评估请求超时时间


class BestsellerRetriever:
    """爆款商品检索器 - 集成数据库管理和混合检索"""

    def __init__(self, uri: str = MILVUS_URI):
        """
        初始化检索器

        Args:
            uri: Milvus 连接 URI
        """
        self.client = MilvusClient(uri=uri)
        self.collection_name = COLLECTION_NAME

    # ==================== 数据库管理 ====================

    def has_collection(self) -> bool:
        """检查 Collection 是否存在"""
        return self.client.has_collection(self.collection_name)

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取 Collection 统计信息"""
        return self.client.get_collection_stats(self.collection_name)

    def create_collection(self, overwrite: bool = False):
        """
        创建混合向量 Collection (Dense + Sparse + Scalar)

        Args:
            overwrite: 是否覆盖已存在的 Collection
        """
        if self.has_collection():
            if overwrite:
                self.client.drop_collection(self.collection_name)
                print(f"已删除旧 Collection: {self.collection_name}")
            else:
                print(f"Collection 已存在: {self.collection_name}")
                return

        # 定义 Schema
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 主键
        schema.add_field("id", DataType.INT64, is_primary=True)

        # 标量字段
        schema.add_field("product_id", DataType.VARCHAR, max_length=20)
        schema.add_field("category", DataType.VARCHAR, max_length=50)
        schema.add_field("color", DataType.VARCHAR, max_length=50)
        schema.add_field("style", DataType.VARCHAR, max_length=50)
        schema.add_field("season", DataType.VARCHAR, max_length=50)
        schema.add_field("sales_count", DataType.INT64)
        schema.add_field("description", DataType.VARCHAR, max_length=500)
        schema.add_field("price", DataType.FLOAT)

        # 向量字段
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        # 配置索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="FLAT",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 创建 Collection
        self.client.create_collection(
            self.collection_name,
            schema=schema,
            index_params=index_params
        )

        print(f"Collection 创建成功: {self.collection_name}")

    def insert_products(
        self,
        products: List[Dict[str, Any]],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[int, float]]
    ):
        """
        插入商品数据

        Args:
            products: 商品元数据列表
            dense_vectors: 图片稠密向量列表
            sparse_vectors: 文本稀疏向量列表
        """
        rows = []
        for i, p in enumerate(products):
            rows.append({
                "product_id": p.get("product_id", f"SKU{i:03d}"),
                "category": p.get("category", ""),
                "color": p.get("color", ""),
                "style": p.get("style", ""),
                "season": p.get("season", ""),
                "sales_count": int(p.get("sales_count", 0)),
                "description": p.get("description", ""),
                "price": float(p.get("price", 0.0)),
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
            })

        print(f"准备插入 {len(rows)} 条数据")
        insert_result = self.client.insert(self.collection_name, rows)

        # 持久化
        self.client.flush(self.collection_name)

        stats = self.get_collection_stats()
        print(f"已插入 {stats.get('row_count', 0)} 条商品数据")

    # ==================== 混合检索 ====================

    def _hybrid_search(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        filter_expr: str,
        top_k: int = 3,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        混合向量检索 (Dense + Sparse + Scalar Filter + RRF)

        Args:
            query_dense: 稠密查询向量
            query_sparse: 稀疏查询向量
            filter_expr: 标量过滤表达式
            top_k: 返回结果数量
            rrf_k: RRF 融合参数

        Returns:
            检索结果列表
        """
        # Dense 向量检索请求
        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=20,
            expr=filter_expr,
        )

        # Sparse 向量检索请求
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=20,
            expr=filter_expr,
        )

        # 执行混合搜索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=rrf_k),
            limit=top_k,
            output_fields=[
                "product_id", "category", "color", "style", "season",
                "sales_count", "description", "price"
            ],
        )

        return results[0] if results else []

    def _single_retrieve(
        self,
        query_dense: List[float],
        query_sparse: dict,
        category: str,
        min_sales: int = MIN_SALES_COUNT,
        top_k: int = 3,
        min_similarity: float = 0.5,
        max_results: int = 6
    ) -> List[Dict]:
        """
        单次混合检索（原始检索逻辑，重命名自 retrieve_similar_bestsellers）

        【保留原有逻辑不变】作为循环检索的核心函数

        Args:
            query_dense: 稠密查询向量
            query_sparse: 稀疏查询向量
            category: 品类筛选
            min_sales: 最低销量
            top_k: 期望返回数量（最终结果可能少于这个值）
            min_similarity: 最低相似度阈值（RRF距离，越小越相关，0-1之间）
            max_results: 最大返回数量

        Returns:
            检索结果列表，包含商品信息和图片（数量可变，取决于相关性和去重结果）
        """
        # 构建过滤表达式
        filter_expr = f'category == "{category}" and sales_count > {min_sales}'

        print(f"\n执行混合检索...")
        print(f"  品类: {category}")
        print(f"  最低销量: {min_sales}")
        print(f"  期望返回: {top_k} 张")

        # 执行检索，获取更多候选结果以便去重和过滤
        candidate_count = max(top_k * 4, 15)  # 获取更多候选
        results = self._hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            filter_expr=filter_expr,
            top_k=candidate_count
        )

        # 过滤和去重
        filtered_results = []
        seen_product_bases = set()  # 用于检测同款不同色

        for hit in results:
            # RRF 距离过滤（越小越相关，范围约0-1）
            if hit["distance"] > min_similarity:
                continue

            entity = hit["entity"]
            product_id = entity["product_id"]

            # 去重逻辑：检查产品基础ID（避免同款不同色）
            product_base = product_id.rsplit('_', 1)[0] if '_' in product_id else product_id

            if product_base in seen_product_bases:
                print(f"  跳过 {product_id}: 已有同系列产品 {product_base}")
                continue

            seen_product_bases.add(product_base)
            filtered_results.append(hit)

            # 达到最大数量时停止
            if len(filtered_results) >= max_results:
                break

        print(f"\n找到 {len(filtered_results)} 个相关且不同的爆款:")
        print("-" * 60)

        # 加载图片并整理结果
        retrieved_with_images = []
        for hit in filtered_results:
            entity = hit["entity"]
            product_id = entity["product_id"]

            # 加载参考图片
            try:
                img_path = IMAGE_DIR / f"{product_id}.jpg"
                ref_img = load_image(str(img_path))
            except FileNotFoundError:
                ref_img = None
                print(f"  警告: 图片未找到 {product_id}.jpg")
                continue

            result = {
                **entity,
                "score": hit["distance"],
                "image": ref_img
            }
            retrieved_with_images.append(result)

            # 打印结果信息
            print(f"  {entity['product_id']} | {entity['category']} | "
                  f"{entity['color']} | {entity['style']}")
            print(f"    销量: {entity['sales_count']} | 价格: ${entity['price']:.1f} | "
                  f"相似度: {hit['distance']:.4f}")
            print("-" * 60)

        # ==================== 【新增】降级逻辑 ====================
        # 如果找不到结果，尝试只按品类检索（去掉风格和销量限制）
        if not retrieved_with_images and category:
            print(f"\n  [降级] 未找到匹配商品，尝试只按品类检索: {category}")

            # 降低销量阈值，去掉风格限制
            fallback_filter = f'category == "{category}"'
            results = self._hybrid_search(
                query_dense=query_dense,
                query_sparse=query_sparse,
                filter_expr=fallback_filter,
                top_k=top_k * 2
            )

            for hit in results:
                entity = hit["entity"]
                product_id = entity["product_id"]

                try:
                    img_path = IMAGE_DIR / f"{product_id}.jpg"
                    ref_img = load_image(str(img_path))
                except FileNotFoundError:
                    continue

                result = {
                    **entity,
                    "score": hit["distance"],
                    "image": ref_img,
                    "fallback": True  # 标记为降级结果
                }
                retrieved_with_images.append(result)

                if len(retrieved_with_images) >= max_results:
                    break

            if retrieved_with_images:
                print(f"\n  [降级成功] 找到 {len(retrieved_with_images)} 个 {category} 商品")

        return retrieved_with_images

    # ==================== 循环检索（新增） ====================

    def retrieve_similar_bestsellers(
        self,
        query_dense: List[float],
        query_sparse: dict,
        category: str,
        min_sales: int = MIN_SALES_COUNT,
        top_k: int = 3,
        min_similarity: float = 0.5,
        max_results: int = 6,
        enable_cycle: bool = True,
        query_category: str = "",
        query_style: str = "",
        query_season: str = "",
        query_scene_hint: str = ""
    ) -> List[Dict]:
        """
        智能循环检索（新增改造版本）

        循环逻辑：
        1. 第1次检索：原始过滤条件
        2. 评分：调用 RetrievalQualityJudge 评分
        3. 判断：平均分≥7分 → 直接返回结果
        4. 否则：第1次重写 → 第2次检索 → 评分
        5. 若仍<7分：第2次重写 → 第3次检索 → 评分
        6. 最多循环3次，返回最后一次结果（不终止任务）

        Args:
            query_dense: 稠密查询向量
            query_sparse: 稀疏查询向量
            category: 品类筛选
            min_sales: 最低销量
            top_k: 期望返回数量
            min_similarity: 最低相似度阈值
            max_results: 最大返回数量
            enable_cycle: 是否启用循环检索（默认True）
            query_category: 新品品类（用于质量评估）
            query_style: 新品风格（用于质量评估）
            query_season: 新品季节（用于质量评估）
            query_scene_hint: 新品场景提示（用于质量评估）

        Returns:
            检索结果列表，包含商品信息和图片
        """
        # 显示查询上下文
        query_context_parts = []
        if query_category:
            query_context_parts.append(f"品类: {query_category}")
        if query_style:
            query_context_parts.append(f"风格: {query_style}")
        if query_season:
            query_context_parts.append(f"季节: {query_season}")
        if query_scene_hint:
            query_context_parts.append(f"场景: {query_scene_hint}")
        query_context = " | ".join(query_context_parts) if query_context_parts else "通用查询"

        # 定义最大轮数（需要在 enable_cycle 判断之前定义）
        max_rounds = 3

        if not enable_cycle:
            # 禁用循环检索，使用原始单次检索逻辑
            print("\n" + "=" * 60)
            print("【单次检索模式】循环检索已禁用")
            print(f"查询条件: {query_context}")
            print("=" * 60)
            return self._single_retrieve(
                query_dense=query_dense,
                query_sparse=query_sparse,
                category=category,
                min_sales=min_sales,
                top_k=top_k,
                min_similarity=min_similarity,
                max_results=max_results
            )

        print("\n" + "=" * 60)
        print("【循环检索状态机】已启用")
        print("=" * 60)
        print(f"查询条件: {query_context}")
        print(f"目标数量: {top_k} | 最大结果: {max_results}")
        print(f"质量阈值: 7.0/10 | 最大轮次: {max_rounds}")
        print("=" * 60)

        # 初始化质量评估器
        judge = RetrievalQualityJudge()

        # 循环检索变量
        best_results = []
        best_score = 0
        best_round = 0  # 记录最佳结果来自第几轮

        for round_num in range(1, max_rounds + 1):
            print(f"\n{'═'*60}")
            print(f"  第 {round_num} 轮检索")
            print(f"{'═'*60}")

            # 第1轮：使用原始条件
            if round_num == 1:
                current_category = category
                current_min_sales = min_sales
                print(f"  检索策略: 原始条件")
                print(f"  过滤条件: category='{current_category}', sales_count > {current_min_sales}")

            # 第2轮：查询重写（第1次重写）
            elif round_num == 2:
                # 获取上一轮的评分
                if best_results:
                    prev_scores = judge.score_retrieval_quality(
                        retrieved_results=best_results,
                        query_category=query_category,
                        query_style=query_style,
                        query_season=query_season,
                        query_scene_hint=query_scene_hint
                    )
                else:
                    prev_scores = {"average": 0, "category_match": 0, "style_match": 0}

                # 显示重写原因
                print(f"  上一轮评分: {prev_scores.get('average', 0):.1f}/10")
                rewrite_reasons = []
                if prev_scores.get('category_match', 10) < 6:
                    rewrite_reasons.append(f"品类匹配低({prev_scores.get('category_match', 0)}/10)")
                if prev_scores.get('style_match', 10) < 6:
                    rewrite_reasons.append(f"风格匹配低({prev_scores.get('style_match', 0)}/10)")
                if prev_scores.get('average', 0) < 7:
                    rewrite_reasons.append(f"整体未达标(需要7.0+)")

                # 执行查询重写
                original_filter = f'category == "{category}" and sales_count > {min_sales}'
                new_filter = query_rewrite(original_filter, prev_scores, rewrite_round=1)

                # 解析重写后的参数
                current_category = category  # 默认保持
                current_min_sales = min_sales

                if 'sales_count > 1000' in new_filter:
                    current_min_sales = 1000
                if 'sales_count > 500' in new_filter:
                    current_min_sales = 500
                # 第1轮重写可能放宽品类
                if 'category like "dress%"' in new_filter or 'category contains "dress"' in new_filter:
                    current_category = "dress"  # 父类

                print(f"  检索策略: 查询重写 (第1次)")
                print(f"  重写原因: {', '.join(rewrite_reasons) if rewrite_reasons else '质量未达标'}")
                print(f"  过滤条件: category='{current_category}', sales_count > {current_min_sales}")

            # 第3轮：查询重写（第2次重写）
            else:  # round_num == 3
                current_category = ""  # 去掉品类限制
                current_min_sales = 500  # 最低销量
                print(f"  检索策略: 查询重写 (第2次 - 最大化召回)")
                print(f"  过滤条件: 无品类限制, sales_count > {current_min_sales}")

            print(f"  ──" + "─" * 56)

            # 执行检索
            if current_category:
                results = self._single_retrieve(
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    category=current_category,
                    min_sales=current_min_sales,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    max_results=max_results
                )
            else:
                # 无品类限制：直接调用底层混合搜索
                filter_expr = f"sales_count > {current_min_sales}"
                candidate_count = max(top_k * 4, 15)
                raw_results = self._hybrid_search(
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    filter_expr=filter_expr,
                    top_k=candidate_count
                )

                # 复用去重和图片加载逻辑
                results = self._process_raw_results(raw_results, min_similarity, max_results)

            if not results:
                print(f"  [!] 第 {round_num} 轮检索无结果")
                continue

            print(f"  ✓ 检索到 {len(results)} 个结果")

            # 质量评分
            print(f"\n  【质量评估】")
            scores = judge.score_retrieval_quality(
                retrieved_results=results,
                query_category=query_category,
                query_style=query_style,
                query_season=query_season,
                query_scene_hint=query_scene_hint
            )

            avg_score = scores.get("average", 0)
            cat_score = scores.get('category_match', 0)
            style_score = scores.get('style_match', 0)
            scene_score = scores.get('scene_match', 0)
            attr_score = scores.get('attribute_match', 0)

            # 评分可视化
            score_bar = "█" * int(avg_score) + "░" * (10 - int(avg_score))
            print(f"    总分: {avg_score:.1f}/10 [{score_bar}]")
            print(f"      ├─ 品类匹配: {cat_score}/10")
            print(f"      ├─ 风格匹配: {style_score}/10")
            print(f"      ├─ 场景匹配: {scene_score}/10")
            print(f"      └─ 属性匹配: {attr_score}/10")

            # 更新最佳结果
            if avg_score > best_score:
                best_score = avg_score
                best_results = results
                best_round = round_num
                print(f"    [*] 已更新为最佳结果")

            # 判断是否满足阈值
            if avg_score >= 7.0:
                print(f"\n  [OK] 质量达标 (>=7.0)，提前返回第 {round_num} 轮结果")
                break

            print(f"    [!] 质量未达标 (<7.0)，继续下一轮...")

        # 循环结束
        print(f"\n{'═'*60}")
        print(f"  【循环检索结束】")
        print(f"{'═'*60}")
        print(f"  执行轮数: {best_round}/{max_rounds}")
        print(f"  最终评分: {best_score:.1f}/10")
        print(f"  返回结果: {len(best_results)} 个商品")
        print(f"{'='*60}")

        return best_results if best_results else []

    def _process_raw_results(
        self,
        raw_results: List[Dict],
        min_similarity: float,
        max_results: int
    ) -> List[Dict]:
        """
        处理原始检索结果（去重+加载图片）

        【新增辅助函数】复用原有去重和图片加载逻辑
        """
        filtered_results = []
        seen_product_bases = set()

        for hit in raw_results:
            if hit["distance"] > min_similarity:
                continue

            entity = hit["entity"]
            product_id = entity["product_id"]

            # 去重逻辑
            product_base = product_id.rsplit('_', 1)[0] if '_' in product_id else product_id
            if product_base in seen_product_bases:
                continue

            seen_product_bases.add(product_base)
            filtered_results.append(hit)

            if len(filtered_results) >= max_results:
                break

        # 加载图片
        retrieved_with_images = []
        for hit in filtered_results:
            entity = hit["entity"]
            product_id = entity["product_id"]

            try:
                img_path = IMAGE_DIR / f"{product_id}.jpg"
                ref_img = load_image(str(img_path))
            except FileNotFoundError:
                continue

            result = {
                **entity,
                "score": hit["distance"],
                "image": ref_img
            }
            retrieved_with_images.append(result)

        return retrieved_with_images

    def format_retrieval_for_prompt(
        self,
        retrieved: List[Dict]
    ) -> Tuple[List[Image.Image], str]:
        """
        格式化检索结果用于 Prompt 生成

        Args:
            retrieved: 检索结果列表

        Returns:
            (图片列表, 文字描述)
        """
        images = []
        description = "畅销时尚产品分析：\n\n"

        for i, item in enumerate(retrieved, 1):
            if item.get("image"):
                images.append(item["image"])

            description += f"{i}. 产品ID: {item['product_id']}\n"
            description += f"   风格: {item['style']}, 颜色: {item['color']}\n"
            description += f"   描述: {item['description']}\n\n"

        return images, description


# ==================== 检索质量评估器（新增类） ====================

class RetrievalQualityJudge:
    """
    检索结果质量评估器

    使用轻量级 LLM 对检索结果进行多维度评分（0-10分制）

    【模型回退策略】
    1. 优先使用 LIGHT_LLM_MODEL（轻量级，免费）
    2. 如果不可用，自动回退到 LLM_MODEL（主模型）
    3. 如果都不可用，使用默认兜底评分（6分）
    """

    # 品类父类映射（用于查询重写）
    CATEGORY_PARENT_MAP = {
        "midi_dress": "dress",
        "maxi_dress": "dress",
        "mini_dress": "dress",
        "skirt": "dress",
        "top": "clothing",
        "pants": "clothing",
    }

    def __init__(self, model: str = None):
        """
        初始化质量评估器

        Args:
            model: 指定评估模型，None 则自动选择（优先 LIGHT_LLM_MODEL）
        """
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=httpx.Timeout(API_TIMEOUT, connect=60)
        )

        # 模型选择逻辑：支持回退
        if model:
            self.model = model
        else:
            # 默认使用轻量级模型，如果不可用会在调用时回退
            self.model = LIGHT_LLM_MODEL
            self.fallback_model = LLM_MODEL  # 回退模型

    def score_retrieval_quality(
        self,
        retrieved_results: List[Dict],
        query_category: str = "",
        query_style: str = "",
        query_season: str = "",
        query_scene_hint: str = ""
    ) -> Dict[str, float]:
        """
        对检索结果进行多维度质量评分

        评分维度（0-10分）：
        - category_match: 品类匹配度（检索结果是否与查询品类相关）
        - style_match: 风格匹配度（风格标签是否一致）
        - scene_match: 场景匹配度（整体视觉风格是否符合预期）
        - attribute_match: 属性匹配度（季节、颜色等属性的综合匹配）
        - average: 平均分

        Args:
            retrieved_results: 检索结果列表（来自 retrieve_similar_bestsellers）
            query_category: 查询品类
            query_style: 查询风格
            query_season: 查询季节
            query_scene_hint: 场景提示

        Returns:
            评分字典 {"category_match": 8, "style_match": 7, ...}
        """
        if not retrieved_results:
            return {
                "category_match": 0,
                "style_match": 0,
                "scene_match": 0,
                "attribute_match": 0,
                "average": 0.0
            }

        # 构建查询上下文
        query_context = self._build_query_context(
            query_category, query_style, query_season, query_scene_hint
        )

        # 构建检索结果描述
        results_summary = self._build_results_summary(retrieved_results)

        # 构建评分 Prompt
        score_prompt = f"""你是一位专业的电商检索质量评估专家。

【查询需求】
{query_context}

【检索结果】
{results_summary}

请评估检索结果与查询需求的相关性，对以下维度进行评分（0-10分）：

1. **category_match**（品类匹配度）：检索结果的品类是否与查询品类相关
2. **style_match**（风格匹配度）：风格标签（如优雅、休闲）是否匹配
3. **scene_match**（场景匹配度）：整体视觉风格是否适合该季节/场景
4. **attribute_match**（属性匹配度）：季节、价格区间等属性的综合匹配度

**评分标准**：
- 9-10分：完美匹配，高度相关
- 7-8分：良好匹配，基本符合需求
- 5-6分：一般匹配，部分相关
- 3-4分：匹配度较低，相关性弱
- 0-2分：完全不匹配，无相关性

只输出JSON格式，例如：
{{"category_match": 8, "style_match": 7, "scene_match": 6, "attribute_match": 7}}
"""

        # 如果有检索结果图片，加入图片进行视觉评估
        content = []
        for result in retrieved_results[:3]:  # 最多评估前3张
            if result.get("image"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_uri(result["image"])}
                })

        content.append({"type": "text", "text": score_prompt})

        # 尝试评分（支持模型回退）
        models_to_try = [self.model]
        if hasattr(self, 'fallback_model'):
            models_to_try.append(self.fallback_model)

        for model_idx, current_model in enumerate(models_to_try):
            try:
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=300,
                    temperature=0.3,  # 低温度保证评分稳定
                )

                content_text = response.choices[0].message.content

                # 解析 JSON
                import re
                scores = self._parse_score_json(content_text)

                if scores:
                    # 计算平均分
                    avg_score = sum(scores.values()) / len(scores)
                    scores["average"] = round(avg_score, 1)
                    scores["is_fallback"] = False
                    scores["model_used"] = current_model

                    if model_idx > 0:
                        print(f"  使用回退模型评分: {current_model}")
                    else:
                        print(f"  使用主模型评分: {current_model}")

                    return scores

            except Exception as e:
                print(f"  模型 {current_model} 评估失败: {e}")
                # 如果是最后一个模型也失败了，继续到兜底逻辑
                if model_idx < len(models_to_try) - 1:
                    print(f"  尝试回退模型...")
                    continue
                else:
                    print(f"  所有模型均失败，使用默认评分")
                    break

        # 默认评分（异常兜底）
        return {
            "category_match": 6,
            "style_match": 6,
            "scene_match": 6,
            "attribute_match": 6,
            "average": 6.0,
            "is_fallback": True
        }

    def _build_query_context(
        self,
        category: str,
        style: str,
        season: str,
        scene_hint: str
    ) -> str:
        """构建查询上下文描述"""
        context_parts = []
        if category:
            context_parts.append(f"品类：{category}")
        if style:
            context_parts.append(f"风格：{style}")
        if season:
            context_parts.append(f"季节：{season}")
        if scene_hint:
            context_parts.append(f"场景：{scene_hint}")

        return "\n".join(context_parts) if context_parts else "通用时尚商品查询"

    def _build_results_summary(self, results: List[Dict]) -> str:
        """构建检索结果摘要"""
        summary_lines = []
        for i, r in enumerate(results[:5], 1):
            line = f"{i}. {r.get('product_id', 'N/A')} | "
            line += f"品类:{r.get('category', 'N/A')} | "
            line += f"风格:{r.get('style', 'N/A')} | "
            line += f"季节:{r.get('season', 'N/A')} | "
            line += f"销量:{r.get('sales_count', 0)}"
            summary_lines.append(line)

        return "\n".join(summary_lines)

    def _parse_score_json(self, content_text: str) -> Optional[Dict[str, float]]:
        """
        解析评分 JSON（复用 image_gen.py 中的解析逻辑）

        Args:
            content_text: LLM 返回的文本

        Returns:
            解析后的评分字典，失败返回 None
        """
        import json
        import re

        # 方法1: 直接解析整个响应
        try:
            scores = json.loads(content_text.strip())
            if isinstance(scores, dict) and all(isinstance(v, (int, float)) for v in scores.values()):
                return scores
        except json.JSONDecodeError:
            pass

        # 方法2: 使用正则匹配 JSON 对象
        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', content_text, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group())
                if isinstance(scores, dict) and all(isinstance(v, (int, float)) for v in scores.values()):
                    return scores
            except json.JSONDecodeError:
                pass

        return None


# ==================== 查询重写函数（新增） ====================

def query_rewrite(
    original_filter: str,
    scores: Dict[str, float],
    rewrite_round: int
) -> str:
    """
    根据质量评分自动重写查询条件

    重写规则：
    - 第1轮重写（rewrite_round=1）：
        * category_match低 → 放宽品类到父类
        * style_match低 → 去掉风格过滤
        * sales_count阈值从1500降到1000
    - 第2轮重写（rewrite_round=2）：
        * 去掉品类、风格、季节过滤
        * 仅保留sales_count>500
        * 仅做核心视觉向量匹配

    Args:
        original_filter: 原始过滤表达式（如 'category == "midi_dress" and sales_count > 1500'）
        scores: 评分字典（来自 RetrievalQualityJudge）
        rewrite_round: 当前重写轮次（1或2）

    Returns:
        新的过滤表达式
    """
    if rewrite_round == 1:
        # 第1轮重写：根据低分维度放宽条件
        new_parts = []

        # 检查品类匹配
        category_match = scores.get("category_match", 10)
        if category_match < 6:
            # 品类匹配度低，尝试放宽到父类
            # 解析原始品类
            import re
            category_match_val = re.search(r'category\s*==\s*"([^"]+)"', original_filter)
            if category_match_val:
                original_category = category_match_val.group(1)
                parent_category = RetrievalQualityJudge.CATEGORY_PARENT_MAP.get(original_category)
                if parent_category and parent_category != original_category:
                    new_parts.append(f'category like "{parent_category}%"')
                else:
                    # 没有父类，去掉品类限制
                    pass
            else:
                new_parts.append('category like "dress%"' if 'dress' in original_filter else '')
        else:
            # 品类匹配度高，保留原品类
            import re
            category_match_val = re.search(r'category\s*==\s*"([^"]+)"', original_filter)
            if category_match_val:
                new_parts.append(f'category == "{category_match_val.group(1)}"')

        # 风格匹配度低时，不添加风格过滤（原始过滤可能也没风格）

        # 销量阈值降低
        new_parts.append("sales_count > 1000")

        return " and ".join([p for p in new_parts if p])

    elif rewrite_round == 2:
        # 第2轮重写：只保留最低销量限制，最大化召回
        return "sales_count > 500"

    else:
        # 未知轮次，返回原过滤
        return original_filter


if __name__ == "__main__":
    # 测试检索器
    retriever = BestsellerRetriever()

    # 检查状态
    if retriever.has_collection():
        stats = retriever.get_collection_stats()
        print(f"Collection 存在，记录数: {stats.get('row_count', 0)}")
    else:
        print("Collection 不存在，请先运行 main.py 初始化数据库")
