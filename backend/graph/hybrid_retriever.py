"""
混合检索器 - Milvus 向量检索 + Neo4j 图谱检索

实现双引擎混合检索策略，融合向量相似度和图谱关联度。
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image

from retrieval import BestsellerRetriever
try:
    from graph import FashionGraphRetriever
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("警告: Neo4j 模块未可用，仅使用 Milvus 检索")

from config import TOP_K_RETRIEVAL, MIN_SALES_COUNT
from utils import load_image


class HybridRetriever:
    """
    混合检索器 - 融合 Milvus 向量检索和 Neo4j 图谱检索

    检索策略：
    1. Milvus 向量检索：基于视觉相似度
    2. Neo4j 图谱检索：基于结构化关联
    3. 结果融合：RRF (Reciprocal Rank Fusion) 算法

    异常处理：
    - Neo4j 不可用时自动降级到纯 Milvus 检索
    """

    # 融合权重配置
    DEFAULT_MILVUS_WEIGHT = 0.6  # 向量检索权重
    DEFAULT_GRAPH_WEIGHT = 0.4   # 图谱检索权重

    def __init__(
        self,
        milvus_weight: float = DEFAULT_MILVUS_WEIGHT,
        graph_weight: float = DEFAULT_GRAPH_WEIGHT
    ):
        """
        初始化混合检索器

        Args:
            milvus_weight: Milvus 向量检索权重 (0-1)
            graph_weight: Neo4j 图谱检索权重 (0-1)
        """
        print(f"\n{'='*60}")
        print("【混合检索器初始化】")
        print(f"{'='*60}")

        # 归一化权重
        total = milvus_weight + graph_weight
        self.milvus_weight = milvus_weight / total if total > 0 else 0.5
        self.graph_weight = graph_weight / total if total > 0 else 0.5

        print(f"  融合权重配置: Milvus={self.milvus_weight:.2f}, Neo4j={self.graph_weight:.2f}")

        # 初始化 Milvus 检索器
        print(f"\n  [1/2] 初始化 Milvus 检索器...")
        self.milvus_retriever = BestsellerRetriever()
        try:
            milvus_ok = self.milvus_retriever.has_collection()
            print(f"       [OK] Milvus 检索器已就绪 ({'已连接' if milvus_ok else '集合未创建'})")
        except Exception as e:
            print(f"       [WARN] Milvus 检索器初始化警告: {e}")

        # 初始化 Neo4j 检索器（如果可用）
        self.graph_retriever: Optional[FashionGraphRetriever] = None
        self.neo4j_enabled = False

        print(f"\n  [2/2] 初始化 Neo4j 检索器...")
        if NEO4J_AVAILABLE:
            try:
                self.graph_retriever = FashionGraphRetriever()
                if self.graph_retriever.is_connected():
                    self.neo4j_enabled = True
                    stats = self.graph_retriever.get_graph_stats()
                    node_count = stats.get('node_count', 0)
                    rel_count = stats.get('relationship_count', 0)
                    print(f"       [OK] Neo4j 检索器已启用 (权重: {self.graph_weight:.2f})")
                    print(f"             图谱统计: {node_count} 节点, {rel_count} 关系")
                else:
                    print(f"       [FAIL] Neo4j 未连接，仅使用 Milvus 检索")
            except Exception as e:
                print(f"       [FAIL] Neo4j 初始化失败: {e}，仅使用 Milvus 检索")
        else:
            print(f"       [FAIL] Neo4j 模块不可用，仅使用 Milvus 检索")

        print(f"\n{'='*60}")
        print(f"[OK] 混合检索器初始化完成")
        print(f"     检索模式: {'Milvus + Neo4j 双引擎' if self.neo4j_enabled else 'Milvus 单引擎'}")
        print(f"{'='*60}\n")

    # ==================== 主检索接口 ====================

    def retrieve_similar_bestsellers(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        category: str = "",
        style: str = "",
        season: str = "",
        scene_hint: str = "",
        min_sales: int = MIN_SALES_COUNT,
        top_k: int = TOP_K_RETRIEVAL,
        min_similarity: float = 0.0,
        max_results: int = 6,
        enable_cycle: bool = True,
        query_category: str = "",
        query_style: str = "",
        query_season: str = "",
        query_scene_hint: str = "",
        enable_multi_hop: bool = True,
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        混合检索：融合 Milvus 向量检索和 Neo4j 图谱检索

        Args:
            query_dense: 稠密查询向量（Milvus）
            query_sparse: 稀疏查询向量（Milvus）
            category: 新品品类
            style: 新品风格
            season: 新品季节
            scene_hint: 场景提示
            min_sales: 最低销量
            top_k: 返回数量
            min_similarity: 相似度阈值（传递给 Milvus）
            max_results: 最大返回数量（传递给 Milvus 循环检索）
            enable_cycle: 是否启用 Milvus 循环检索
            query_category: 查询品类（用于质量评估）
            query_style: 查询风格（用于质量评估）
            query_season: 查询季节（用于质量评估）
            query_scene_hint: 查询场景提示（用于质量评估）
            enable_multi_hop: 是否启用多跳推理（默认True）
            max_hops: 最大跳数（默认3）

        Returns:
            融合后的检索结果列表
        """
        print(f"\n{'='*60}")
        print("【混合检索】Milvus + Neo4j")
        print(f"{'='*60}")
        print(f"  多跳推理: {'启用' if enable_multi_hop else '禁用'}")

        # ==================== 1. Milvus 向量检索 ====================
        print("\n[1/2] Milvus 向量检索...")
        print(f"      查询参数: category={category or 'All'}, min_sales={min_sales}, top_k={top_k * 2}")
        print(f"      循环检索: {'启用' if enable_cycle else '禁用'}")

        milvus_results = self.milvus_retriever.retrieve_similar_bestsellers(
            query_dense=query_dense,
            query_sparse=query_sparse,
            category=category,
            min_sales=min_sales,
            top_k=top_k * 2,  # 获取更多候选
            min_similarity=min_similarity,
            max_results=max_results,
            enable_cycle=enable_cycle,
            query_category=query_category,
            query_style=query_style,
            query_season=query_season,
            query_scene_hint=query_scene_hint
        )

        print(f"  [完成] Milvus 检索到 {len(milvus_results)} 个结果")
        if milvus_results:
            print(f"      Top 3: {', '.join([r.get('product_id', 'N/A') for r in milvus_results[:3]])}")

        # ==================== 2. Neo4j 图谱检索（支持多跳推理）====================
        graph_results = []
        if self.neo4j_enabled and self.graph_retriever:
            print("\n[2/2] Neo4j 图谱检索...")
            print(f"      查询参数: category={category or 'All'}, style={style or 'All'}, season={season or 'All'}")
            print(f"      场景提示: {scene_hint or query_scene_hint or 'None'}")
            try:
                graph_results = self.graph_retriever.retrieve_by_graph(
                    category=category,
                    style=style,
                    season=season,
                    scene_hint=scene_hint or query_scene_hint,
                    top_k=top_k * 2,
                    enable_multi_hop=enable_multi_hop,
                    max_hops=max_hops
                )
                print(f"  [完成] Neo4j 检索到 {len(graph_results)} 个结果")
                if graph_results:
                    print(f"      Top 3: {', '.join([r.get('product_id', 'N/A') for r in graph_results[:3]])}")
                    # 显示各策略检索结果数量
                    if enable_multi_hop:
                        hop_counts = {}
                        for r in graph_results:
                            hop = r.get('hop_count', 'unknown')
                            hop_counts[hop] = hop_counts.get(hop, 0) + 1
                        if hop_counts:
                            print(f"      跳数分布: {', '.join([f'{k}跳:{v}' for k, v in hop_counts.items()])}")
                    else:
                        strategy_counts = {}
                        for r in graph_results:
                            strategy = r.get('retrieval_strategy', 'unknown')
                            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                        if strategy_counts:
                            print(f"      策略分布: {', '.join([f'{k}:{v}' for k, v in strategy_counts.items()])}")
            except Exception as e:
                print(f"  [失败] Neo4j 检索失败: {e}")
        else:
            print("\n[2/2] Neo4j 图谱检索 (跳过 - 未启用)")

        # ==================== 3. 结果融合 ====================
        print(f"\n[3/3] 结果融合 (RRF算法)...")
        print(f"      融合权重: Milvus={self.milvus_weight:.2f}, Neo4j={self.graph_weight:.2f}")

        fused_results = self._fuse_results(
            milvus_results=milvus_results,
            graph_results=graph_results,
            top_k=top_k
        )

        print(f"  [完成] 融合后返回 {len(fused_results)} 个结果")

        # 输出融合结果摘要
        if fused_results:
            # 统计来源分布
            source_counts = {"milvus": 0, "graph": 0, "hybrid": 0}
            for r in fused_results:
                source = r.get("source", "unknown")
                if source in source_counts:
                    source_counts[source] += 1

            print(f"\n  {'='*50}")
            print(f"  【融合结果 Top {len(fused_results)}】")
            print(f"  {'='*50}")
            for i, r in enumerate(fused_results, 1):
                source = r.get("source", "unknown")
                score = r.get("fused_score", 0)
                milvus_rank = r.get("milvus_rank")
                graph_rank = r.get("graph_rank")

                # 来源标识
                source_label = {
                    "milvus": "[Milvus]",
                    "graph": "[Neo4j]",
                    "hybrid": "[Hybrid]"
                }.get(source, "[???]")

                # 排名信息
                rank_info = []
                if milvus_rank:
                    rank_info.append(f"M#{milvus_rank}")
                if graph_rank:
                    rank_info.append(f"G#{graph_rank}")

                rank_str = " ".join(rank_info) if rank_info else "N/A"

                # 显示跳数信息（如果有多跳推理）
                hop_info = r.get("hop_count", "")
                if hop_info:
                    rank_str += f" | {hop_info}跳"

                print(f"    {i}. {r.get('product_id', 'N/A'):8s} | {source_label:8s} | 评分:{score:.4f} | 排名:{rank_str}")

            print(f"  {'='*50}")
            print(f"  来源分布: Milvus独有={source_counts['milvus']}, Neo4j独有={source_counts['graph']}, 混合={source_counts['hybrid']}")

        return fused_results

    # ==================== 结果融合算法 ====================

    def _fuse_results(
        self,
        milvus_results: List[Dict],
        graph_results: List[Dict],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        融合 Milvus 和 Neo4j 检索结果

        使用 RRF (Reciprocal Rank Fusion) 算法：
        - score = milvus_weight * (1 / (k + milvus_rank)) + graph_weight * (1 / (k + graph_rank))
        - k = 60 (RRF 常数)

        Args:
            milvus_results: Milvus 检索结果
            graph_results: Neo4j 检索结果
            top_k: 返回数量

        Returns:
            融合后的结果列表
        """
        k = 60  # RRF 常数

        # 收集所有商品ID
        all_product_ids = set()

        milvus_scores = {}  # product_id -> (rank, score)
        for i, r in enumerate(milvus_results):
            pid = r.get("product_id")
            if pid:
                all_product_ids.add(pid)
                # RRF 分数: 1 / (k + rank)
                rrf_score = 1 / (k + i + 1)
                milvus_scores[pid] = {
                    "rank": i + 1,
                    "raw_score": r.get("score", 0),
                    "rrf_score": rrf_score,
                    "data": r
                }

        graph_scores = {}  # product_id -> (rank, score)
        for i, r in enumerate(graph_results):
            pid = r.get("product_id")
            if pid:
                all_product_ids.add(pid)
                rrf_score = 1 / (k + i + 1)
                graph_scores[pid] = {
                    "rank": i + 1,
                    "raw_score": r.get("score", 0),
                    "rrf_score": rrf_score,
                    "data": r
                }

        print(f"      [RRF计算] 候选商品数: {len(all_product_ids)} (Milvus:{len(milvus_scores)}, Neo4j:{len(graph_scores)})")

        # 融合计算
        fused_results = []
        hybrid_count = 0
        milvus_only_count = 0
        graph_only_count = 0

        for pid in all_product_ids:
            milvus_info = milvus_scores.get(pid)
            graph_info = graph_scores.get(pid)

            # 确定数据来源
            if milvus_info and graph_info:
                # 两者都有，融合
                fused_rrf = (
                    self.milvus_weight * milvus_info["rrf_score"] +
                    self.graph_weight * graph_info["rrf_score"]
                )
                source = "hybrid"
                base_data = milvus_info["data"]  # 优先使用 Milvus 的完整数据
                hybrid_count += 1
            elif milvus_info:
                # 仅 Milvus
                fused_rrf = milvus_info["rrf_score"]
                source = "milvus"
                base_data = milvus_info["data"]
                milvus_only_count += 1
            else:
                # 仅 Neo4j
                fused_rrf = graph_info["rrf_score"]
                source = "graph"
                base_data = graph_info["data"]
                graph_only_count += 1

            # 构建融合结果
            result = {
                **base_data,
                "fused_score": fused_rrf,
                "source": source,
                "milvus_rank": milvus_info["rank"] if milvus_info else None,
                "graph_rank": graph_info["rank"] if graph_info else None,
            }

            fused_results.append(result)

        # 按融合评分排序
        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)

        print(f"      [融合统计] Hybrid:{hybrid_count}, Milvus独有:{milvus_only_count}, Neo4j独有:{graph_only_count}")

        return fused_results[:top_k]

    # ==================== 快速检索接口 ====================

    def retrieve_by_attributes(
        self,
        category: str = "",
        style: str = "",
        season: str = "",
        scene_hint: str = "",
        top_k: int = TOP_K_RETRIEVAL,
        enable_multi_hop: bool = True,
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        仅基于属性检索（不需要向量）

        适用于：新品图片尚未编码时的快速检索

        Args:
            category: 品类
            style: 风格
            season: 季节
            scene_hint: 场景提示
            top_k: 返回数量
            enable_multi_hop: 是否启用多跳推理（默认True）
            max_hops: 最大跳数（默认3）

        Returns:
            检索结果列表
        """
        print(f"\n{'='*50}")
        print("【属性检索】Neo4j 图谱检索")
        print(f"{'='*50}")
        print(f"  查询参数: category={category or 'All'}, style={style or 'All'}, season={season or 'All'}")
        print(f"  场景提示: {scene_hint or 'None'}")
        print(f"  多跳推理: {'启用' if enable_multi_hop else '禁用'}")

        if self.neo4j_enabled and self.graph_retriever:
            # 使用 Neo4j 图谱检索（支持多跳推理）
            results = self.graph_retriever.retrieve_by_graph(
                category=category,
                style=style,
                season=season,
                scene_hint=scene_hint,
                top_k=top_k,
                enable_multi_hop=enable_multi_hop,
                max_hops=max_hops
            )
            print(f"  [完成] 检索到 {len(results)} 个结果")
            if results:
                print(f"      Top 3: {', '.join([r.get('product_id', 'N/A') for r in results[:3]])}")
                # 显示跳数分布
                hop_counts = {}
                for r in results:
                    hop = r.get('hop_count', 'unknown')
                    hop_counts[hop] = hop_counts.get(hop, 0) + 1
                if enable_multi_hop and hop_counts:
                    print(f"      跳数分布: {', '.join([f'{k}跳:{v}' for k, v in hop_counts.items()])}")
            return results
        else:
            # Neo4j 不可用，返回空
            print("  [失败] Neo4j 未启用，属性检索不可用")
            return []

    # ==================== 两阶段检索：Neo4j过滤 + Milvus精排 ====================

    def two_stage_retrieve(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        # Stage 1: Neo4j 过滤参数
        category: str = "",
        style: str = "",
        season: str = "",
        min_sales: int = 0,
        sales_top_k: int = None,
        max_candidates: int = 1000,
        # Stage 2: Milvus 精排参数
        top_k: int = TOP_K_RETRIEVAL,
        enable_cycle: bool = False,  # 两阶段检索通常不需要循环
        # 质量评估参数
        query_category: str = "",
        query_style: str = "",
        query_season: str = "",
        query_scene_hint: str = "",
        # 多跳推理参数
        enable_multi_hop: bool = True,
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        两阶段检索架构：Neo4j 多跳推理 + Milvus 向量精排

        业务场景：
        1. Neo4j 第一阶段：多跳推理扩展候选集
           - 第1跳: 目标风格直接匹配
           - 第2跳: 相似风格扩展
           - 结合业务规则（品类、销量Top K）
        2. Milvus 第二阶段：在候选集中做向量相似度精排

        优势：
        - 降低向��检索计算量（只在小范围候选集检索）
        - 结合业务规则（销量、点击率）和视觉相似度
        - 多跳推理扩展候选集，增加召回多样性

        Args:
            query_dense: 稠密查询向量
            query_sparse: 稀疏查询向量
            category: 品类过滤
            style: 风格过滤
            season: 季节过滤
            min_sales: 最低销量
            sales_top_k: 取销量Top K
            max_candidates: 最大候选数量
            top_k: 最终返回数量
            enable_cycle: 是否启用循环检索
            query_category: 查询品类（质量评估用）
            query_style: 查询风格（质量评估用）
            query_season: 查询季节（质量评估用）
            query_scene_hint: 查询场景（质量评估用）
            enable_multi_hop: 是否启用多跳推理（默认True）
            max_hops: 最大跳数（默认3）

        Returns:
            检索结果列表
        """
        print(f"\n{'='*60}")
        print("【两阶段检索架构】Neo4j 多跳推理 + Milvus 向量精排")
        print(f"{'='*60}")
        print(f"  多跳推理: {'启用' if enable_multi_hop else '禁用'}")

        # ==================== Stage 1: Neo4j 多跳推理获取候选集 ====================
        candidate_ids = []

        if self.neo4j_enabled and self.graph_retriever:
            if enable_multi_hop and style:
                # 使用多跳推理获取候选集
                print(f"\n  [Stage 1.1] Neo4j 多跳推理获取候选集...")
                multi_hop_results = self.graph_retriever.multi_hop_retrieve(
                    category=category,
                    style=style,
                    season=season,
                    scene_hint=query_scene_hint or "",
                    top_k=max_candidates,  # 获取更多候选
                    max_hops=max_hops
                )
                candidate_ids = [r["product_id"] for r in multi_hop_results]

                # 显示跳数分布
                hop_counts = {}
                for r in multi_hop_results:
                    hop = r.get('hop_count', 'unknown')
                    hop_counts[hop] = hop_counts.get(hop, 0) + 1
                if hop_counts:
                    print(f"  [完成] 候选集跳数分布: {', '.join([f'{k}跳:{v}' for k, v in hop_counts.items()])}")

            else:
                # 使用传统过滤获取候选集
                print(f"\n  [Stage 1.1] Neo4j 传统过滤获取候选集...")
                candidate_ids = self.graph_retriever.filter_candidate_products(
                    category=category,
                    style=style,
                    season=season,
                    min_sales=min_sales,
                    sales_top_k=sales_top_k,
                    limit=max_candidates
                )

        else:
            print("  [跳过] Neo4j 未启用，使用 Milvus 全量检索")
            # 使用纯 Milvus 检索（单次检索，不需要循环）
            milvus_results = self.milvus_retriever._single_retrieve(
                query_dense=query_dense,
                query_sparse=query_sparse,
                category=category,
                min_sales=min_sales,
                top_k=top_k
            )
            return milvus_results

        if not candidate_ids:
            print("  [警告] Neo4j 候选集为空，降级到 Milvus 全量检索")
            # 使用纯 Milvus 检索（单次检索）
            milvus_results = self.milvus_retriever._single_retrieve(
                query_dense=query_dense,
                query_sparse=query_sparse,
                category=category,
                min_sales=min_sales,
                top_k=top_k
            )
            return milvus_results

        # ==================== Stage 2: Milvus 向量精排 ====================
        print(f"\n{'='*60}")
        print("【两阶段检索 - Stage 2】Milvus 向量精排")
        print(f"{'='*60}")
        print(f"  候选集大小: {len(candidate_ids)} 个商品")
        print(f"  目标返回: {top_k} 个最相似商品")

        # 在 Milvus 中检索（通过候选ID过滤）
        # 注意：这里需要 Milvus 支持按ID列表过滤
        # 如果 Milvus 不支持，则使用全量检索后过滤

        try:
            # 调用 Milvus 检索（需要支持 candidate_ids 过滤）
            results = self._milvus_retrieve_with_candidates(
                query_dense=query_dense,
                query_sparse=query_sparse,
                candidate_ids=candidate_ids,
                category=category,  # 传递 category 给 Milvus
                top_k=top_k,
                enable_cycle=enable_cycle,
                query_category=query_category,
                query_style=query_style,
                query_season=query_season,
                query_scene_hint=query_scene_hint
            )

            print(f"  [完成] 精排完成，返回 {len(results)} 个结果")

            # 输出最终结果
            if results:
                print(f"\n  最终结果 Top {len(results)}:")
                for i, r in enumerate(results, 1):
                    print(f"    {i}. {r.get('product_id', 'N/A')} | "
                          f"相似度:{r.get('score', 0):.4f} | "
                          f"销量:{r.get('sales_count', 0)}")

            print(f"{'='*60}")

            return results

        except Exception as e:
            print(f"  [失败] Milvus 精排失败: {e}，降级到 Milvus 全量检索")
            # 使用纯 Milvus 检索（单次检索）
            milvus_results = self.milvus_retriever._single_retrieve(
                query_dense=query_dense,
                query_sparse=query_sparse,
                category=category,
                min_sales=min_sales,
                top_k=top_k
            )
            return milvus_results

    def _milvus_retrieve_with_candidates(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        candidate_ids: List[str],
        category: str = "",  # 添加 category 参数
        top_k: int = 3,
        enable_cycle: bool = False,
        query_category: str = "",
        query_style: str = "",
        query_season: str = "",
        query_scene_hint: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Milvus 向量检索（在候选ID列表中检索）

        Args:
            query_dense: 稠密查询向量
            query_sparse: 稀疏查询向量
            candidate_ids: 候选商品ID列表
            category: 品类过滤（传递给 Milvus）
            top_k: 返回数量
            enable_cycle: 是否启用循环检索
            query_category: ���询品类
            query_style: 查询风格
            query_season: 查询季节
            query_scene_hint: 查询场景

        Returns:
            检索结果列表
        """
        # 先获取全量结果 - 使用单次检索
        all_results = self.milvus_retriever._single_retrieve(
            query_dense=query_dense,
            query_sparse=query_sparse,
            category=category,
            min_sales=0,
            top_k=top_k * 3
        )

        # 过滤出候选集中的结果
        candidate_set = set(candidate_ids)
        filtered_results = [
            r for r in all_results
            if r.get("product_id") in candidate_set
        ]

        # 按相似度排序并返回top_k
        filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return filtered_results[:top_k]

    # ==================== 状态查询 ====================

    def get_status(self) -> Dict[str, Any]:
        """
        获取混合检索器状态

        Returns:
            状态信息字典
        """
        status = {
            "milvus_enabled": True,
            "milvus_weight": self.milvus_weight,
            "neo4j_enabled": self.neo4j_enabled,
            "graph_weight": self.graph_weight,
        }

        if self.neo4j_enabled and self.graph_retriever:
            status["neo4j_connected"] = self.graph_retriever.is_connected()
            status["graph_stats"] = self.graph_retriever.get_graph_stats()
        else:
            status["neo4j_connected"] = False
            status["graph_stats"] = {}

        # Milvus 状态
        try:
            status["milvus_connected"] = self.milvus_retriever.has_collection()
            if status["milvus_connected"]:
                stats = self.milvus_retriever.get_collection_stats()
                status["milvus_stats"] = {
                    "row_count": stats.get("row_count", 0)
                }
        except Exception:
            status["milvus_connected"] = False
            status["milvus_stats"] = {}

        return status


# ==================== 便捷函数 ====================

def create_hybrid_retriever(
    milvus_weight: float = 0.6,
    graph_weight: float = 0.4
) -> HybridRetriever:
    """
    创建混合检索器实例

    Args:
        milvus_weight: Milvus 权重
        graph_weight: Neo4j 权重

    Returns:
        混合检索器实例
    """
    return HybridRetriever(
        milvus_weight=milvus_weight,
        graph_weight=graph_weight
    )


if __name__ == "__main__":
    """
    测试混合检索器
    """
    import argparse

    parser = argparse.ArgumentParser(description="混合检索器测试")
    parser.add_argument("--status", action="store_true", help="显示检索器状态")
    parser.add_argument("--test", action="store_true", help="执行测试检索")

    args = parser.parse_args()

    retriever = create_hybrid_retriever()

    if args.status:
        status = retriever.get_status()
        print("\n混合检索器状态:")
        print(f"  Milvus: {'[OK]' if status['milvus_enabled'] else '[FAIL]'}")
        print(f"  Neo4j: {'[OK]' if status['neo4j_enabled'] else '[FAIL]'}")
        print(f"  权重: Milvus {status['milvus_weight']:.2f} / Graph {status['graph_weight']:.2f}")

    if args.test:
        # 测试检索（需要实际数据）
        print("\n执行测试检索...")
        # 这里需要实际的查询向量
        print("  (需要实际的商品数据才能测试)")
