"""
电商服饰知识图谱 - 图谱检索器

功能说明：
- 基于知识图谱进行多跳推理检索
- 结合品类、风格、季节、场景等维度进行关联推理
- 实现 Milvus 向量检索 + Neo4j 图谱检索的混合引擎

使用示例：
    >>> from graph import FashionGraphRetriever
    >>> retriever = FashionGraphRetriever()
    >>> results = retriever.retrieve_by_graph(
    ...     category="midi_dress",
    ...     style="elegant",
    ...     season="summer",
    ...     scene_hint="beach",
    ...     top_k=3
    ... )
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from neo4j import GraphDatabase, Driver as Neo4jDriver

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB,
    IMAGE_DIR, MIN_SALES_COUNT
)
from utils import load_image

logger = logging.getLogger(__name__)

# 输入验证白名单
VALID_CATEGORIES = {
    'midi_dress', 'maxi_dress', 'mini_dress', 'skirt', 'top', 'pants',
    'jumpsuit', 'playsuit', 'romper', 'bodysuit', 'overalls', 'dungarees'
}

VALID_STYLES = {
    'casual', 'formal', 'sporty', 'elegant', 'vintage', 'modern',
    'classic', 'boho', 'bohemian', 'romantic', 'minimalist', 'edgy',
    'preppy', 'chic', 'streetwear', 'rock', 'feminine', 'masculine',
    'knitted', 'drawstring', 'lace', 'floral', 'peasant', 'pencil',
    'pleated', 'cropped', 'skinny', 'tailored', 'navy', 'camel', 'olive',
    'pink', 'party', 'professional', 'blouse', 'bolero', 'linen', 'basic'
}

VALID_SEASONS = {
    'all_season', 'spring', 'summer', 'autumn', 'winter'
}


def validate_category(category: Optional[str]) -> str:
    """验证并规范化品类输入"""
    if category is None:
        return ""
    category = category.strip().lower()
    if category and category not in VALID_CATEGORIES:
        logger.warning(f"未知品类: {category}，将忽略品类过滤")
        return ""
    return category


def validate_style(style: Optional[str]) -> str:
    """验证并规范化风格输入"""
    if style is None:
        return ""
    style = style.strip().lower()
    if style and style not in VALID_STYLES:
        logger.warning(f"未知风格: {style}，将忽略风格过滤")
        return ""
    return style


def validate_season(season: Optional[str]) -> str:
    """验证并规范化季节输入"""
    if season is None:
        return "all_season"
    season = season.strip().lower()
    if season and season not in VALID_SEASONS:
        logger.warning(f"未知季节: {season}，使用 all_season")
        return "all_season"
    return season


class FashionGraphRetriever:
    """
    电商服饰知识图谱检索器

    实现基于图谱的多跳推理检索，通过关联路径找到相似爆款。

    检索策略：
    1. 基于品类、风格、季节的直接匹配
    2. 基于场景的多跳推理：新品需求 → 匹配场景 → 匹配风格 → 找到商品
    3. 同风格商品的关联推荐
    4. 综合评分：路径长度、销量、风格相似度
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        初始化图谱检索器，连接 Neo4j 数据库

        Args:
            uri: Neo4j 连接 URI，默认使用 config.NEO4J_URI
            user: 用户名，默认使用 config.NEO4J_USER
            password: 密码，默认使用 config.NEO4J_PASSWORD

        Raises:
            ConnectionError: Neo4j 连接失败时抛出
        """
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.database = NEO4J_DB

        self.driver: Optional[Neo4jDriver] = None

        # 测试连接
        try:
            self._connect()
            print(f"[OK] Neo4j 检索器连接成功: {self.uri}")
        except Exception as e:
            print(f"[FAIL] Neo4j 连接失败: {e}")
            print("  ! 将使用 Milvus 向量检索兜底")
            # 不抛出异常，允许兜底到 Milvus 检索

    def _connect(self):
        """建立 Neo4j 连接"""
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
            connection_timeout=30
        )

        # 验证连接
        with self.driver.session(database=self.database) as session:
            result = session.run("RETURN 1 AS test")
            result.single()

    def close(self):
        """关闭 Neo4j 连接"""
        if self.driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def is_connected(self) -> bool:
        """
        检查 Neo4j 连接状态

        Returns:
            是否已连接
        """
        if self.driver is None:
            return False
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 AS test").single()
            return True
        except Exception:
            return False

    # ==================== 图谱检索 ====================

    def retrieve_by_graph(
        self,
        category: Optional[str] = None,
        style: Optional[str] = None,
        season: Optional[str] = None,
        scene_hint: Optional[str] = None,
        top_k: int = 3,
        enable_multi_hop: bool = True,
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        基于知识图谱的多跳推理检索

        检���逻辑：
        1. 【多跳推理】启用时使用风格扩展多跳推理
           - 第1跳: 目标风格直接匹配
           - 第2跳: 相似风格扩展
           - 第3跳: 跨品类扩展
        2. 【传统检索】未启用多跳时使用原逻辑
           - 直接匹配
           - 场景推理
           - 同风格关联
        3. 【综合排序】按关联度评分排序返回

        Args:
            category: 新品品类（如 midi_dress, maxi_dress）
            style: 新品风格（如 elegant, casual）
            season: 新品季节（如 summer, winter, all_season）
            scene_hint: 场景提示（如 beach, office, party）
            top_k: 返回商品数量
            enable_multi_hop: 是否启用多跳推理（默认True）
            max_hops: 最大跳数（默认3）

        Returns:
            检索结果列表，每个结果包含：
            - product_id: 商品ID
            - category: 品类
            - style: 风格
            - season: 季节
            - color: 颜色
            - sales_count: 销量
            - price: 价格
            - description: 描述
            - score: 关联度评分
            - match_reason: 匹配原因（用于调试）
            - hop_count: 跳数（多跳推理时）
            - image: PIL 图片对象

            示例：
            [
                {
                    "product_id": "SKU001",
                    "category": "midi_dress",
                    "style": "elegant",
                    "score": 0.95,
                    "match_reason": "hop_1_direct:elegant",
                    "hop_count": 1,
                    "image": <PIL.Image>
                },
                ...
            ]
        """
        if not self.is_connected():
            print("  ! Neo4j 未连接，返回空结果")
            return []

        # 输入验证和规范化
        category = validate_category(category)
        style = validate_style(style)
        season = validate_season(season)
        scene_hint = scene_hint.strip() if scene_hint else ""

        # 检查图谱是否为空
        stats = self.get_graph_stats()
        node_count = stats.get('node_count', 0)
        if node_count == 0:
            logger.warning("Neo4j 图谱为空，跳过图谱检索")
            return []

        # 如果启用了多跳推理且有风格信息，使用多跳推理
        if enable_multi_hop and style:
            return self.multi_hop_retrieve(
                category=category,
                style=style,
                season=season,
                scene_hint=scene_hint,
                top_k=top_k,
                max_hops=max_hops
            )

        # 否则使用传统检索逻辑
        print(f"\n{'='*60}")
        print("【知识图谱检索】传统模式")
        print(f"{'='*60}")
        print(f"  品类: {category or '不限'}")
        print(f"  风格: {style or '不限'}")
        print(f"  季节: {season or '不限'}")
        print(f"  场景: {scene_hint or '不限'}")
        print(f"  返回数量: {top_k}")

        all_results = []

        # ==================== 策略1: 直接匹配 ====================
        if category or style or season:
            direct_results = self._direct_match(category, style, season, top_k)
            all_results.extend(direct_results)

        # ==================== 策略2: 场景推理 ====================
        if scene_hint:
            scene_results = self._scene_inference(
                category, style, season, scene_hint, top_k
            )
            # 去重：避免与直接匹配重复
            existing_ids = {r["product_id"] for r in all_results}
            for result in scene_results:
                if result["product_id"] not in existing_ids:
                    all_results.append(result)

        # ==================== 策略3: 同风格关联 ====================
        if style:
            similar_results = self._similar_style_match(
                category, style, season, top_k
            )
            # 去重
            existing_ids = {r["product_id"] for r in all_results}
            for result in similar_results:
                if result["product_id"] not in existing_ids:
                    all_results.append(result)

        # ==================== 加载图片并排序 ====================
        results_with_images = []
        for result in all_results:
            try:
                img_path = IMAGE_DIR / f"{result['product_id']}.jpg"
                if img_path.exists():
                    from utils import load_image
                    result["image"] = load_image(str(img_path))
                    results_with_images.append(result)
                else:
                    print(f"  ! 图片未找到: {result['product_id']}.jpg")
            except Exception as e:
                print(f"  ! 图片加载失败 ({result['product_id']}): {e}")

        # 按评分排序
        results_with_images.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 取 top_k
        final_results = results_with_images[:top_k]

        # 输出结果摘要
        print(f"\n  找到 {len(final_results)} 个相关商品:")
        for i, r in enumerate(final_results, 1):
            print(f"    {i}. {r['product_id']} | {r.get('category', 'N/A')} | "
                  f"{r.get('style', 'N/A')} | 评分: {r.get('score', 0):.2f}")

        return final_results

    def _direct_match(
        self,
        category: str,
        style: str,
        season: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        策略1: 直接匹配检索

        查找同时满足品类、风格、季节条件的高销量商品

        Cypher 逻辑：
        1. 匹配 Product 节点
        2. 通过 HAS_CATEGORY, HAS_STYLE, HAS_SEASON 关系过滤
        3. 按销量降序排序
        4. 计算匹配度评分

        Args:
            category: 品类过滤
            style: 风格过滤
            season: 季节过滤
            top_k: 返回数量

        Returns:
            匹配结果列表
        """
        results = []

        # 构建 Cypher 查询
        # 动态构建 MATCH 条件
        match_clauses = []
        where_conditions = ["p.sales_count >= $min_sales"]
        params = {"min_sales": MIN_SALES_COUNT}

        if category:
            match_clauses.append("MATCH (p)-[:HAS_CATEGORY]->(c:Category {name: $category})")
            params["category"] = category
        if style:
            match_clauses.append("MATCH (p)-[:HAS_STYLE]->(s:Style {name: $style})")
            params["style"] = style
        if season:
            # all_season 作为通用匹配
            where_conditions.append("(p.season = $season OR p.season = 'all_season')")
            params["season"] = season

        # 组合查询
        match_query = "MATCH (p:Product)\n" + "\n".join(match_clauses)
        where_query = "WHERE " + " AND ".join(where_conditions)

        cypher = f"""
        {match_query}
        {where_query}
        RETURN p.product_id AS product_id,
               p.category AS category,
               p.style AS style,
               p.season AS season,
               p.color AS color,
               p.sales_count AS sales_count,
               p.price AS price,
               p.description AS description
        ORDER BY p.sales_count DESC
        LIMIT {top_k * 2}
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = session.run(cypher, params)

                for record in records:
                    # 计算匹配度评分
                    score = self._calculate_direct_match_score(
                        record, category, style, season
                    )

                    results.append({
                        "product_id": record["product_id"],
                        "category": record["category"],
                        "style": record["style"],
                        "season": record["season"],
                        "color": record["color"],
                        "sales_count": record["sales_count"],
                        "price": record["price"],
                        "description": record["description"],
                        "score": score,
                        "match_reason": "direct_match"
                    })

        except Exception as e:
            print(f"  ! 直接匹配查询失败: {e}")

        return results

    def _scene_inference(
        self,
        category: str,
        style: str,
        season: str,
        scene_hint: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        策略2: 场景推理检索

        通过场景提示找到适合该场景的商品

        Cypher 逻辑：
        1. 匹配 Scene 节点（支持模糊匹配）
        2. 找到与该场景关联的商品
        3. 结合品类、风格、季节进一步过滤
        4. 按销量和场景关联度排序

        Args:
            category: 品类过滤
            style: 风格过滤
            season: 季节过滤
            scene_hint: 场景提示词
            top_k: 返回数量

        Returns:
            匹配结果列表
        """
        results = []

        # 场景关键词映射
        scene_keywords = {
            "beach": ["beach", "sea", "ocean", "vacation", "summer"],
            "office": ["office", "work", "business", "formal"],
            "party": ["party", "evening", "night", "cocktail"],
            "outdoor": ["outdoor", "street", "casual", "daily"],
            "indoor": ["indoor", "home", "lounge", "relax"],
        }

        # 查找匹配的场景节点
        matched_scenes = []
        scene_lower = scene_hint.lower()

        for scene_name, keywords in scene_keywords.items():
            if any(kw in scene_lower for kw in keywords):
                matched_scenes.append(scene_name)

        # 如果没有匹配的场景，使用原始提示词
        if not matched_scenes:
            matched_scenes = [scene_hint.lower()]

        # 构建 Cypher 查询
        params = {
            "min_sales": MIN_SALES_COUNT,
            "scenes": matched_scenes
        }

        # 可选过滤条件
        if category:
            params["category"] = category
        if style:
            params["style"] = style
        if season:
            params["season"] = season

        category_filter = "AND p.category = $category" if category else ""
        style_filter = "AND p.style = $style" if style else ""
        season_filter = "AND (p.season = $season OR p.season = 'all_season')" if season else ""

        cypher = f"""
        // 匹配场景节点
        MATCH (sc:Scene)
        WHERE sc.name IN $scenes

        // 找到与该场景关联的商品
        MATCH (sc)<-[:SUITABLE_SCENE]-(p:Product)

        // 过滤条件
        WHERE p.sales_count >= $min_sales
        {category_filter}
        {style_filter}
        {season_filter}

        RETURN p.product_id AS product_id,
               p.category AS category,
               p.style AS style,
               p.season AS season,
               p.color AS color,
               p.sales_count AS sales_count,
               p.price AS price,
               p.description AS description
        ORDER BY p.sales_count DESC
        LIMIT {top_k * 2}
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = session.run(cypher, params)

                for record in records:
                    # 计算场景推理评分
                    score = self._calculate_scene_score(
                        record, category, style, season
                    )

                    results.append({
                        "product_id": record["product_id"],
                        "category": record["category"],
                        "style": record["style"],
                        "season": record["season"],
                        "color": record["color"],
                        "sales_count": record["sales_count"],
                        "price": record["price"],
                        "description": record["description"],
                        "score": score,
                        "match_reason": f"scene_inference:{scene_hint}"
                    })

        except Exception as e:
            print(f"  ! 场景推理查询失败: {e}")

        return results

    def _similar_style_match(
        self,
        category: str,
        style: str,
        season: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        策略3: 同风格关联检索

        找到与目标风格相似的其他爆款商品

        Cypher 逻辑：
        1. 找到目标风格的 Style 节点
        2. 找到该风格下的所有商品
        3. 如果有品类限制，优先同品类商品
        4. 按 SIMILAR_STYLE 关系的权重排序

        Args:
            category: 品类过滤
            style: 目标风格
            season: 季节过滤
            top_k: 返回数量

        Returns:
            匹配结果列表
        """
        results = []

        if not style:
            return results

        params = {
            "style": style,
            "min_sales": 1000  # 略低的阈值，扩大召回
        }

        category_filter = "AND p.category = $category" if category else ""
        season_filter = "AND (p.season = $season OR p.season = 'all_season')" if season else ""

        if category:
            params["category"] = category
        if season:
            params["season"] = season

        cypher = f"""
        // 匹配目标风格节点
        MATCH (s:Style {{name: $style}})

        // 找到该风格下的商品
        MATCH (p:Product)-[:HAS_STYLE]->(s)

        // 过滤条件
        WHERE p.sales_count >= $min_sales
        {category_filter}
        {season_filter}

        RETURN p.product_id AS product_id,
               p.category AS category,
               p.style AS style,
               p.season AS season,
               p.color AS color,
               p.sales_count AS sales_count,
               p.price AS price,
               p.description AS description
        ORDER BY p.sales_count DESC
        LIMIT {top_k * 2}
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = session.run(cypher, params)

                for record in records:
                    # 计算同风格评分
                    score = self._calculate_style_score(
                        record, category, season
                    )

                    results.append({
                        "product_id": record["product_id"],
                        "category": record["category"],
                        "style": record["style"],
                        "season": record["season"],
                        "color": record["color"],
                        "sales_count": record["sales_count"],
                        "price": record["price"],
                        "description": record["description"],
                        "score": score,
                        "match_reason": f"similar_style:{style}"
                    })

        except Exception as e:
            print(f"  ! 同风格关联查询失败: {e}")

        return results

    # ==================== 评分计算 ====================

    def _calculate_direct_match_score(
        self,
        record,
        category: str,
        style: str,
        season: str
    ) -> float:
        """
        计算直接匹配的关联度评分

        评分公式：
        - 基础分: 0.5
        - 品类匹配: +0.3
        - 风格匹配: +0.3
        - 季节匹配: +0.1
        - 销量加成: 最多 +0.2

        Args:
            record: Neo4j 查询记录
            category: 目标品类
            style: 目标风格
            season: 目标季节

        Returns:
            关联度评分 (0-1)
        """
        score = 0.5  # 基础分

        # 维度匹配加成
        if category and record.get("category") == category:
            score += 0.3
        if style and record.get("style") == style:
            score += 0.3
        if season:
            record_season = record.get("season", "")
            if record_season == season or record_season == "all_season":
                score += 0.1

        # 销量加成（归一化到 0-0.2）
        sales = record.get("sales_count", 0)
        sales_bonus = min(sales / 10000, 0.2)
        score += sales_bonus

        return min(score, 1.0)

    def _calculate_scene_score(
        self,
        record,
        category: str,
        style: str,
        season: str
    ) -> float:
        """
        计算场景推理的关联度评分

        评分公式：
        - 基础分: 0.4（场景推理略低于直接匹配）
        - 品类匹配: +0.25
        - 风格匹配: +0.25
        - 季节匹配: +0.1

        Args:
            record: Neo4j 查询记录
            category: 目标品类
            style: 目标风格
            season: 目标季节

        Returns:
            关联度评分 (0-1)
        """
        score = 0.4  # 基础分（略低）

        if category and record.get("category") == category:
            score += 0.25
        if style and record.get("style") == style:
            score += 0.25
        if season:
            record_season = record.get("season", "")
            if record_season == season or record_season == "all_season":
                score += 0.1

        return min(score, 1.0)

    def _calculate_style_score(
        self,
        record,
        category: str,
        season: str
    ) -> float:
        """
        计算同风格关联的关联度评分

        评分公式：
        - 基础分: 0.5
        - 品类匹配: +0.3
        - 季节匹配: +0.2

        Args:
            record: Neo4j 查询记录
            category: 目标品类
            season: 目标季节

        Returns:
            关联度评分 (0-1)
        """
        score = 0.5  # 基础分

        if category and record.get("category") == category:
            score += 0.3
        if season:
            record_season = record.get("season", "")
            if record_season == season or record_season == "all_season":
                score += 0.2

        return min(score, 1.0)

    # ==================== 工具方法 ====================

    def get_product_details(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        获取单个商品的详细信息

        Args:
            product_id: 商品ID

        Returns:
            商品详细信息，不存在返回 None
        """
        if not self.is_connected():
            return None

        cypher = """
        MATCH (p:Product {product_id: $product_id})
        OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
        OPTIONAL MATCH (p)-[:HAS_STYLE]->(s:Style)
        OPTIONAL MATCH (p)-[:HAS_SEASON]->(se:Season)
        OPTIONAL MATCH (p)-[:HAS_COLOR]->(co:Color)
        OPTIONAL MATCH (p)-[:HAS_MATERIAL]->(m:Material)
        OPTIONAL MATCH (p)-[:SUITABLE_SCENE]->(sc:Scene)
        RETURN p,
               collect(DISTINCT c.category_name) AS categories,
               collect(DISTINCT s.style_name) AS styles,
               collect(DISTINCT se.season_name) AS seasons,
               collect(DISTINCT co.color_name) AS colors,
               collect(DISTINCT m.material_name) AS materials,
               collect(DISTINCT sc.scene_name) AS scenes
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, {"product_id": product_id})
                record = result.single()

                if record:
                    return {
                        "product_id": product_id,
                        "categories": record["categories"],
                        "styles": record["styles"],
                        "seasons": record["seasons"],
                        "colors": record["colors"],
                        "materials": record["materials"],
                        "scenes": record["scenes"],
                    }
        except Exception as e:
            print(f"  ! 获取商品详情失败: {e}")

        return None

    def get_similar_products(
        self,
        product_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        获取与指定商品相似的其他商品

        相似判断依据：
- 同品类
- 同风格
- 同季节

        Args:
            product_id: 商品ID
            top_k: 返回数量

        Returns:
            相似商品列表
        """
        if not self.is_connected():
            return []

        cypher = f"""
        // 查找目标商品
        MATCH (target:Product {{product_id: $product_id}})

        // 找到同品类的商品
        MATCH (target)-[:HAS_CATEGORY]->(c:Category)<-[:HAS_CATEGORY]-(similar:Product)

        // 找到同风格的商品
        MATCH (target)-[:HAS_STYLE]->(s:Style)<-[:HAS_STYLE]-(similar)

        // 过滤条件
        WHERE similar.product_id <> $product_id
        AND similar.sales_count >= 500

        RETURN similar.product_id AS product_id,
               similar.category AS category,
               similar.style AS style,
               similar.sales_count AS sales_count,
               similar.price AS price
        ORDER BY similar.sales_count DESC
        LIMIT {top_k}
        """

        results = []
        try:
            with self.driver.session(database=self.database) as session:
                records = session.run(cypher, {"product_id": product_id})

                for record in records:
                    results.append({
                        "product_id": record["product_id"],
                        "category": record["category"],
                        "style": record["style"],
                        "sales_count": record["sales_count"],
                        "price": record["price"]
                    })
        except Exception as e:
            print(f"  ! 查找相似商品失败: {e}")

        return results

    # ==================== 两阶段检索：候选集过滤 ====================

    def filter_candidate_products(
        self,
        category: Optional[str] = None,
        style: Optional[str] = None,
        season: Optional[str] = None,
        min_sales: int = 0,
        max_sales: int = None,
        min_price: float = None,
        max_price: float = None,
        sales_top_k: int = None,
        limit: int = 1000
    ) -> List[str]:
        """
        两阶段检索第一步：基于图谱的多维度过滤，返回候选商品ID列表

        业务场景：
        - 先过滤出"女装衬衫类目、近3个月销量Top100、点击率>5%"的商品
        - 再在候选集中做向量相似度精排

        Args:
            category: 品类过滤
            style: 风格过滤
            season: 季节过滤
            min_sales: 最低销量
            max_sales: 最高销量
            min_price: 最低价格
            max_price: 最高价格
            sales_top_k: 取销量Top K（如Top100）
            limit: 最大返回数量

        Returns:
            候选商品ID列表
        """
        # 输入验证和规范化
        category = validate_category(category)
        style = validate_style(style)
        season = validate_season(season)

        print(f"\n{'='*60}")
        print("【两阶段检索 - Stage 1】Neo4j 候选集过滤")
        print(f"{'='*60}")
        print(f"  主要过滤: 品类={category or '全部'}")
        print(f"  排序依据: 销量Top{sales_top_k if sales_top_k else '全部'}")
        if style and style not in ["all", "all_season", ""]:
            print(f"  风格参考: {style} (不强制过滤)")
        if season and season != "all_season":
            print(f"  季节过滤: {season}")
        print(f"  最大候选数: {limit}")

        # 构建Cypher查询 - 放宽过滤条件
        match_clauses = ["MATCH (p:Product)"]
        where_conditions = []

        # 品类：必须匹配（主要筛选维度）
        if category:
            match_clauses.append("MATCH (p)-[:HAS_CATEGORY]->(c:Category {name: $category})")

        # 风格：改为可选过滤（只在有明确指定时才精确匹配）
        # 为了增加候选集，暂时不按风格精确过滤，只在排序时考虑风格
        # if style and style not in ["all", "all_season", ""]:
        #     match_clauses.append("MATCH (p)-[:HAS_STYLE]->(s:Style {name: $style})")

        # 季节：只有当不是 all_season 时才过滤
        if season and season != "all_season":
            match_clauses.append("MATCH (p)-[:HAS_SEASON]->(sn:Season {name: $season})")

        # 销量：只用于排序，不作为硬性过滤条件（放宽以增加候选集）
        # if min_sales > 0:
        #     where_conditions.append("p.sales_count >= $min_sales")
        if max_sales:
            where_conditions.append("p.sales_count <= $max_sales")
        if min_price is not None:
            where_conditions.append("p.price >= $min_price")
        if max_price is not None:
            where_conditions.append("p.price <= $max_price")

        cypher = " ".join(match_clauses)

        if where_conditions:
            cypher += " WHERE " + " AND ".join(where_conditions)

        cypher += " RETURN p.product_id AS product_id"

        # 销量Top K排序
        if sales_top_k:
            cypher += f" ORDER BY p.sales_count DESC LIMIT {sales_top_k}"
        elif limit:
            cypher += f" LIMIT {limit}"

        params = {
            "category": category,
            "style": style,
            "season": season,
            "min_sales": min_sales,
            "max_sales": max_sales,
            "min_price": min_price,
            "max_price": max_price
        }

        # 移除None值
        params = {k: v for k, v in params.items() if v is not None and v != ""}

        candidate_ids = []
        try:
            with self.driver.session(database=self.database) as session:
                results = session.run(cypher, params)
                candidate_ids = [record["product_id"] for record in results]

            print(f"  [完成] 筛选出 {len(candidate_ids)} 个候选商品")

        except Exception as e:
            print(f"  [失败] 候选集过滤失败: {e}")

        print(f"{'='*60}")

        return candidate_ids

    # ==================== 多跳推理检索 ====================

    # 风格相似性映射（用于多跳推理扩展）
    STYLE_SIMILARITY = {
        "elegant": ["classic", "sophisticated", "formal", "refined"],
        "casual": ["relaxed", "informal", "comfortable", "laid_back"],
        "vintage": ["retro", "classic", "nostalgic", "traditional"],
        "sporty": ["athletic", "active", "performance", "dynamic"],
        "modern": ["contemporary", "sleek", "minimalist", "urban"],
        "formal": ["business", "professional", "elegant", "dressy"],
        "classic": ["elegant", "traditional", "timeless", "vintage"],
        "bohemian": ["boho", "free_spirit", "eclectic", "artistic"],
        "romantic": ["feminine", "delicate", "soft", "dreamy"],
        "minimalist": ["simple", "clean", "modern", "sleek"],
        "edgy": ["bold", "avant_garde", "rebellious", "fierce"],
        "preppy": ["collegiate", "polished", "classic", "clean_cut"],
        "chic": ["stylish", "fashionable", "elegant", "trendy"],
        "streetwear": ["urban", "casual", "modern", "trendy"],
        "rock": ["edgy", "bold", "rebellious", "alternative"],
        "feminine": ["romantic", "delicate", "soft", "elegant"],
        "masculine": ["strong", "structured", "bold", "classic"],
    }

    def multi_hop_retrieve(
        self,
        category: Optional[str] = None,
        style: Optional[str] = None,
        season: Optional[str] = None,
        scene_hint: Optional[str] = None,
        top_k: int = 3,
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        多跳推理检索 - 通过风格扩展获取更多候选商品

        推理路径示例：
        - 第1跳: 查找目标风格节点 (如 "elegant")
        - 第2跳: 查找相似风格节点 (如 "classic", "sophisticated")
        - 第3跳: 查找这些风格下的商品

        Args:
            category: 品类过滤
            style: 目标风格（起点）
            season: 季节过滤
            scene_hint: 场景提示
            top_k: 返回数量
            max_hops: 最大跳数（默认3）

        Returns:
            检索结果列表，包含 hop_count 字段显示跳数
        """
        if not self.is_connected():
            print("  ! Neo4j 未连接，返回空结果")
            return []

        # 输入验证和规范化
        category = validate_category(category)
        style = validate_style(style)
        season = validate_season(season)
        scene_hint = scene_hint.strip() if scene_hint else ""

        print(f"\n{'='*60}")
        print("【多跳推理检索】风格扩展模式")
        print(f"{'='*60}")
        print(f"  起始风格: {style or '未指定'}")
        print(f"  品类过滤: {category or '不限'}")
        print(f"  季节过滤: {season or '不限'}")
        print(f"  最大跳数: {max_hops}")

        all_results = []
        hop_summary = {}  # 记录每跳的结果

        # ==================== 第1跳：直接匹配目标风格 ====================
        if style:
            print(f"\n  >> 第1跳: 查找风格节点 '{style}'")
            hop_1_results = self._hop_1_target_style(category, style, season, top_k)
            hop_summary[1] = len(hop_1_results)
            all_results.extend(hop_1_results)

            # ==================== 第2跳：扩展到相似风格 ====================
            if max_hops >= 2 and style in self.STYLE_SIMILARITY:
                similar_styles = self.STYLE_SIMILARITY[style]
                print(f"\n  >> 第2跳: 查找相似风格节点 {similar_styles}")

                hop_2_results = self._hop_2_similar_styles(
                    category, similar_styles, season, top_k, all_results
                )
                hop_summary[2] = len(hop_2_results)
                all_results.extend(hop_2_results)

                # ==================== 第3跳：跨品类扩展（可选）====================
                if max_hops >= 3 and hop_2_results:
                    print(f"\n  >> 第3跳: 跨品类扩展（同风格不同品类）")

                    hop_3_results = self._hop_3_cross_category(
                        style, similar_styles, season, top_k, all_results
                    )
                    hop_summary[3] = len(hop_3_results)
                    all_results.extend(hop_3_results)

        # ==================== 场景扩展（额外跳数）====================
        if scene_hint and max_hops >= 2:
            print(f"\n  >> 场景扩展跳: 查找场景 '{scene_hint}' 相关商品")
            scene_results = self._scene_inference(
                category, style, season, scene_hint, top_k
            )
            # 场景扩展不计入基础跳数，但标记为场景跳
            for result in scene_results:
                result["hop_type"] = "scene"
                result["hop_count"] = f"scene:{scene_hint}"
            existing_ids = {r["product_id"] for r in all_results}
            for result in scene_results:
                if result["product_id"] not in existing_ids:
                    all_results.append(result)

        # ==================== 加载图片并排序 ====================
        results_with_images = []
        for result in all_results:
            try:
                img_path = IMAGE_DIR / f"{result['product_id']}.jpg"
                if img_path.exists():
                    result["image"] = load_image(str(img_path))
                    results_with_images.append(result)
            except Exception as e:
                print(f"  ! 图片加载失败 ({result['product_id']}): {e}")

        # 按评分排序
        results_with_images.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 取 top_k
        final_results = results_with_images[:top_k]

        # ==================== 输出多跳推理摘要 ====================
        print(f"\n{'='*60}")
        print("【多跳推理摘要】")
        print(f"{'='*60}")
        for hop_num, count in hop_summary.items():
            print(f"  第{hop_num}跳: 找到 {count} 个商品")

        print(f"\n  总计: {len(all_results)} 个候选商品")
        print(f"  去重后: {len(final_results)} 个最终结果")

        print(f"\n  最终结果:")
        for i, r in enumerate(final_results, 1):
            hop_info = r.get("hop_count", "direct")
            print(f"    {i}. {r['product_id']} | {r.get('style', 'N/A')} | "
                  f"跳数: {hop_info} | 评分: {r.get('score', 0):.2f}")

        return final_results

    def _hop_1_target_style(
        self,
        category: str,
        style: str,
        season: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        第1跳：查找目标风格下的商品

        路径：Style节点 → Product节点
        """
        results = []

        params = {
            "style": style,
            "min_sales": MIN_SALES_COUNT
        }

        match_clauses = ["MATCH (s:Style {name: $style})", "MATCH (p:Product)-[:HAS_STYLE]->(s)"]
        where_conditions = ["p.sales_count >= $min_sales"]

        if category:
            match_clauses.append("MATCH (p)-[:HAS_CATEGORY]->(c:Category {name: $category})")
            params["category"] = category

        if season and season != "all_season":
            where_conditions.append("(p.season = $season OR p.season = 'all_season')")
            params["season"] = season

        cypher = " ".join(match_clauses) + " WHERE " + " AND ".join(where_conditions)
        cypher += f"""
        RETURN p.product_id AS product_id,
               p.category AS category,
               p.style AS style,
               p.season AS season,
               p.color AS color,
               p.sales_count AS sales_count,
               p.price AS price
        ORDER BY p.sales_count DESC
        LIMIT {top_k * 2}
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = session.run(cypher, params)
                for record in records:
                    results.append({
                        "product_id": record["product_id"],
                        "category": record["category"],
                        "style": record["style"],
                        "season": record["season"],
                        "color": record["color"],
                        "sales_count": record["sales_count"],
                        "price": record["price"],
                        "score": 0.9,  # 第1跳直接匹配，高分
                        "match_reason": f"hop_1_direct:{style}",
                        "hop_count": 1,
                        "hop_type": "direct_match"
                    })
        except Exception as e:
            print(f"    ! 第1跳查询失败: {e}")

        print(f"    [OK] 第1跳完成: 找到 {len(results)} 个商品")
        return results

    def _hop_2_similar_styles(
        self,
        category: str,
        similar_styles: List[str],
        season: str,
        top_k: int,
        existing_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        第2跳：查找相似风格下的商品

        路径：目标Style → 相似Style节点 → Product节点
        """
        results = []
        existing_ids = {r["product_id"] for r in existing_results}

        # 构建相似风格的 Cypher 查询
        params = {
            "similar_styles": similar_styles,
            "min_sales": MIN_SALES_COUNT
        }

        match_clauses = ["MATCH (s:Style)", "WHERE s.name IN $similar_styles", "MATCH (p:Product)-[:HAS_STYLE]->(s)"]
        where_conditions = ["p.sales_count >= $min_sales"]

        if category:
            match_clauses.append("MATCH (p)-[:HAS_CATEGORY]->(c:Category {name: $category})")
            params["category"] = category

        if season and season != "all_season":
            where_conditions.append("(p.season = $season OR p.season = 'all_season')")
            params["season"] = season

        cypher = " ".join(match_clauses) + " WHERE " + " AND ".join(where_conditions)
        cypher += f"""
        RETURN p.product_id AS product_id,
               p.category AS category,
               p.style AS style,
               p.season AS season,
               p.color AS color,
               p.sales_count AS sales_count,
               p.price AS price
        ORDER BY p.sales_count DESC
        LIMIT {top_k * 3}
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = session.run(cypher, params)
                for record in records:
                    product_id = record["product_id"]
                    if product_id not in existing_ids:
                        results.append({
                            "product_id": product_id,
                            "category": record["category"],
                            "style": record["style"],
                            "season": record["season"],
                            "color": record["color"],
                            "sales_count": record["sales_count"],
                            "price": record["price"],
                            "score": 0.7,  # 第2跳相似风格，中等分数
                            "match_reason": f"hop_2_similar_style:{record['style']}",
                            "hop_count": 2,
                            "hop_type": "similar_style"
                        })
        except Exception as e:
            print(f"    ! 第2跳查询失败: {e}")

        print(f"    [OK] 第2跳完成: 找到 {len(results)} 个新商品")
        return results

    def _hop_3_cross_category(
        self,
        style: str,
        similar_styles: List[str],
        season: str,
        top_k: int,
        existing_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        第3跳：跨品类扩展 - 同风格不同品类

        路径：Style节点 → 不同品类的Product节点
        用于当同风格同品类商品不足时，扩展到其他品类
        """
        results = []
        existing_ids = {r["product_id"] for r in existing_results}

        # 获取已存在的品类列表（避免重复）
        existing_categories = {r.get("category") for r in existing_results if r.get("category")}

        all_styles = [style] + similar_styles
        params = {
            "all_styles": all_styles,
            "min_sales": MIN_SALES_COUNT,
            "existing_categories": list(existing_categories)
        }

        # 查找这些风格下，其他品类的商品
        cypher = """
        MATCH (s:Style)
        WHERE s.name IN $all_styles
        MATCH (p:Product)-[:HAS_STYLE]->(s)
        WHERE p.sales_count >= $min_sales
        AND NOT p.category IN $existing_categories
        """

        if season and season != "all_season":
            cypher += " AND (p.season = $season OR p.season = 'all_season')"
            params["season"] = season

        cypher += f"""
        RETURN p.product_id AS product_id,
               p.category AS category,
               p.style AS style,
               p.season AS season,
               p.color AS color,
               p.sales_count AS sales_count,
               p.price AS price
        ORDER BY p.sales_count DESC
        LIMIT {top_k * 2}
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = session.run(cypher, params)
                for record in records:
                    product_id = record["product_id"]
                    if product_id not in existing_ids:
                        results.append({
                            "product_id": product_id,
                            "category": record["category"],
                            "style": record["style"],
                            "season": record["season"],
                            "color": record["color"],
                            "sales_count": record["sales_count"],
                            "price": record["price"],
                            "score": 0.5,  # 第3跳跨品类，较低分数
                            "match_reason": f"hop_3_cross_category:{record['category']}",
                            "hop_count": 3,
                            "hop_type": "cross_category"
                        })
        except Exception as e:
            print(f"    ! 第3跳查询失败: {e}")

        print(f"    [OK] 第3跳完成: 找到 {len(results)} 个新商品")
        return results

    def get_graph_stats(self) -> Dict[str, int]:
        """
        获取图谱统计信息

        Returns:
            统计信息字典，包含 node_count 和 relationship_count
        """
        if not self.is_connected():
            return {"node_count": 0, "relationship_count": 0}

        stats = {}
        node_count = 0
        rel_count = 0

        queries = {
            "total_products": "MATCH (p:Product) RETURN count(p) AS count",
            "total_categories": "MATCH (c:Category) RETURN count(c) AS count",
            "total_styles": "MATCH (s:Style) RETURN count(s) AS count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
        }

        try:
            with self.driver.session(database=self.database) as session:
                for name, cypher in queries.items():
                    result = session.run(cypher)
                    count = result.single()["count"]
                    stats[name] = count

                    # 累计节点数
                    if name != "total_relationships":
                        node_count += count
                    else:
                        rel_count = count

        except Exception as e:
            print(f"  ! 获取统计信息失败: {e}")

        # 添加混合检索器期望的键
        stats["node_count"] = node_count
        stats["relationship_count"] = rel_count

        return stats


if __name__ == "__main__":
    """
    测试代码：图谱检索
    """
    import argparse

    parser = argparse.ArgumentParser(description="电商服饰知识图谱检索器")
    parser.add_argument("--category", type=str, help="品类")
    parser.add_argument("--style", type=str, help="风格")
    parser.add_argument("--season", type=str, help="季节")
    parser.add_argument("--scene", type=str, help="场景")
    parser.add_argument("--top-k", type=int, default=3, help="返回数量")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")

    args = parser.parse_args()

    try:
        with FashionGraphRetriever() as retriever:
            if args.stats:
                stats = retriever.get_graph_stats()
                print("\n图谱统计:")
                for name, count in stats.items():
                    print(f"  {name}: {count}")
            elif args.category or args.style or args.season or args.scene:
                results = retriever.retrieve_by_graph(
                    category=args.category or "",
                    style=args.style or "",
                    season=args.season or "",
                    scene_hint=args.scene or "",
                    top_k=args.top_k
                )
                print(f"\n找到 {len(results)} 个结果")
            else:
                print("请指定检索条件或使用 --stats 查看统计信息")

    except Exception as e:
        print(f"[FAIL] 检索失败: {e}")
        print("\n请检查：")
        print("  1. Neo4j 是否启动")
        print("  2. 图谱是否已构建")
