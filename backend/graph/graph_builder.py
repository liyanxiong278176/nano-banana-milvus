"""
电商服饰知识图谱 - 图谱构建器

功能说明：
- 从 products.csv 和 images/ 目录构建 Neo4j 知识图谱
- 支持结构化数据（品类、风格、销量）直接抽取
- 支持非结构化数据（面料、场景、构图）通过 LLM 抽取
- 复用现有配置和工具函数

Schema 设计：
- 节点类型：Product(商品), Category(品类), Style(风格), Season(季节),
           Color(颜色), Material(面料), Scene(场景), Pose(姿势), Attribute(属性)
- 关系类型：HAS_CATEGORY, HAS_STYLE, HAS_SEASON, HAS_COLOR, HAS_MATERIAL,
           SUITABLE_SCENE, HAS_POSE, HAS_ATTRIBUTE, SIMILAR_STYLE, etc.

使用示例：
    >>> from graph import FashionGraphBuilder
    >>> builder = FashionGraphBuilder()
    >>> builder.create_schema()
    >>> builder.build_from_csv()
"""
import csv
import re
import json
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from neo4j import GraphDatabase, Driver as Neo4jDriver

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB,
    IMAGE_DIR, PRODUCT_CSV, LIGHT_LLM_MODEL,
    OPENROUTER_API_KEY, ENABLE_CACHE
)
from utils import image_to_uri, get_cache_key, save_to_cache, load_from_cache


class FashionGraphBuilder:
    """
    电商服饰知识图谱构建器

    负责从商品数据构建 Neo4j 知识图谱，支持实体抽取和关系构建。
    """

    # ==================== 节点标签定义 ====================
    NODE_LABELS = {
        "Product": "商品节点",
        "Category": "品类节点",
        "Style": "风格节点",
        "Season": "季节节点",
        "Color": "颜色节点",
        "Material": "面料节点",
        "Scene": "场景节点",
        "Pose": "姿势节点",
        "Attribute": "属性节点",
    }

    # ==================== 关系类型定义 ====================
    RELATIONSHIP_TYPES = {
        "HAS_CATEGORY": "商品属于品类",
        "HAS_STYLE": "商品具有风格",
        "HAS_SEASON": "商品适用季节",
        "HAS_COLOR": "商品具有颜色",
        "HAS_MATERIAL": "商品材质为",
        "SUITABLE_SCENE": "商品适合场景",
        "HAS_POSE": "商品展示姿势",
        "HAS_ATTRIBUTE": "商品具有属性",
        "SIMILAR_STYLE": "风格相似",
        "SAME_CATEGORY": "同品类",
    }

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        初始化图谱构建器，连接 Neo4j 数据库

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
            print(f"[OK] Neo4j 连接成功: {self.uri}")
            print(f"  数据库: {self.database}")
        except Exception as e:
            print(f"[FAIL] Neo4j 连接失败: {e}")
            raise ConnectionError(f"无法连接到 Neo4j: {e}")

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
            print("Neo4j 连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== Schema 创建 ====================

    def create_schema(self):
        """
        创建图谱 Schema（约束和索引）

        创建唯一性约束确保数据完整性：
        - Product.product_id
        - Category.category_name
        - Style.style_name
        - Season.season_name
        - Color.color_name
        - Material.material_name
        - Scene.scene_name

        创建索引提升查询性能：
        - Product 销量、价格索引
        - 各节点的名称索引
        """
        print("\n" + "=" * 60)
        print("创建知识图谱 Schema")
        print("=" * 60)

        with self.driver.session(database=self.database) as session:
            # ==================== 唯一性约束 ====================
            constraints = [
                # 商品节点约束
                ("Product", "product_id", "商品ID"),
                # 属性节点约束
                ("Category", "category_name", "品类名称"),
                ("Style", "style_name", "风格名称"),
                ("Season", "season_name", "季节名称"),
                ("Color", "color_name", "颜色名称"),
                ("Material", "material_name", "面料名称"),
                ("Scene", "scene_name", "场景名称"),
                ("Pose", "pose_name", "姿势名称"),
            ]

            for label, property_, name in constraints:
                try:
                    # 检查约束是否已存在
                    check_query = f"""
                    SHOW CONSTRAINTS
                    WHERE label = '{label}' AND properties = ['{property_}']
                    """
                    result = session.run(check_query)
                    exists = result.single() is not None

                    if not exists:
                        create_query = f"""
                        CREATE CONSTRAINT IF NOT EXISTS
                        FOR (n:{label}) REQUIRE n.{property_} IS UNIQUE
                        """
                        session.run(create_query)
                        print(f"  [OK] 创建约束: {label}.{property_}")
                    else:
                        print(f"  - 约束已存在: {label}.{property_}")
                except Exception as e:
                    print(f"  ! 约束创建警告 ({label}.{property_}): {e}")

            # ==================== 索引创建 ====================
            indexes = [
                # Product 节点索引
                ("Product", "sales_count", "销量"),
                ("Product", "price", "价格"),
                # 全文搜索索引
                ("Product", "description", "描述"),
            ]

            for label, property_, name in indexes:
                try:
                    # 检查索引是否已存在
                    check_query = f"""
                    SHOW INDEXES
                    WHERE labelName = '{label}' AND properties = ['{property_}']
                    """
                    result = session.run(check_query)
                    exists = result.single() is not None

                    if not exists:
                        create_query = f"""
                        CREATE INDEX IF NOT EXISTS
                        FOR (n:{label}) ON (n.{property_})
                        """
                        session.run(create_query)
                        print(f"  [OK] 创建索引: {label}.{property_}")
                    else:
                        print(f"  - 索引已存在: {label}.{property_}")
                except Exception as e:
                    print(f"  ! 索引创建警告 ({label}.{property_}): {e}")

        print("\n[OK] Schema 创建完成")

    # ==================== 实体抽取 ====================

    def extract_entity_triplets(
        self,
        product_dict: Dict[str, Any],
        product_image: Image.Image = None
    ) -> List[Tuple[str, str, str]]:
        """
        从商品数据中抽取实体三元组

        三元组格式：(实体1, 关系, 实体2)

        抽取策略：
        1. 结构化数据（品类、风格、季节、颜色、销量）直接抽取
        2. 非结构化数据（面料、场景、姿势）通过 LLM 抽取
        3. 支持缓存，相同商品避免重复 LLM 调用

        Args:
            product_dict: 商品元数据字典（来自 products.csv 的一行）
                包含: product_id, category, style, season, color, sales_count,
                      description, price 等
            product_image: 商品图片（可选，用于视觉特征抽取）

        Returns:
            三元组列表，格式：[(实体1, 关系, 实体2), ...]

            示例：
            [
                ("SKU001", "HAS_CATEGORY", "midi_dress"),
                ("SKU001", "HAS_STYLE", "elegant"),
                ("SKU001", "HAS_SEASON", "summer"),
                ("SKU001", "HAS_COLOR", "blue"),
                ("SKU001", "HAS_MATERIAL", "cotton"),
                ("SKU001", "SUITABLE_SCENE", "beach"),
                ...
            ]
        """
        triplets = []
        product_id = product_dict.get("product_id", "UNKNOWN")

        # ==================== 1. 结构化数据直接抽取 ====================

        # 品类关系
        category = product_dict.get("category", "")
        if category:
            triplets.append((product_id, "HAS_CATEGORY", category))

        # 风格关系
        style = product_dict.get("style", "")
        if style:
            triplets.append((product_id, "HAS_STYLE", style))

        # 季节关系
        season = product_dict.get("season", "all_season")
        if season:
            triplets.append((product_id, "HAS_SEASON", season))

        # 颜色关系
        color = product_dict.get("color", "")
        if color:
            # 处理多颜色情况（如 "red,blue"）
            colors = [c.strip() for c in color.split(",")]
            for c in colors:
                triplets.append((product_id, "HAS_COLOR", c))

        # 销量属性
        sales_count = int(product_dict.get("sales_count", 0) or 0)
        if sales_count:
            triplets.append((product_id, "HAS_ATTRIBUTE", f"high_sales" if sales_count > 1500 else "medium_sales"))

        # 价格区间
        price = float(product_dict.get("price", 0) or 0.0)
        if price:
            price_range = self._get_price_range(price)
            triplets.append((product_id, "HAS_ATTRIBUTE", f"price_{price_range}"))

        # ==================== 2. 非结构化数据 LLM 抽取 ====================

        # 检查缓存
        if ENABLE_CACHE and product_image:
            cache_content = f"{product_id}_{product_dict.get('description', '')}"
            cache_key = get_cache_key("graph_extraction", cache_content)
            cached_triplets = load_from_cache(cache_key)
            if cached_triplets is not None:
                # 将缓存的列表合并到三元组
                triplets.extend(cached_triplets)
                return triplets

        # LLM 抽取（仅在有图片时进行）
        if product_image:
            llm_triplets = self._extract_with_llm(product_dict, product_image)
            triplets.extend(llm_triplets)

            # 保存到缓存
            if ENABLE_CACHE:
                cache_key = get_cache_key("graph_extraction", cache_content)
                save_to_cache(cache_key, llm_triplets)

        return triplets

    def _get_price_range(self, price: float) -> str:
        """
        将价格转换为区间标签

        Args:
            price: 价格数值

        Returns:
            价格区间标签
        """
        if price < 30:
            return "low"
        elif price < 60:
            return "medium"
        else:
            return "high"

    def _extract_with_llm(
        self,
        product_dict: Dict[str, Any],
        product_image: Image.Image
    ) -> List[Tuple[str, str, str]]:
        """
        使用 LLM 从图片和描述中抽取非结构化三元组

        抽取维度：
        - Material（面料）: cotton, silk, linen, wool, etc.
        - Scene（场景）: beach, office, party, outdoor, etc.
        - Pose（姿势）: standing, sitting, walking, etc.

        Args:
            product_dict: 商品元数据
            product_image: 商品图片

        Returns:
            LLM 抽取的三元组列表
        """
        from openai import OpenAI
        import httpx

        triplets = []

        # 构建 LLM 请求
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=httpx.Timeout(60, connect=30)
        )

        # Few-shot Prompt 示例
        few_shot_prompt = """你是一位专业的时尚商品分析师，擅长从商品图片和描述中提取结构化知识。

请分析以下商品图片和描述，提取出以下维度的实体关系：

**提取维度：**
1. **Material（面料）**：如 cotton, silk, linen, wool, polyester, denim, knit, etc.
2. **Scene（场景）**：如 beach, office, party, outdoor, indoor, street, casual, formal, etc.
3. **Pose（姿势）**：如 standing, sitting, walking, dynamic, static, full_body, close_up, etc.

**输出格式：**
每行一个三元组，格式：实体1|关系|实体2

示例输出：
SKU001|HAS_MATERIAL|cotton
SKU001|SUITABLE_SCENE|beach
SKU001|HAS_POSE|standing

**商品信息：**
- 商品ID: {product_id}
- 品类: {category}
- 风格: {style}
- 描述: {description}

请只输出三元组，每行一个，不要其他内容。"""

        # 构建 content
        content = [
            {"type": "image_url", "image_url": {"url": image_to_uri(product_image, max_size=512)}},
            {"type": "text", "text": few_shot_prompt.format(
                product_id=product_dict.get("product_id", "UNKNOWN"),
                category=product_dict.get("category", ""),
                style=product_dict.get("style", ""),
                description=product_dict.get("description", "")
            )}
        ]

        try:
            response = client.chat.completions.create(
                model=LIGHT_LLM_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=300,
                temperature=0.3
            )

            result_text = response.choices[0].message.content.strip()

            # 解析 LLM 输出
            for line in result_text.split("\n"):
                line = line.strip()
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) == 3:
                        entity1, relation, entity2 = [p.strip() for p in parts]
                        # 标准化关系类型
                        relation = self._normalize_relation(relation)
                        if relation:
                            triplets.append((entity1, relation, entity2))

        except Exception as e:
            print(f"  ! LLM 抽取失败: {e}")

        return triplets

    def _normalize_relation(self, relation: str) -> Optional[str]:
        """
        标准化关系名称

        将各种可能的关系名称映射到标准关系类型

        Args:
            relation: 原始关系名称

        Returns:
            标准化后的关系名称，无效则返回 None
        """
        # 关系映射表
        relation_map = {
            # 面料关系
            "HAS_MATERIAL": ["HAS_MATERIAL", "MATERIAL", "材质", "面料", "FABRIC"],
            # 场景关系
            "SUITABLE_SCENE": ["SUITABLE_SCENE", "SCENE", "场景", "场合", "SUITABLE_FOR"],
            # 姿势关系
            "HAS_POSE": ["HAS_POSE", "POSE", "姿势", "姿态", "POSTURE"],
            # 属性关系
            "HAS_ATTRIBUTE": ["HAS_ATTRIBUTE", "ATTRIBUTE", "属性", "FEATURE"],
        }

        relation_upper = relation.upper()
        for standard, variants in relation_map.items():
            if relation_upper in [v.upper() for v in variants]:
                return standard

        return None

    # ==================== 数据插入 ====================

    def insert_product(
        self,
        product_dict: Dict[str, Any],
        product_image: Image.Image = None
    ):
        """
        插入单个商品到知识图谱

        流程：
        1. 抽取三元组
        2. 创建/更新节点
        3. 创建关系

        Args:
            product_dict: 商品元数据字典
            product_image: 商品图片（可选，用于 LLM 抽取）
        """
        product_id = product_dict.get("product_id")
        if not product_id:
            print("  ! 商品 ID 缺失，跳过")
            return

        # 抽取三元组
        triplets = self.extract_entity_triplets(product_dict, product_image)

        with self.driver.session(database=self.database) as session:
            # ==================== 1. 创建/更新 Product 节点 ====================
            create_product_query = """
            // 创建或更新商品节点
            MERGE (p:Product {product_id: $product_id})
            SET p.category = $category,
                p.style = $style,
                p.season = $season,
                p.color = $color,
                p.sales_count = $sales_count,
                p.price = $price,
                p.description = $description
            """
            session.run(create_product_query, {
                "product_id": product_id,
                "category": product_dict.get("category", ""),
                "style": product_dict.get("style", ""),
                "season": product_dict.get("season", ""),
                "color": product_dict.get("color", ""),
                "sales_count": int(product_dict.get("sales_count", 0)),
                "price": float(product_dict.get("price", 0.0)),
                "description": product_dict.get("description", ""),
            })

            # ==================== 2. 处理三元组，创建节点和关系 ====================
            for entity1, relation, entity2 in triplets:
                self._create_triplet(session, entity1, relation, entity2)

            # ==================== 3. 创建同风格商品关系 ====================
            # 找到相同风格的其他商品，建立 SIMILAR_STYLE 关系
            style = product_dict.get("style", "")
            if style:
                similar_style_query = """
                // 找到相同风格的其他商品
                MATCH (p1:Product {product_id: $product_id})
                MATCH (p1)-[:HAS_STYLE]->(s:Style {style_name: $style})
                MATCH (p2:Product)-[:HAS_STYLE]->(s)
                WHERE p2.product_id <> $product_id
                AND p2.sales_count > 1000  // 只关联高销量商品
                MERGE (p1)-[r:SIMILAR_STYLE]-(p2)
                SET r.weight = CASE
                    WHEN p1.category = p2.category THEN 1.0  // 同品类同风格，权重高
                    ELSE 0.5
                END
                """
                session.run(similar_style_query, {
                    "product_id": product_id,
                    "style": style
                })

    def _create_triplet(
        self,
        session,
        entity1: str,
        relation: str,
        entity2: str
    ):
        """
        创建单个三元组（节点和关系）

        Args:
            session: Neo4j 会话
            entity1: 实体1名称
            relation: 关系类型
            entity2: 实体2名称
        """
        # 根据关系类型确定实体2的节点标签
        node_label_map = {
            "HAS_CATEGORY": "Category",
            "HAS_STYLE": "Style",
            "HAS_SEASON": "Season",
            "HAS_COLOR": "Color",
            "HAS_MATERIAL": "Material",
            "SUITABLE_SCENE": "Scene",
            "HAS_POSE": "Pose",
            "HAS_ATTRIBUTE": "Attribute",
        }

        label = node_label_map.get(relation, "Attribute")

        # 创建属性节点和关系
        cypher = f"""
        // 创建或更新属性节点
        MERGE (n:{label} {{name: $entity2}})

        // 创建商品与属性的关系
        WITH n
        MATCH (p:Product {{product_id: $entity1}})
        MERGE (p)-[r:{relation}]->(n)
        """
        session.run(cypher, {
            "entity1": entity1,
            "entity2": entity2
        })

    # ==================== 批量构建 ====================

    def build_from_csv(self):
        """
        从 products.csv 批量构建知识图谱

        流程：
        1. 读取 products.csv
        2. 遍历每个商品
        3. 加载对应图片
        4. 插入图谱

        注意：
        - 自动去重，重复商品会更新而非报错
        - 进度条显示构建进度
        - 失败商品记录日志，不影响整体流程
        """
        print("\n" + "=" * 60)
        print("从 products.csv 批量构建知识图谱")
        print("=" * 60)

        # 检查文件
        if not PRODUCT_CSV.exists():
            print(f"[FAIL] 商品文件不存在: {PRODUCT_CSV}")
            return

        # 读取商品数据
        products = []
        with open(PRODUCT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append(row)

        print(f"  读取到 {len(products)} 个商品")

        # 批量插入
        success_count = 0
        skip_count = 0

        for product in tqdm(products, desc="构建图谱"):
            product_id = product.get("product_id")

            # 加载图片
            try:
                img_path = IMAGE_DIR / f"{product_id}.jpg"
                if not img_path.exists():
                    img_path = IMAGE_DIR / f"{product_id}.png"

                if img_path.exists():
                    from utils import load_image
                    product_image = load_image(str(img_path))
                else:
                    product_image = None
                    print(f"\n  ! 图片未找到: {product_id}")

                # 插入商品
                self.insert_product(product, product_image)
                success_count += 1

            except Exception as e:
                print(f"\n  [FAIL] 插入失败 ({product_id}): {e}")
                skip_count += 1

        print(f"\n[OK] 图谱构建完成")
        print(f"  成功: {success_count} 个")
        print(f"  跳过: {skip_count} 个")

        # 输出图谱统计
        self._print_graph_stats()

    def _print_graph_stats(self):
        """输出图谱统计信息"""
        with self.driver.session(database=self.database) as session:
            stats_queries = {
                "商品节点": "MATCH (n:Product) RETURN count(n) AS count",
                "品类节点": "MATCH (n:Category) RETURN count(n) AS count",
                "风格节点": "MATCH (n:Style) RETURN count(n) AS count",
                "季节节点": "MATCH (n:Season) RETURN count(n) AS count",
                "场景节点": "MATCH (n:Scene) RETURN count(n) AS count",
                "总关系数": "MATCH ()-[r]->() RETURN count(r) AS count",
            }

            print(f"\n{'='*60}")
            print("图谱统计")
            print(f"{'='*60}")

            for name, query in stats_queries.items():
                result = session.run(query)
                count = result.single()["count"]
                print(f"  {name}: {count}")

    # ==================== 工具方法 ====================

    def clear_graph(self):
        """
        清空整个图谱

        [WARN]️ 警告：此操作不可逆!
        """
        with self.driver.session(database=self.database) as session:
            # 删除所有节点和关系
            session.run("MATCH (n) DETACH DELETE n")
            print("[OK] 图谱已清空")

    def export_schema(self, output_path: str = None):
        """
        导出图谱 Schema 为 JSON 文件

        Args:
            output_path: 输出路径，默认为 PROJECT_ROOT/graph_schema.json
        """
        from config import PROJECT_ROOT

        if output_path is None:
            output_path = PROJECT_ROOT / "graph_schema.json"

        schema_info = {
            "node_labels": self.NODE_LABELS,
            "relationship_types": self.RELATIONSHIP_TYPES,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema_info, f, ensure_ascii=False, indent=2)

        print(f"[OK] Schema 已导出: {output_path}")


if __name__ == "__main__":
    """
    测试代码：构建知识图谱
    """
    import argparse

    parser = argparse.ArgumentParser(description="电商服饰知识图谱构建器")
    parser.add_argument("--create-schema", action="store_true", help="创建 Schema")
    parser.add_argument("--build", action="store_true", help="从 CSV 构建图谱")
    parser.add_argument("--clear", action="store_true", help="清空图谱")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")

    args = parser.parse_args()

    try:
        with FashionGraphBuilder() as builder:
            if args.create_schema:
                builder.create_schema()

            if args.clear:
                confirm = input("确认清空图谱？(yes/no): ")
                if confirm.lower() == "yes":
                    builder.clear_graph()

            if args.build:
                builder.create_schema()
                builder.build_from_csv()

            if args.stats:
                builder._print_graph_stats()

    except ConnectionError as e:
        print(f"[FAIL] 连接失败: {e}")
        print("\n请检查：")
        print("  1. Neo4j 是否启动 (docker ps | grep neo4j)")
        print("  2. 连接配置是否正确 (config.NEO4J_URI)")
        print("  3. 密码是否正确 (.env NEO4J_PASSWORD)")
