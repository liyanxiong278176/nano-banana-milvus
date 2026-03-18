# 电商服饰知识图谱模块

## 概述

本模块实现了基于 Neo4j 的电商服饰知识图谱，与现有 Milvus 向量检索配合，实现**双引擎混合检索**。

## 模块结构

```
backend/graph/
├── __init__.py           # 模块入口
├── graph_builder.py      # 图谱构建器
├── graph_retriever.py    # 图谱检索器
├── example_usage.py      # 使用示例
└── README.md            # 本文档
```

## Schema 设计

### 节点类型

| 标签 | 说明 | 属性 |
|------|------|------|
| `Product` | 商品节点 | product_id, category, style, season, color, sales_count, price, description |
| `Category` | 品类节点 | category_name |
| `Style` | 风格节点 | style_name |
| `Season` | 季节节点 | season_name |
| `Color` | 颜色节点 | color_name |
| `Material` | 面料节点 | material_name |
| `Scene` | 场景节点 | scene_name |
| `Pose` | 姿势节点 | pose_name |
| `Attribute` | 属性节点 | name |

### 关系类型

| 关系 | 说明 |
|------|------|
| `HAS_CATEGORY` | 商品属于品类 |
| `HAS_STYLE` | 商品具有风格 |
| `HAS_SEASON` | 商品适用季节 |
| `HAS_COLOR` | 商品具有颜色 |
| `HAS_MATERIAL` | 商品材质为 |
| `SUITABLE_SCENE` | 商品适合场景 |
| `HAS_POSE` | 商品展示姿势 |
| `HAS_ATTRIBUTE` | 商品具有属性 |
| `SIMILAR_STYLE` | 风格相似（商品间） |

## 快速开始

### 1. 安装依赖

```bash
pip install neo4j>=5.0.0
```

### 2. 配置环境变量

在 `.env` 文件中添加：

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
NEO4J_DB=fashion_graph
```

### 3. 启动 Neo4j

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -v $(pwd)/neo4j_data:/data \
  neo4j:latest
```

### 4. 构建知识图谱

```bash
cd backend
python -m graph.graph_builder --create-schema --build
```

## 使用示例

### 图谱构建

```python
from graph import FashionGraphBuilder

# 创建 Schema
with FashionGraphBuilder() as builder:
    builder.create_schema()
    builder.build_from_csv()  # 从 products.csv 批量构建
```

### 图谱检索

```python
from graph import FashionGraphRetriever

with FashionGraphRetriever() as retriever:
    # 精确匹配检索
    results = retriever.retrieve_by_graph(
        category="midi_dress",
        style="elegant",
        season="summer",
        top_k=3
    )

    # 场景推理检索
    results = retriever.retrieve_by_graph(
        category="midi_dress",
        scene_hint="beach party",
        top_k=3
    )
```

### 混合检索（向量 + 图谱）

```python
from graph import FashionGraphRetriever
from retrieval import BestsellerRetriever

# 图谱检索
with FashionGraphRetriever() as graph_retriever:
    graph_results = graph_retriever.retrieve_by_graph(...)

# 向量检索
vector_retriever = BestsellerRetriever()
vector_results = vector_retriever.retrieve_similar_bestsellers(...)

# 结果融合
fused_results = merge_results(graph_results, vector_results)
```

## API 参考

### FashionGraphBuilder

| 方法 | 说明 |
|------|------|
| `create_schema()` | 创建约束和索引 |
| `extract_entity_triplets(product_dict, product_image)` | 抽取三元组 |
| `insert_product(product_dict, product_image)` | 插入单个商品 |
| `build_from_csv()` | 批量构建图谱 |
| `clear_graph()` | 清空图谱 |
| `export_schema(path)` | 导出 Schema |

### FashionGraphRetriever

| 方法 | 说明 |
|------|------|
| `retrieve_by_graph(category, style, season, scene_hint, top_k)` | 多跳推理检索 |
| `get_product_details(product_id)` | 获取商品详情 |
| `get_similar_products(product_id, top_k)` | 获取相似商品 |
| `get_graph_stats()` | 获取图谱统计 |

## 检索策略

### 1. 直接匹配
- 匹配品类、风格、季节完全相同的商品
- 按销量排序

### 2. 场景推理
- 通过场景关键词找到适合的商品
- 支持模糊匹配（如 beach → sea, ocean, vacation）

### 3. 同风格关联
- 找到同风格的其他爆款
- 同品类优先

## 异常处理

- Neo4j 连接失败时，自动记录日志
- 检索失败时返回空列表，不影响主流程
- 可单独使用 Milvus 向量检索兜底

## 性能优化

- 使用索引加速查询
- 批量插入支持
- 缓存 LLM 抽取结果
