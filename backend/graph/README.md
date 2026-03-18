# 电商服饰知识图谱模块

## 概述

本模块实现了基于 Neo4j 的电商服饰知识图谱，与 Milvus 向量检索配合，实现**双引擎混合检索**和**两阶段检索架构**。

## 核心特性

### 1. 多跳推理检索

通过图谱关系自动扩展候选集，解决"直接匹配无结果"问题：

```
第1跳: 目标风格直接匹配
  ↓
第2跳: 相似风格扩展 (通过 SIMILAR_STYLE 关系)
  ↓
第3跳: 跨品类扩展 (最大化召回)
```

### 2. 两阶段检索架构

```
Stage 1: Neo4j 多跳推理 → 获取候选集 (最多1000个商品)
Stage 2: Milvus 向量精排 → 在候选集中排序
```

### 3. 智能降级

- Neo4j 连接失败 → 自动使用 Milvus 全量检索
- 候选集为空 → 自动降级到 Milvus 全量检索

## 模块结构

```
backend/graph/
├── __init__.py              # 模块入口
├── graph_builder.py         # 图谱构建器
├── graph_retriever.py       # 图谱检索器（多跳推理）
├── hybrid_retriever.py      # 混合检索器（Milvus + Neo4j）
└── README.md               # 本文档
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
| `SIMILAR_STYLE` | 风格相似（商品间，用于多跳推理）|

## 检索策略

### 1. 多跳推理检索

**适用场景**：两阶段检索的 Stage 1

```python
# 启用多跳推理
results = retriever.multi_hop_retrieve(
    category="midi_dress",
    style="elegant",
    season="summer",
    max_hops=3,  # 最多3跳
    top_k=1000   # 获取大量候选
)

# 跳数分布示例：
# 1跳: 50个 (直接匹配 elegant 风格)
# 2跳: 30个 (相似风格扩展，如 romantic, classic)
# 3跳: 20个 (跨品类扩展，如 elegant 的 top 爆款)
```

### 2. 传统属性检索

```python
results = retriever.retrieve_by_graph(
    category="midi_dress",
    style="elegant",
    season="summer",
    top_k=3
)
```

### 3. 场景推理检索

```python
results = retriever.retrieve_by_graph(
    category="midi_dress",
    scene_hint="beach party",
    top_k=3
)
```

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

### 两阶段检索（推荐）

```python
from graph import HybridRetriever

# 创建混合检索器
retriever = HybridRetriever(
    milvus_weight=0.6,
    graph_weight=0.4
)

# 两阶段检索
results = retriever.two_stage_retrieve(
    query_dense=dense_vector,
    query_sparse=sparse_vector,
    category="midi_dress",
    style="elegant",
    season="summer",
    enable_multi_hop=True,  # 启用多跳推理
    max_hops=3,
    top_k=3
)
```

### 混合检索（并行）

```python
# Milvus + Neo4j 并行检索，RRF 融合
results = retriever.retrieve_similar_bestsellers(
    query_dense=dense_vector,
    query_sparse=sparse_vector,
    category="midi_dress",
    style="elegant",
    enable_multi_hop=True,
    enable_cycle=True,  # 启用循环检索状态机
    top_k=3
)
```

### 仅属性检索

```python
# 不需要向量，仅基于属性检索
results = retriever.retrieve_by_attributes(
    category="midi_dress",
    style="elegant",
    season="summer",
    enable_multi_hop=True,
    top_k=10
)
```

## API 参考

### HybridRetriever

| 方法 | 说明 |
|------|------|
| `two_stage_retrieve(...)` | 两阶段检索（Neo4j候选 + Milvus精排）|
| `retrieve_similar_bestsellers(...)` | 混合检索（并行 + RRF融合）|
| `retrieve_by_attributes(...)` | 仅属性检索 |

### FashionGraphRetriever

| 方法 | 说明 |
|------|------|
| `multi_hop_retrieve(...)` | 多跳推理检索 |
| `retrieve_by_graph(...)` | 传统属性检索 |
| `filter_candidate_products(...)` | 过滤候选商品 |
| `get_product_details(product_id)` | 获取商品详情 |
| `get_similar_products(product_id, top_k)` | 获取相似商品 |
| `get_graph_stats()` | 获取图谱统计 |

## 输入验证

系统使用白名单验证输入参数：

### 有效品类

```python
VALID_CATEGORIES = {
    'midi_dress', 'maxi_dress', 'mini_dress', 'skirt',
    'top', 'pants', 'jumpsuit', 'playsuit', 'romper',
    'bodysuit', 'overalls', 'dungarees'
}
```

### 有效风格

```python
VALID_STYLES = {
    'casual', 'formal', 'sporty', 'elegant', 'vintage',
    'modern', 'classic', 'boho', 'bohemian', 'romantic',
    'minimalist', 'edgy', 'preppy', 'chic', 'streetwear',
    'grunge', 'retro'
}
```

### 有效季节

```python
VALID_SEASONS = {
    'spring', 'summer', 'autumn', 'winter', 'all_season'
}
```

## 异常处理

- Neo4j 连接失败时，自动记录日志并降级到 Milvus 检索
- 无效输入时，记录警告日志并忽略该条件
- 检索失败时返回空列表，不影响主流程

## 性能优化

- 使用索引加速查询
- 批量插入支持
- 候选集缓存（减少重复检索）
- 智能降级机制
