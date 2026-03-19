# Nano Banana + Milvus + Neo4j 电商 AI 生图流水线

> 基于 LangGraph 多Agent 工作流、向量检索、知识图谱和 AIGC 的电商宣传图自动生成系统

---

## 项目背景

### 痛点问题

中小电商企业在商品宣传图制作中面临以下挑战：

- **成本高**：专业拍摄 + 后期修图，单张成本可达几十至上百元
- **周期长**：从新品上架到宣传图完成，通常需要数天时间
- **质量不稳定**：依赖设计师个人能力，风格难以统一

### 解决方案

本项目通过 **LangGraph 多Agent 工作流 + 并行混合检索 + 智能降级 + 风格分析 + 图像生成** 技术实现：

```
新品平铺图 → 多Agent工作流 → 并行混合检索 → 提取拍摄风格 → AI 生成宣传图
```

**核心价值**：
- 无需本地 GPU，所有模型通过 OpenRouter API 调用
- 端到端自动化，单张图片生成成本约 $0.014
- 基于 LangGraph 的多Agent 编排，流程清晰可扩展
- **并行混合检索**：Milvus 向量 + Neo4j 图谱 RRF 融合
- **智能降级机制**：无结果时自动放宽过滤条件
- **本地缓存**：风格分析结果自动缓存

### 第一性原则

**服装必须来自用户上传的图片**：系统生成的宣传图中，模特身上穿的服装必须是用户上传的原款服装。参考爆款图只用于提取**拍摄风格**（场景、光线、构图、氛围），不提供服装款式参考。

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **工作流编排** | LangGraph 1.0+ | 多Agent 状态机编排 |
| **向量数据库** | Milvus 2.4+ | 混合向量检索（Dense + Sparse + Scalar） |
| **图数据库** | Neo4j 5.0+ | 知识图谱、多跳推理 |
| **Embedding 模型** | nvidia/llama-nemotron-embed-vl-1b-v2 | 图文向量化（2048维）|
| **LLM 模型** | qwen/qwen3-vl-8b-instruct | 视觉风格分析 + 质量评估 |
| **图像生成** | black-forest-labs/flux.2-klein-4b | 宣传图生成 |
| **API 聚合** | OpenRouter | 统一模型调用接口 |
| **文本处理** | scikit-learn (TF-IDF) | 稀疏向量生成 |
| **后端框架** | FastAPI | RESTful API 服务 |
| **前端框架** | Vue 3 + Vite | Web 界面 |

---

## 架构设计

### LangGraph 多Agent 工作流架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph 多Agent 工作流架构                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                      START                                 │   │
│  └───────────────────────────────────┬───────────────────────┘   │
│                                      ▼                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                   UploadAgent                            │   │
│  │  • 接收文件字节流                                              │   │
│  │  • 验证图片格式                                                │   │
│  │  • 生成 product_id/task_id                                   │   │
│  └───────────────────────────────────┬───────────────────────┘   │
│                                      ▼                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                 EmbeddingAgent                           │   │
│  │  • NVIDIA Embedding 生成 Dense 向量 (2048维)             │   │
│  │  • TF-IDF 生成 Sparse 向量                                   │   │
│  └───────────────────────────────────┬───────────────────────┘   │
│                                      ▼                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              HybridRetrievalAgent                        │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  Milvus 向量检索 (权重0.6)                       │     │   │
│  │  │  • Dense + Sparse 混合搜索                       │     │   │
│  │  │  • 智能降级：无结果时放宽过滤条件               │     │   │
│  │  └─────────────────────────────────────────────────┘     │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  neo4j 检索 (权重0.4)                        │     │   │
│  │  │  • 多跳推理：3跳风格扩展                           │     │   │
│  │  │  • 智能降级：无结果时只按品类检索               │     │   │
│  │  └─────────────────────────────────────────────────┘     │   │
│  │  ───────── RRF 融合 ────────────────────────────────│     │   │
│  └───────────────────────────────────┬───────────────────────┘   │
│                                      ▼                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              StyleAnalysisAgent                        │   │
│  │  • Qwen3-VL 分析参考图拍摄风格                            │   │
│  │  • 本地缓存：相同参考图直接返回缓存结果                      │   │
│  └───────────────────────────────────┬───────────────────────┘   │
│                                      ▼                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                 ImageGenAgent                            │   │
│  │  • FLUX.2 Klein 生成宣传图                                  │   │
│  │  • 输入：用户服装图 + 风格描述                               │   │
│  └───────────────────────────────────┬───────────────────────┘   │
│                                      ▼                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              QualityJudgeAgent (可选)                     │   │
│  │  • 多模态 LLM 质量评估                                       │   │
│  │  • 决定是否需要重新生成                                     │   │
│  └───────────────────────────────────┬───────────────────────┘   │
│                                      ▼                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                  ResultAgent                             │   │
│  │  • 保存生成图片和参考图                                       │   │
│  │  • 保存风格 prompt 和质量评分                               │   │
│  └───────────────────────────────────┬───────────────────────┘   │
│                                      ▼                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                        END                              │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│  【异常处理】任何 Agent 失败 → FallbackAgent → ResultAgent → END    │
└─────────────────────────────────────────────────────────────────┘
```

### 并行混合检索架构

```
                    ┌─────────────────────────────────────┐
                    │         并行混合检索架构            │
                    ├─────────────────────────────────────┤
                    │                                     │
    ┌───────────────┴───────────────┐ ┌─────────────────┴───────────────┐
    │       Milvus 向量引擎         │ │      Neo4j 图谱引擎            │
    │  ┌─────────────────────────┐  │ │  ┌─────────────────────────┐  │
    │  │ Dense 向量 (视觉特征)   │  │ │  │ 多跳推理 (风格扩展)      │  │
    │  │ Sparse 向量 (文本特征)  │  │ │  │ 第1跳: 目标风格匹配      │  │
    │  │ Scalar 过滤 (销量等)    │  │ │  │ 第2跳: 相似风格扩展      │  │
    │  │ 智能降级: 无结果时放宽  │  │ │  │ 第3跳: 跨品类扩展        │  │
    │  └─────────────────────────┘  │ │  │ 智能降级: 无结果时只按  │  │
    └───────────────┬───────────────┘ │  │   品类检索               │  │
                    │                   │  └─────────────────────────┘  │
                    │                   │             │                 │
                    └───────────────────┴───────────────┘                 │
                                        ▼                                │
                    ┌─────────────────────────────────────────────────┐  │
                    │              RRF 结果融合                       │  │
                    │         score = 0.6/(60+r1) + 0.4/(60+r2)    │  │
                    └─────────────────────────────────────────────────┘  │
                                                              │         │
                                                              ▼         │
                                                ┌─────────────────────┐   │
                                                │   相似爆款商品       │   │
                                                │   (RRF排序去重)      │   │
                                                └─────────────────────┘   │
```

### 智能降级机制

当精确匹配找不到结果时，系统会自动降级：

```
┌─────────────────────────────────────────────────────────────┐
│                     智能降级流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  查询: category=midi_dress, style=elegant                      │
│        │                                                    │
│        ▼                                                    │
│  ┌─────────────────────────────────────────────────┐       │
│  │  Stage 1: 精确匹配                                  │       │
│  │  • Milvus: category + style + min_sales           │       │
│  │  • Neo4j: 多跳推理 (elegant → similar)         │       │
│  └───────────────┬─────────────────────────────────┘       │
│                    │                                           │
│              0 结果                                        │
│                    │                                           │
│                    ▼                                           │
│  ┌─────────────────────────────────────────────────┐       │
│  │  Stage 2: 智能降级                                  │       │
│  │  • Milvus: 只按 category 检索 (去掉style限制)      │       │
│  │  • Neo4j: 只按 category 检索 (去掉style限制)      │       │
│  └───────────────┬─────────────────────────────────┘       │
│                    │                                           │
│                    ▼                                           │
│              返回品类相关商品                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
nano-banana-milvus/
├── backend/                          # FastAPI 后端
│   ├── api.py                       # FastAPI 服务入口
│   ├── config.py                    # 配置管理（API Key、模型ID等）
│   ├── workflow.py                  # LangGraph 工作流定义
│   │
│   ├── agents/                      # Multi-Agent 模块
│   │   ├── state.py                 # PipelineState 状态定义
│   │   ├── base.py                  # BaseAgent 基类
│   │   ├── upload_agent.py          # 上传解析 Agent
│   │   ├── embedding_agent.py       # 向量编码 Agent
│   │   ├── hybrid_retrieval_agent.py # 混合检索 Agent
│   │   ├── style_analysis_agent.py  # 风格分析 Agent
│   │   ├── image_gen_agent.py        # 图像生成 Agent
│   │   ├── quality_judge_agent.py    # 质量评估 Agent
│   │   ├── result_agent.py           # 结果管理 Agent
│   │   └── fallback_agent.py         # 兜底 Agent
│   │
│   ├── graph/                       # Neo4j 知识图谱模块
│   │   ├── __init__.py
│   │   ├── graph_builder.py          # 图谱构建器
│   │   ├── graph_retriever.py        # 图谱检索器（多跳推理+降级）
│   │   └── hybrid_retriever.py       # 混合检索器（RRF融合）
│   │
│   ├── embedding.py                 # 向量嵌入生成
│   ├── retrieval.py                 # Milvus 检索（含降级逻辑）
│   ├── image_gen.py                 # 图像生成 + 质量评估 + 风格分析
│   ├── utils.py                     # 工具函数 + 缓存
│   │
│   ├── images/                      # 历史爆款图片（40条）
│   ├── new_products/                # 新品图片
│   ├── output/                      # 生成结果
│   ├── cache/                       # 本地缓存目录（风格分析等）
│   └── products.csv                 # 商品元数据
│
└── frontend/                         # Vue3 前端
    ├── index.html                    # 入口文件
    ├── package.json                  # 依赖配置
    ├── vite.config.js                # Vite 配置
    └── src/
        ├── main.js
        ├── App.vue
        └── components/
            ├── UploadForm.vue       # 上传表单
            └── ResultsGallery.vue    # 结果展示
```

### 核心模块说明

| 文件 | 作用 | 核心类/函数 |
|------|------|------------|
| `workflow.py` | LangGraph 工作流定义 | `create_workflow()`, `run_workflow()` |
| `agents/state.py` | 工作流状态定义 | `PipelineState`, `create_initial_state()` |
| `agents/hybrid_retrieval_agent.py` | 混合检索 Agent | `HybridRetrievalAgent.run()` |
| `graph/hybrid_retriever.py` | 混合检索器 | `retrieve_similar_bestsellers()` (RRF融合) |
| `graph/graph_retriever.py` | Neo4j 图谱检索器 | `multi_hop_retrieve()` (多跳推理+降级) |
| `retrieval.py` | Milvus 检索器 | `_single_retrieve()` (含降级逻辑) |
| `image_gen.py` | 图像生成模块 | `ImageGenerator`, `ImageQualityJudge` |
| `api.py` | FastAPI 服务 | `/api/upload`, `/api/tasks/{task_id}` |

---

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- Milvus 2.4+ (Docker)
- Neo4j 5.0+ (Docker)
- OpenRouter API Key

### 1. 启动 Milvus

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -v $(pwd)/backend/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

### 2. 启动 Neo4j

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/12345678 \
  -v $(pwd)/neo4j_data:/data \
  neo4j:latest
```

### 3. 配置环境变量

创建 `backend/.env` 文件：

```bash
OPENROUTER_API_KEY=your-key-here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
```

### 4. 启动后端

```bash
cd backend
pip install -r requirements.txt
python api.py
```

后端运行在 http://localhost:8000

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端运行在 http://localhost:3000

---

## API 使用

### 上传并生成

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@new_product.jpg" \
  -F "category=midi_dress" \
  -F "style=elegant" \
  -F "season=summer" \
  -F "retrieval_mode=hybrid" \
  -F "enable_multi_hop=true"
```

### 查询任务状态

```bash
curl "http://localhost:8000/api/tasks/{task_id}"
```

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/upload` | POST | 上传并生成 |
| `/api/tasks/{task_id}` | GET | 任务状态轮询 |
| `/api/output/{product_id}/{filename}` | GET | 获取生成图片 |
| `/api/categories` | GET | 品类列表 |
| `/api/styles` | GET | 风格列表 |

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `category` | string | 必填 | 商品品类 |
| `style` | string | 必填 | 商品风格 |
| `season` | string | all_season | 季节 |
| `scene_hint` | string | "" | 场景提示 |
| `retrieval_mode` | string | "hybrid" | 检索模式 |
| `enable_multi_hop` | boolean | true | 是否启用多跳推理 |
| `max_hops` | int | 3 | 最大跳数 |
| `enable_quality_check` | boolean | false | 是否启用质量评估 |

---

## 工作流日志示例

```
╔══════════════════════════════════════════════════════════╗
║              LangGraph Multi-Agent Workflow                  ║
╠═══════════════��════════════════════════════════════════╣
║  Task ID: ac4591d2......                                   ║
╚════════════════════════════════════════════════════════╝

[1/7] 初始化工作流组件...
[2/7] 创建工作流状态...
[3/7] 执行工作流...
      流程: Upload → Embedding → Retrieval → Style → ImageGen → Quality → Result

      [upload] 完成 - 进度: 20%
      [embedding] 完成 - 进度: 30%

============================================================
【混合检索】Milvus + Neo4j
============================================================
  多跳推理: 启用

[1/2] Milvus 向量检索...
  查询参数: category=midi_dress, min_sales=500, top_k=12
  循环检索: 禁用

找到 0 个相关且不同的爆款

[2/2] Neo4j 图谱检索...
  查询参数: category=midi_dress, style=elegant, season=all_season

============================================================
【多跳推理检索】风格扩展模式
============================================================
  起始风格: elegant
  品类过滤: midi_dress
  季节过滤: all_season
  最大跳数: 3

  >> 第1跳: 查找风格节点 'elegant'
    [OK] 第1跳完成: 找到 0 个商品

  [降级] 未找到匹配商品，尝试只按品类检索: midi_dress
  [降级成功] 找到 3 个 midi_dress 商品

  [完成] 融合后返回 3 个结果

      [hybrid_retrieval] 完成 - 进度: 50%
      [style_analysis] 完成 - 进度: 60%
      [image_gen] 完成 - 进度: 75%
      [quality_judge] 完成 - 进度: 85%
      [result] 完成 - 进度: 95%

============================================================
  Task Completed!
============================================================
```

---

## 智能降级机制

### Milvus 降级逻辑

```python
# 精确匹配：category + style + min_sales
filter_expr = 'category == "midi_dress" and style == "elegant" and sales_count > 500'

# 结果为空时，自动降级为：
filter_expr = 'category == "midi_dress"'  # 去掉风格和销量限制
```

### Neo4j 降级逻辑

```python
# 精确匹配：category + style (多跳推理)
MATCH (p:Product {category: "midi_dress"})-[:HAS_STYLE]->(s:Style {name: "elegant"})

# 结果为空时，自动降级为：
MATCH (p:Product {category: "midi_dress"})  # 只按品类检索
```

---

## 常见问题

**Q: 检索结果为空？**

A: 智能降级会自动放宽条件，先检查数据库是否有对应品类的商品。

**Q: Neo4j 连接失败？**

A: 系统会自动使用 Milvus 检索，不影响主流程。

**Q: 图像生成超时？**

A: 检查 OPENROUTER_API_KEY 是否有效，FLUX.2 模型可能需要代理访问。

---

## 更新日志

### v4.0.0 (2026-03)

**重构架构**：
- ✨ **LangGraph 多Agent 工作流**：7个 Agent 协同工作
- ✨ **并行混合检索架构**：RRF 融合 Milvus + Neo4j
- ✨ **智能降级机制**：无结果时自动放宽过滤条件
- ✨ **详细调试日志**：便于问题定位

**优化**：
- 🚀 禁用循环检索，提升检索速度
- 🎯 启用多跳推理，扩展召回能力
- 🔧 完善异常处理和错误追踪

### v3.0.0 (2026-03)

**新增功能**：
- ✨ **Neo4j 知识图谱模块**：支持多跳推理检索
- ✨ **两阶段检索架构**：Neo4j 多跳推理 + Milvus 向量精排
- ✨ **输入验证**：品类/风格/季节白名单验证

---

## 许可证

本项目仅供学习参考使用。
