# Nano Banana + Milvus 电商 AI 生图流水线

> 基于 LangGraph 多 Agent 工作流、向量检索、AIGC 的电商宣传图自动生成系统

---

## 目录

- [项目简介](#项目简介)
- [核心价值](#核心价值)
- [技术架构](#技术架构)
- [工作流程](#工作流程)
- [指标量化](#指标量化)
- [快速开始](#快速开始)
- [API 使用](#api-使用)
- [项目结构](#项目结构)

---

## 项目简介

### 痛点问题

中小电商企业在商品宣传图制作中面临以下挑战：

| 痛点 | 传统方式 | 影响 |
|------|----------|------|
| 成本高 | 专业拍摄 + 后期修图 | 单张成本 ¥50-200 |
| 周期长 | 从新品上架到宣传图完成 | 通常需要 2-5 天 |
| 质量不稳定 | 依赖设计师个人能力 | 风格难以统一 |

### 解决方案

```
新品平铺图 → AI 分析 → 检索爆款风格 → 生成宣传图
```

**核心原则**：模特身上的服装必须是用户上传的原款，参考爆款图只用于提取拍摄风格（场景、光线、构图、氛围）。

---

## 核心价值

- **低成本**：单张图片生成成本约 $0.014（约 ¥0.1）
- **无 GPU 依赖**：所有模型通过 OpenRouter API 调用
- **端到端自动化**：从上传到生成全程无需人工干预
- **智能检索**：循环检索状态机，最多 3 轮查询优化
- **质量保证**：LLM 质量评估驱动检索优化
- **异步处理**: 后台任务执行，前端轮询进度

---

## 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           前端 (Vue 3)                              │
│                      UploadForm + ResultsGallery                    │
└────────────────────────────────────┬────────────────────────────────┘
                                     │ HTTP (RESTful API)
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI 后端服务                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  LangGraph 多 Agent 工作流 v2.0               │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │  UploadAgent → RetrievalAgent → GenAgent → QualityJudgeAgent │   │
│  │       ↓               ↓              ↓            ↓            │   │
│  │  ResultAgent ←───────────────────────────────────────────────┘   │
│  └───────────────────────────────────┬─────────────────────────┘   │
│                                      ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              TaskManager 任务管理器                         │   │
│  │  • 任务生命周期: PENDING → QUEUED → RUNNING → COMPLETED    │   │
│  │  • 并发控制: Semaphore-based (max_concurrent=5)            │   │
│  │  • 进度跟踪: 实时更新 + 前端轮询                            │   │
│  │  • API: GET /api/tasks/{task_id}                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            数据层                                    │
│  • Milvus: 向量数据库 (Dense + BM25 Sparse + Scalar)              │
│  • 文件系统: 图片存储 / 缓存 / 输出                                 │
└──────────────────────��──────────────────────────────────────────────┘
```
### 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 工作流编排 | LangGraph 1.0+ | 多 Agent 状态机编排 |
| 向量数据库 | Milvus 2.4+ | 混合向量检索 |
| Embedding | nvidia/llama-nemotron-embed-vl-1b-v2 | 图文向量化（2048维）|
| 视觉 LLM | qwen/qwen3-vl-8b-instruct | 风格分析 + 质量评估 |
| 图像生成 | black-forest-labs/flux.2-klein-4b | FLUX.2 生图 |
| API 聚合 | OpenRouter | 统一模型调用接口 |
| 文本处理 | rank-bm25 (BM25) + scikit-learn | 稀疏向量生成 |
| 后端框架 | FastAPI | RESTful API 服务 |
| 前端框架 | Vue 3 + Vite | Web 界面 |

---

## 工作流程

### 整体流程图

```
用户上传图片
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                    1. 创建任务 (TaskManager)               │
│  • 文件类型验证                                              │
│  • TaskManager.register_task() 注册任务                     │
│  • 并发控制: Semaphore (max_concurrent=5)                   │
│  • 任务状态: PENDING → QUEUED                               │
│  • 立即返回 task_id（异步执行）                              │
└────────────────────────────────────┬────────────────────────┘
                                     │
                                     ▼ (获取执行许可后)
┌─────────────────────────────────────────────────────────────┐
│              2. LangGraph 多 Agent 工作流 v2.0               │
│              (TaskManager.update_task_status() 实时更新进度)       │
├─────────────────────────────────��───────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ① UploadAgent (20%)                                    │    │
│  │     接收文件 → 验证格式 → 生成 ID                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ② RetrievalAgent (50%) 【合并】                        │    │
│  │     • Dense 向量: NVIDIA Embedding (2048维)             │    │
│  │     • Sparse 向量: BM25 (k1=1.5, b=0.75)                               │    │
│  │     • Milvus 混合检索 + 循环状态机                       │    │
│  │     • LLM 质量评估驱动查询优化                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ③ GenAgent (75%) 【合并: 风格分析 + 图像生成】         │
│  │     • Qwen3-VL 分析参考图拍摄风格                       │
│  │     • FLUX.2 Klein 生成宣传图                           │
│  │     • 本地缓存（相同参考图直接返回）                    │
│  └─────────────────────────────────────────────────────────┘
│                           │
│                           ▼
│  ┌────────────────��──────────────────────────────────────────┐
│  │  ④ QualityJudgeAgent (85%, 可选)                      │
│  │     • 多模态 LLM 质量评估                               │
│  │     • 决定是否重新生成（最多1次）                        │
│  └─────────────────────────────────────────────────────────┘
│                           │
│                           ▼
│  ┌─────────────────────────────────────────────────────────┐
│  │  ⑤ ResultAgent (95%)                                   │
│  │     • 保存生成图片和参考图                              │
│  │     • 保存风格 prompt 和质量评分                        │
│  └─────────────────────────────────────────────────────────┘
│                                                                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    5. 进度查询（前端轮询）                    │
│  • 前端每1秒轮询 GET /api/tasks/{task_id}                │
│  • TaskManager.get_task_status() 返回实时状态            │
│  • 状态: QUEUED → RUNNING → COMPLETED/FAILED             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    6. 结果返回                                │
│  • 状态: completed                                           │
│  • 进度: 100%                                                │
│  • 生成图片 URL                                              │
│  • 参考图 URL                                                │
│  • 风格描述                                                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ (5分钟后)
┌─────────────────────────────────────────────────────────────┐
│                    7. 资源清理 (TaskManager)                │
│  • TaskManager 延迟清理 (默认5分钟后)                        │
│  • 删除临时中间文件                                          │
│  • 保留: original.png, generated_*.png, reference_*.png     │
│  • 任务记录保留在内存中（支持查询历史任务）                   │
└─────────────────────────────────────────────────────────────┘
```

### 循环检索状态机详解

```
┌─────────────────────────────────────────────────────────────────┐
│                     循环检索状态机                               │
│                     (最多 3 轮优化)                              │
└─────────────────────────────────────────────────────────────────┘

第 1 轮: 原始条件检索
┌─────────────────────────────────────────────────────────────┐
│ 过滤条件: category == "midi_dress" and sales_count > 500   │
│ 检索执行: Milvus 混合检索                                    │
│ 质量评估: LLM 多维度评分                                     │
│                                                             │
│ ┌─────────────────┐                                        │
│ │ 平均分 >= 7.0?  │──Yes──▶ 返回结果                        │
│ └────────┬────────┘                                        │
│          │ No                                               │
│          ▼                                                  │
└─────────────────────────────────────────────────────────────┘

第 2 轮: 查询重写（第 1 次）
┌─────────────────────────────────────────────────────────────┐
│ 重写策略:                                                    │
│   • 品类匹配低 → midi_dress → dress (父类)                  │
│   • 销量阈值: 500 → 1000                                    │
│                                                             │
│ 过滤条件: category like "dress%" and sales_count > 1000    │
│ 检索执行 → 质量评估                                          │
│                                                             │
│ ┌─────────────────┐                                        │
│ │ 平均分 >= 7.0?  │──Yes──▶ 返回结果                        │
│ └────────┬────────┘                                        │
│          │ No                                               │
│          ▼                                                  │
└─────────────────────────────────────────────────────────────┘

第 3 轮: 查询重写（第 2 次 - 最大化召回）
┌─────────────────────────────────────────────────────────────┐
│ 重写策略:                                                    │
│   • 去掉品类限制                                             │
│   • 最低销量: 500                                           │
│                                                             │
│ 过滤条件: sales_count > 500                                │
│ 检索执行 → 质量评估                                          │
│                                                             │
│ ┌─────────────────┐                                        │
│ │ 返回最佳结果    │                                         │
│ └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 指标量化

### 成本指标

| 项目 | 模型 | 成本 | 备注 |
|------|------|------|------|
| Embedding | nvidia/llama-nemotron-embed-vl-1b-v2 | 免费 | NVIDIA 免费 API |
| 视觉分析 | qwen/qwen3-vl-8b-instruct | 免费 | OpenRouter 免费 |
| 图像生成 | flux.2-klein-4b | ~$0.014/张 | 主要成本 |
| 质量评估 | qwen/qwen3-vl-8b-instruct | 免费 | 可选功能 |

**单张总成本**: 约 $0.014（约 ¥0.1）

### 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 并发处理数 | 5 | 可配置 |
| 队列长度 | 10 | 可配置 |
| 平均处理时间 | 30-60秒 | 取决于网络和模型负载 |
| 缓存命中率 | 可变 | 相同参考图复用风格分析 |

### 质量指标

| 维度 | 评分标准 | 阈值 |
|------|----------|------|
| 品类匹配 | 0-10分 | - |
| 风格匹配 | 0-10分 | - |
| 场景匹配 | 0-10分 | - |
| 属性匹配 | 0-10分 | - |
| **平均分** | **0-10分** | **>= 7.0 满足条件** |

### 资源指标

| 资源 | 默认限制 | 说明 |
|------|----------|------|
| 最大文件大小 | 10MB | 可配置 |
| 最小文件大小 | 100 bytes | 防止空文件 |
| 缓存大小 | 500MB | 自动清理 |
| 资源清理延迟 | 5分钟 | 可配置 |

---

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- Docker（用于 Milvus）
- OpenRouter API Key

### 1. 获取 API Key

访问 [OpenRouter](https://openrouter.ai/keys) 获取免费 API Key。

### 2. 启动 Milvus

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -v $(pwd)/backend/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

### 3. 配置环境变量

创建 `backend/.env` 文件：

```bash
# 必填：OpenRouter API Key
OPENROUTER_API_KEY=your-openrouter-api-key-here

# 可选配置
# SPARSE_METHOD=bm25  # 稀疏向量方法: tfidf 或 bm25 (默认 bm25)
# OVERRUN_MODEL_TIER=standard
# ENABLE_CACHE=true
```

### 4. 安装依赖并启动后端

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

### 6. 访问应用

打开浏览器访问 http://localhost:3000，上传图片开始使用。

---

## API 使用

### 核心接口

#### 上传并生成

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@new_product.jpg" \
  -F "category=midi_dress" \
  -F "style=elegant" \
  -F "season=summer" \
  -F "scene_hint=beach"
```

**响应示例**:
```json
{
  "task_id": "task_abc123",
  "status": "pending",
  "product_id": "NEW_xxx",
  "message": "图片上传成功，正在处理中"
}
```

#### 查询任务状态

```bash
curl "http://localhost:8000/api/tasks/task_abc123"
```

**响应示例**:
```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "product_id": "NEW_xxx",
    "generated_images": ["/api/output/..."],
    "reference_images": ["/api/output/..."],
    "style_prompt": "..."
  }
}
```

### 完整接口列表

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/upload` | POST | 上传并生成 |
| `/api/tasks/{task_id}` | GET | 任务状态轮询 |
| `/api/stats` | GET | 系统统计信息 |
| `/ws/{task_id}` | WebSocket | 预留接口（当前使用轮询）|
| `/api/output/{product_id}/{filename}` | GET | 获取生成图片 |
| `/api/categories` | GET | 品类列表 |
| `/api/styles` | GET | 风格列表 |

### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | 是 | - | 新品平铺图片 |
| `category` | string | 是 | - | 商品品类 |
| `style` | string | 是 | - | 商品风格 |
| `season` | string | 否 | all_season | 季节 |
| `scene_hint` | string | 否 | "" | 场景提示 |
| `enable_quality_check` | boolean | 否 | false | 是否启用质量评估 |
| `use_workflow` | boolean | 否 | true | 是否使用 LangGraph 工作流 |

---

## 项目结构

```
nano-banana-milvus/
├── backend/
│   ├── api.py                    # FastAPI 服务入口
│   ├── main.py                   # 主程序入口
│   ├── config.py                 # 配置管理
│   │
│   ├── agents/                   # Multi-Agent 模块
│   │   ├── state.py              # PipelineState 状态定义
│   │   ├── base.py               # BaseAgent 基类
│   │   ├── upload_agent.py       # 上传解析 Agent
│   │   ├── retrieval_agent.py    # 检索 Agent (合并)
│   │   ├── gen_agent.py          # 生图 Agent (合并)
│   │   ├── quality_judge_agent.py # 质量评估 Agent
│   │   ├── result_agent.py       # 结果管理 Agent
│   │   ├── fallback_agent.py     # 兜底 Agent
│   │   # 以下保留向后兼容
│   │   ├── embedding_agent.py    # 向量编码 Agent
│   │   ├── hybrid_retrieval_agent.py
│   │   ├── style_analysis_agent.py
│   │   └── image_gen_agent.py
│   │
│   ├── workflow/                 # 工作流模块
│   │   └── core.py               # LangGraph 工作流定义
│   │
│   ├── vectorization/            # 向量化模块
│   │   ├── embedding.py          # 向量嵌入生成 (TF-IDF/BM25)
│   │   └── bm25.py               # BM25 稀疏向量
│   │
│   ├── retrieval/                # 检索模块
│   │   ├── retrieval.py          # Milvus 检索 + 循环状态机
│   │   └── wrapper.py            # 检索包装器
│   │
│   ├── generation/               # 生成模块
│   │   └── image_gen.py         # 图像生成 + 风格分析
│   │
│   ├── prompts/                  # 提示词模块
│   │   ├── prompts.py            # 提示词基础
│   │   ├── v2.py                 # 提示词 v2 (A/B测试)
│   │   └── v3.py                 # 提示词 v3 (用户反馈)
│   │
│   ├── utils/                    # 工具模块
│   │   ├── core.py               # 核心工具函数
│   │   └── console.py            # 控制台工具
│   │
│   ├── log_config/               # 日志配置模块
│   │   └── __init__.py           # 日志设置 (按天切分，保留30天)
│   │
│   └── task/                     # 任务管理模块
│       ├── __init__.py           # 模块导出
│       ├── limiter.py            # 并发控制 (Semaphore)
│       ├── record.py             # 任务记录模型 (TaskStatus, TaskRecord)
│       └── manager.py            # 任务管理器 (TaskManager)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadForm.vue    # 上传表单
│   │   │   └── ResultsGallery.vue # 结果展示
│   │   ├── App.vue
│   │   └── main.js
│   └── package.json
│
└── README.md
```

### 核心模块说明

| 模块 | 文件 | 核心类/函数 |
|------|------|-------------|
| 工作流 | `workflow/core.py` | `create_workflow_v2()` |
| 向量检索 | `retrieval/retrieval.py` | `retrieve_similar_bestsellers()` |
| 检索包装 | `retrieval/wrapper.py` | `RetrievalWrapper` |
| BM25 向量化 | `vectorization/bm25.py` | `BM25Vectorizer` |
| 向量嵌入 | `vectorization/embedding.py` | `EmbeddingGenerator` |
| ���像生成 | `generation/image_gen.py` | `ImageGenerator` |
| 风格分析 | `generation/image_gen.py` | `analyze_reference_images_style()` |
| 提示词 | `prompts/prompts.py` | `PromptBuilder` |
| 任务管理 | `task/manager.py` | `TaskManager`, `get_task_manager()` |
| 任务记录 | `task/record.py` | `TaskRecord`, `TaskStatus` |
| 并发控制 | `task/limiter.py` | `TaskLimiter`, `get_limiter()` |
| 日志配置 | `log_config/__init__.py` | `setup_logging()`, `get_logger()` |

### 日志系统

- **日志目录**: `backend/log_config/`
- **按天切分**: 每天午夜自动创建新日志文件
- **保留策略**: 自动清理30天前的日志
- **日志格式**: `[时间] [级别] [模块] 消息`
- **输出目标**: 同时输出到文件和控制台
- **日志文件**:
  - `api_YYYYMMDD.log` - API 服务日志
  - `task_YYYYMMDD.log` - 任务管理器日志
  - `limiter_YYYYMMDD.log` - 并发限制器日志
  - `workflow_YYYYMMDD.log` - 工作流日志

## 版本历史

### v7.2.0 (2026-03) - TaskManager 集成 + 日志系统
- **任务管理升级**: 简单字典 → TaskManager
  - 企业级任务生命周期管理 (PENDING → QUEUED → RUNNING → COMPLETED/FAILED)
  - 并发控制: Semaphore-based (max_concurrent=5, max_queue_size=10)
  - 进度跟踪: 实时更新 + 前端轮询
  - 自动清理: 延迟清理任务资源 (默认5分钟)
- **日志系统**: 所有日志输出到 `log_config/` 目录
  - 按天自动切分日志文件
  - 自动清理30天前的日志
  - 中文日志输出
  - 同时输出到文件和控制台
- **删除未使用文件**: `task/websocket_manager.py` (前端使用轮询)
- **API 兼容**: 保持原有 API 接口不变
- **任务管理升级**: 简单字典 → TaskManager
  - 企业级任务生命周期管理 (PENDING → QUEUED → RUNNING → COMPLETED/FAILED)
  - 并发控制: Semaphore-based (max_concurrent=5, max_queue_size=10)
  - 进度跟踪: 实时更新 + 前端轮询
  - 自动清理: 延迟清理任务资源 (默认5分钟)
- **删除未使用文件**: `task/websocket_manager.py` (前端使用轮询)
- **API 兼容**: 保持原有 API 接口不变

### v7.1.0 (2026-03) - BM25 检索升级
- **稀疏向量升级**: TF-IDF → BM25
  - 引入词频饱和函数 (k1=1.5)
  - 文档长度归一化 (b=0.75)
  - 检索效果提升 5-15%
- **可配置切换**: 支持 SPARSE_METHOD 环境变量
- **向后兼容**: 保留 TF-IDF 支持
- **新增文件**: `bm25.py`
- **新增依赖**: `rank-bm25>=0.2.2`

### v7.0.0 (2026-03) - 架构简化版
- **Agent 合并优化**: 7个 → 5个
  - `RetrievalAgent` = EmbeddingAgent + HybridRetrievalAgent
  - `GenAgent` = StyleAnalysisAgent + ImageGenAgent
- **性能提升**: 减少状态传递开销
- **向后兼容**: 保留原有Agent，`create_workflow_v2()` 使用新架构
- **新增文件**: `agents/retrieval_agent.py`, `agents/gen_agent.py`

### v6.2.0 (2026-03) - 提示词工程完整版
- **P0**: 提示词配置化 (`prompts.py`)
  - 版本管理系统
  - 负向提示词支持 (145字符)
  - 结构化输出格式
- **P1**: 效果监控与 A/B 测试 (`prompts_v2.py`)
  - 提示词执行指标追踪
  - 少样本示例 (Few-Shot)
  - A/B 测试框架
- **P2**: 用户反馈闭环 (`prompts_v3.py`)
  - 用户反馈收集与分析
  - 优化建议自动生成
  - 多语言支持 (中英双语)
- **新增 API**: `/api/feedback`, `/api/feedback/analytics`, `/api/feedback/suggestions`

### v6.0.0 (2026-03)
- 企业级优化架构（OpenClaw SubAgent 设计）
- 异步任务管理 + 参数校验 + 并发控制
- WebSocket 任务状态查询（前端轮询）
- 完整测试覆盖

### v5.0.0 (2026-03)
- 循环检索状态机
- LLM 质量评估驱动优化
- 缓存自动清理

### v4.0.0 (2026-03)
- LangGraph 多 Agent 工作流
- 并行混合检索架构
- 智能降级机制

## 许可证

本项目仅供学习参考使用。
