# Nano Banana + Milvus 电商 AI 生图流水线

> 基于向量检索和 AIGC 的电商宣传图自动生成系统

---

## 项目背景

### 痛点问题

中小电商企业在商品宣传图制作中面临以下挑战：

- **成本高**：专业拍摄 + 后期修图，单张成本可达几十至上百元
- **周期长**：从新品上架到宣传图完成，通常需要数天时间
- **质量不稳定**：依赖设计师个人能力，风格难以统一

### 解决方案

本项目通过 **AI 向量检索 + 风格分析 + 图像生成 + 质量评估** 技术实现：

```
新品平铺图 → 自动检索相似爆款 → 提取拍摄风格 → AI 生成宣传图 → AI 质量评分
```

**核心价值**：
- 无需本地 GPU，所有模型通过 API 调用
- 端到端自动化，单张图片生成成本约 $0.06
- 基于历史爆款数据，保证生成风格符合品牌调性
- AI 自动质量评估，确保生成图片可用性

### 第一性原则

**服装必须来自用户上传的图片**：系统生成的宣传图中，模特身上穿的服装必须是用户上传的原款服装。参考爆款图只用于提取**拍摄风格**（场景、光线、构图、氛围），不提供服装款式参考。

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **向量数据库** | Milvus 2.4+ | 混合向量检索（Dense + Sparse + Scalar） |
| **Embedding 模型** | nvidia/llama-nemotron-embed-vl-1b-v2 | 图文向量化（2048维） |
| **LLM 模型** | qwen/qwen3-vl-8b-instruct | 视觉风格分析 + 质量评估 |
| **图像生成** | bytedance-seed/seedream-4.5 | 宣传图生成 |
| **API 聚合** | OpenRouter | 统一模型调用接口 |
| **文本处理** | scikit-learn (TF-IDF) | 稀疏向量生成 |
| **图像处理** | Pillow | 图片加载、格式转换 |
| **后端框架** | FastAPI | RESTful API 服务 |
| **前端框架** | Vue 3 + Vite | Web 界面 |

---

## 架构设计

### 核心设计理念：第一性原则

**服装必须来自用户上传的图片**：系统生成的宣传图中，模特身上穿的服装必须是用户上传的原款服装。参考爆款图只用于提取**拍摄风格**（场景、光线、构图、氛围），绝不提供服装款式参考。

### 数据流向图

```
                    ┌─────────────────┐
                    │   用户上传       │
                    │   新品服装图     │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌───────────────┐         ┌───────────────┐
        │  向量化编码    │         │   元数据提取    │
        │  Dense+Sparse │         │  品类/风格/季节 │
        └───────┬───────┘         └───────┬───────┘
                │                         │
                └────────────┬────────────┘
                             ▼
                    ┌──────────────────┐
                    │   Milvus 混合    │
                    │   向量检索       │
                    │   (Dense+Sparse) │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  返回相似爆款图  │
                    │  (1-6张，去重)   │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │  【关键分离点】          │
                │  ┌─────┴─────┐          │
                │  │           │          │
                ▼  ▼           ▼          ▼
        ┌──────────┐   ┌──────────────┐
        │ 爆款参考图 │   │  LLM 风格分析 │
        │ (仅用于   │──▶│  Qwen3-VL    │
        │ 风格分析)  │   │              │
        └──────────┘   │ 输出：拍摄风格 │
                        │ (文字描述)     │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  图像生成模型  │
                        │  Seedream 4.5 │
                        │  输入：        │
                        │  ├ 用户服装图  │
                        │  └ 风格(文字)  │
                        │  ⚠️ 不传入参考图│
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │   生成宣传图   │
                        │ 服装=用户原图  │
                        │ 风格=爆款风格  │
                        └───────┬───────┘
                                │
                        ┌───────┴───────┐
                        │               │
                        ▼               ▼
                 ┌──────────┐    ┌──────────┐
                 │ 可选：    │    │  输出：   │
                 │ AI质量评估│    │ 保存图片 │
                 └──────────┘    └──────────┘
```

### 模块架构详解

#### 模块1：向量化 (embedding.py)

```python
# 输入：新品图片 + 元数据
# 处理：Dense向量(视觉) + Sparse向量(文本)
# 输出：2048维向量 + TF-IDF稀疏向量

Dense向量 ──┐
            ├──▶ RRF融合 ──▶ Milvus检索
Sparse向量─┘
```

#### 模块2：智能检索 (retrieval.py)

```python
# 输入：Dense向量 + Sparse向量 + 过滤条件
# 处理：混合检索 + 智能去重
# 输出：相似爆款列表(1-6张)

检索策略：
├── Dense Search  ──┐
├── Sparse Search ──┼──▶ RRF Ranker ──▶ 去重过滤
└── Scalar Filter ──┘      │
                          ├── 颜色去重
                          ├── 同款去重
                          └── 销量过滤
```

#### 模块3：拍摄风格分析 (image_gen.py)

```python
# 输入：爆款参考图(1-6张)
# 模型：Qwen3-VL
# 输出：拍摄风格描述(文字)

# ⚠️ 重要：只分析拍摄风格，不描述服装款式！
风格分析维度：
├── 场景/背景设置    # 如：纯白背景、自然户外
├── 光线和色调      # 如：柔和自然光、暖色调
├── 模特姿势和构图   # 如：全身站姿、动态抓拍
└── 整体氛围和美学   # 如：极简主义、商务专业

# ❌ 不提取：
#    × 服装款式
#    × 服装颜色
#    × 服装材质
```

#### 模块4：图像生成 (image_gen.py)

```python
# 输入：用户服装图 + 拍摄风格(文字)
# 模型：Seedream 4.5
# 输出：宣传图(服装=用户原图)

# 【第一性原则】生成模型输入：
gen_content = [
    {"image": 用户上传的服装图},     # ✅ 传入
    # ❌ 不传入参考图！避免服装款式被参考
    {"text": """
        【第一性原则】模特必须穿着第一张图片的服装！
        拍摄风格参考：{style_prompt}
        场景提示：{scene_hint}
    """}
]
```

#### 模块5：质量评估 (image_gen.py)

```python
# 输入：新品图 + 生成图 + 参考爆款图
# 模型：Qwen3-VL
# 输出：多维度评分(1-5分)

评分维度：
├── clothing_accuracy   # 服装准确性（第一性原则核心指标）
├── pose_naturalness    # 姿势自然度
├── scene_quality       # 场景质量
├── lighting_quality    # 布光质量
└── commercial_value    # 商业价值

# 质量判断：
# - clothing_accuracy < 4 → 违反第一性原则，需重新生成
# - average < 3.5 → 建议重新生成
```

### 代码层面的第一性原则体现

**关键代码位置**：`backend/image_gen.py:399-405`

```python
# 【第一性原则】只传用户上传的服装图，不传参考图！
# 参考图的拍摄风格已经通过 LLM 分析提取到 style_prompt 中了
# 如果传参考图给生成模型，模型会同时参考服装款式，导致生成的服装变化
gen_content = [
    {"type": "image_url", "image_url": {"url": image_to_uri(new_product_image)}},
    {"type": "text", "text": gen_prompt}
]
# 注意：循环传入参考图的代码已被删除！
```

### 智能检索特性

**去重与过滤机制**：
- **相关性过滤**：只返回相似度高于阈值的结果
- **颜色去重**：不返回相同颜色的产品
- **同款去重**：不返回同款不同色的产品
- **可变数量**：返回 1-6 张参考图，取决于实际符合条件的数量

**RRF 融合公式**：`score(d) = Σ 1 / (k + rank_i(d))`

---

## 项目结构

```
nano-banana-milvus/
├── backend/              # FastAPI 后端
│   ├── api.py           # API 服务
│   ├── config.py        # 配置文件
│   ├── embedding.py     # 向量嵌入
│   ├── retrieval.py     # 检索模块（智能去重）
│   ├── image_gen.py     # 图像生成 + 质量评估 + 风格分析
│   ├── utils.py         # 工具函数
│   ├── main.py          # CLI 入口
│   ├── images/          # 历史爆款图片
│   ├── new_products/    # 新品图片
│   ├── output/          # 生成结果
│   ├── products.csv     # 商品元数据
│   └── requirements.txt # Python 依赖
│
└── frontend/            # Vue3 前端
    ├── index.html       # 入口文件
    ├── package.json     # 依赖配置
    ├── vite.config.js   # Vite 配置
    └── src/
        ├── main.js
        ├── App.vue
        └── components/
            ├── UploadForm.vue      # 上传表单（含高级选项）
            └── ResultsGallery.vue   # 结果展示（含质量评分）
```

### 核心模块说明

| 文件 | 作用 | 核心类/函数 |
|------|------|------------|
| `api.py` | FastAPI 服务 | `/api/upload`, `/api/tasks/{task_id}` |
| `retrieval.py` | 智能检索 | `retrieve_similar_bestsellers()` (去重+过滤) |
| `embedding.py` | 向量生成 | `EmbeddingGenerator` 类 |
| `image_gen.py` | 图像生成+评估+风格分析 | `ImageGenerator`, `ImageQualityJudge` |
| `config.py` | 配置管理 | API Key、模型 ID、路径常量 |
| `utils.py` | 工具函数 | `get_image_embeddings()`, `image_to_uri()` |

---

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- Milvus 2.4+ (Docker 或本地安装)
- OpenRouter API Key

### 1. 启动 Milvus

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -v $(pwd)/backend/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

### 2. 启动后端

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python api.py
```

后端运行在 http://localhost:8000

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端运行在 http://localhost:3000

---

## 使用指南

### Web 界面使用

1. **上传新品图片**：拖拽或点击上传
2. **选择商品属性**：品类、风格、季节
3. **场景提示（可选）**：描述想要的场景
4. **高级选项**：
   - 启用 AI 质量评估
   - 选择裁判模型（可选）
   - 启用后将生成图片并进行多维度评分

### API 调用

**上传并生成**：
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@new_product.jpg" \
  -F "category=midi_dress" \
  -F "style=elegant" \
  -F "season=summer" \
  -F "enable_quality_check=true" \
  -F "judge_model=qwen/qwen3-vl-8b-instruct"
```

**查询任务状态**：
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

### 高级选项参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_quality_check` | boolean | false | 是否启用 AI 质量评估 |
| `judge_model` | string | "" | 裁判模型（空则使用默认模型） |
| `scene_hint` | string | "" | 场景提示描述 |

---

## AI 质量评估

启用质量评估后，AI 裁判会对生成的图片进行多维度评分：

| 评分维度 | 说明 | 满分 |
|---------|------|------|
| `clothing_accuracy` | 服装准确性：与原图匹配度 | 5 |
| `pose_naturalness` | 姿势自然度：模特姿势和合身度 | 5 |
| `scene_quality` | 场景质量：背景/场景专业性 | 5 |
| `lighting_quality` | 布光质量：光线质量 | 5 |
| `commercial_value` | 商业价值：是否适合电商使用 | 5 |
| `average` | 平均分 | 5 |

**质量判断**：
- 平均分 ≥ 3.5：质量合格
- 平均分 < 3.5：建议重新生成
- 服装准确性 < 4：建议重新生成（违反第一性原则）

---

## 配置说明

### 模型配置 (`config.py`)

```python
# API 配置
OPENROUTER_API_KEY = "your-key-here"

# 模型选择
LLM_MODEL = "qwen/qwen3-vl-8b-instruct"      # LLM（风格分析+质量评估）
IMAGE_GEN_MODEL = "bytedance-seed/seedream-4.5"  # 图像生成
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"  # 向量嵌入

# 图像生成配置
DEFAULT_ASPECT_RATIO = "3:4"  # 宽高比
DEFAULT_IMAGE_SIZE = "2K"     # 分辨率

# 检索配置
MIN_SALES_COUNT = 1500        # 最低销量阈值
MIN_SIMILARITY = 0.5          # 相似度阈值（0-1，越小越相关）
MAX_RESULTS = 6               # 最多返回参考图数量
```

---

## 常见问题

**Q: Milvus 连接失败？**

A: 检查 Milvus 是否启动：`docker ps | grep milvus`

**Q: API 调用报错？**

A: 检查 API Key 是否正确，访问 https://openrouter.ai/settings/credits 查看余额

**Q: 检索结果为空？**

A: 检查数据库是否已初始化，确认品类和销量设置合理

**Q: 生成的参考图很少？**

A: 智能检索会自动去重，返回数量取决于符合条件的不同产品数量

**Q: 质量评分偏低？**

A: 检查服装准确性得分，如果低于4分说明服装与原图不匹配

**Q: 生成的服装不是原图款式？**

A: 这违反了第一性原则，请检查 `image_gen.py` 中是否正确配置了不传入参考图

---

## 性能指标

### 检索性能

基于 120 个商品的测试集：

| 检索方法 | Precision@5 | 说明 |
|---------|-------------|------|
| Dense 向量 | 76% | 仅使用视觉特征向量 |
| Sparse 向量 | 40% | 仅使用文本特征向量 |
| **混合检索 (RRF)** | **40%** | Dense + Sparse 融合 + 智能去重 |

**详细指标** (Dense 向量):
- **MRR**: 0.50 (第一个相关商品平均排名)
- **MAP**: 1.84 (平均准确率均值)
- **NDCG@5**: 0.96 (归一化折损累计增益)

### 成本对比

| 场景 | 传统拍摄 | AI生图 | 节省 |
|------|---------|--------|------|
| 10商品 | RMB 4,250 | RMB 17 | **99.6%** |
| 200商品/年 | RMB 27,050 | RMB 349 | **98.7%** |
| 1000商品/年 | RMB 124,800 | RMB 1,745 | **98.6%** |

单张成本：传统 RMB 30-100 vs AI RMB 0.44

### 时间成本

| 项目 | 传统方式 | AI方式 |
|------|---------|--------|
| 准备周期 | 3-7天 | <5分钟 |
| 单张生成 | 30-60分钟 | 30-60秒 |

---

## 许可证

本项目仅供学习参考使用。
