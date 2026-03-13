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

本项目通过 **AI 向量检索 + 图像生成** 技术，实现：

```
新品平铺图 → 自动检索相似爆款 → 分析爆款风格 → AI 生成宣传图
```

**核心价值**：
- 无需本地 GPU，所有模型通过 API 调用
- 端到端自动化，单张图片生成成本约 $0.06
- 基于历史爆款数据，保证生成风格符合品牌调性

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **向量数据库** | Milvus 2.4+ | 混合向量检索（Dense + Sparse + Scalar） |
| **Embedding 模型** | nvidia/llama-nemotron-embed-vl-1b-v2 | 图文向量化（2048维） |
| **LLM 模型** | qwen/qwen3-vl-8b-instruct | 视觉风格分析 |
| **图像生成** | bytedance-seed/seedream-4.5 | 宣传图生成 |
| **API 聚合** | OpenRouter | 统一模型调用接口 |
| **文本处理** | scikit-learn (TF-IDF) | 稀疏向量生成 |
| **图像处理** | Pillow | 图片加载、格式转换 |

---

## 架构设计

### 系统架构图

```
┌───────────────────────────────────────────────────────────────────────┐
│                          输入：新品平铺图                               │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  模块1：向量化 (embedding.py + utils.py)                              │
│  ┌─────────────────────┐              ┌─────────────────────┐         │
│  │   Dense 向量        │              │   Sparse 向量       │         │
│  │   (视觉特征 2048维) │              │   (文本特征 TF-IDF) │         │
│  │   API: NVIDIA Embed │              │   本地计算          │         │
│  └─────────────────────┘              └─────────────────────┘         │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  模块2：混合检索 (milvus_db.py + retrieval.py)                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│  │  Dense   │   │  Sparse  │   │  Scalar  │   │   RRF    │           │
│  │  Search  │   │  Search  │   │  Filter  │   │  Ranker   │           │
│  │          │   │          │   │ 品类+销量│   │  融合结果 │           │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘           │
│       │              │              │              │                 │
│       └──────────────┼──────────────┘              │                 │
│                      ▼                             ▼                 │
│                 Milvus Collection              Top-K 爆款             │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  模块3：风格分析 (image_gen.py)                                       │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │   输入：Top-K 爆款宣传图                                      │     │
│  │   模型：Qwen3-VL (多模态 LLM)                                │     │
│  │   输出：风格描述 Prompt (场景/灯光/姿势/氛围)                 │     │
│  └─────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  模块4：图像生成 (image_gen.py)                                       │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │   输入：新品图 + 爆款参考图 + 风格 Prompt                     │     │
│  │   模型：Seedream 4.5                                         │     │
│  │   输出：新品宣传图 × N 张                                     │     │
│  └─────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                       输出：output/{product_id}/                       │
│   ├── {id}_original.png     (新品原图)                               │
│   ├── {id}_reference_*.png   (参考爆款图)                             │
│   ├── {id}_generated_*.png   (AI 生成图)                              │
│   └── {id}_style_prompt.txt  (风格描述)                               │
└───────────────────────────────────────────────────────────────────────┘
```

### 混合检索原理

```
                    查询向量
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ Dense   │     │ Sparse  │     │ Scalar  │
   │ Search  │     │ Search  │     │ Filter  │
   └────┬────┘     └────┬────┘     └────┬────┘
        │               │               │
    视觉相似度      语义匹配       品类+销量过滤
        │               │               │
        └───────────────┼───────────────┘
                        ▼
                   ┌─────────┐
                   │   RRF   │
                   │ Ranker  │
                   └────┬────┘
                        │
                    Top-K 结果
```

**RRF 融合公式**：`score(d) = Σ 1 / (k + rank_i(d))`

---

## 项目结构

```
nano-banana-milvus/
├── config.py              # 配置文件（API Key、模型选择、路径）
├── utils.py               # 工具函数（图片处理、API 调用）
├── embedding.py           # 向量嵌入生成（Dense + Sparse）
├── milvus_db.py           # Milvus 数据库操作
├── retrieval.py           # 爆款检索模块
├── image_gen.py           # 图像生成模块（风格分析 + 生图）
├── main.py                # 主程序入口
├── batch_process.py       # 批量处理脚本
├── init_db.py             # 数据库初始化脚本
├── test_pipeline.py       # 测试脚本
│
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量模板
├── .gitignore             # Git 忽略规则
├── README.md              # 本文档
│
├── images/                # 历史爆款商品图片
├── products.csv           # 历史商品元数据
├── new_products/          # 新品图片
├── new_products.csv       # 新品元数据
├── output/                # 生成结果输出目录
└── milvus_data/           # Milvus 数据存储
```

### 核心文件说明

| 文件 | 作用 | 核心类/函数 |
|------|------|------------|
| `config.py` | 集中配置管理 | API Key、模型 ID、路径常量 |
| `utils.py` | 通用工具函数 | `get_image_embeddings()`, `image_to_uri()` |
| `embedding.py` | 向量生成 | `EmbeddingGenerator` 类 |
| `milvus_db.py` | 数据库操作 | `FashionMilvusDB` 类，`hybrid_search()` |
| `retrieval.py` | 检索业务 | `BestsellerRetriever` 类 |
| `image_gen.py` | 图像生成 | `ImageGenerator` 类 |
| `main.py` | 流水线编排 | `FashionImagePipeline` 类 |

---

## 快速开始

### 环境要求

- Python 3.9+
- Milvus 2.4+ (Docker 或本地安装)
- OpenRouter API Key

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 Milvus

```bash
# 使用 Docker 启动 Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

### 3. 配置 API Key

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 OpenRouter API Key
# OPENROUTER_API_KEY=your-key-here
```

或直接在 `config.py` 中设置：

```python
OPENROUTER_API_KEY = "your-openrouter-api-key-here"
```

### 4. 准备数据

**商品数据格式** (`products.csv`)：

```csv
product_id,image_path,category,color,style,season,sales_count,description,price
SKU001,SKU001.jpg,midi_dress,pink,drawstring,summer,2800,Pink midi dress with drawstring waist,49.99
```

**新品数据格式** (`new_products.csv`)：

```csv
new_id,image_path,category,style,season,prompt_hint
NEW001,NEW001.jpg,midi_dress,knitted,autumn,Blue knitted cardigan over gray skirt, cozy cafe
```

将对应图片放到 `images/` 和 `new_products/` 目录。

### 5. 运行项目

```bash
# 方式一：使用主程序

# 1. 构建向量数据库
python main.py --build-db

# 2. 处理单个新品
python main.py --process NEW001

# 3. 批量处理所有新品
python main.py --batch

# 方式二：使用批量处理脚本
python batch_process.py --all

# 方式三：测试环境配置
python test_pipeline.py --all
```

---

## 使用示例

### 处理单个新品

```bash
python main.py --process NEW001
```

**输出**：
```
============================================================
处理新品: NEW001
============================================================
品类: midi_dress
风格: knitted
季节: autumn
场景提示: Blue knitted cardigan over gray skirt, cozy cafe

执行混合检索...
  品类: midi_dress
  最低销量: 1500
找到 3 个相似爆款:
  SKU001 | midi_dress | pink | drawstring
    销量: 2800 | 相似度: 0.8234
  ...

使用 Qwen3.5 分析爆款风格...
风格分析结果:
A chic urban street scene with soft natural lighting...

使用 Seedream 4.5 生成宣传图...
  宽高比: 3:4
  分辨率: 2K

所有结果已保存到: output/NEW001/
```

---

## 常见问题

**Q: Milvus 连接失败？**

A: 检查 Milvus 是否启动：`docker ps | grep milvus`

**Q: API 调用报错？**

A: 检查 API Key 是否正确，访问 https://openrouter.ai/settings/credits 查看余额

**Q: 图片生成质量不好？**

A: 尝试调整 `config.py` 中的 `IMAGE_GEN_MODEL` 或优化 prompt_hint

---

## 许可证

本项目仅供学习参考使用。
