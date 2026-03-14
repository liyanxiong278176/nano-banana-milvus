# 测试模块说明

本目录包含电商 AI 生图流水线的评估测试脚本。

## 目录结构

```
test/
├── __init__.py                  # 模块初始化
├── test_retrieval_metrics.py    # 检索召回率和准确率测试
├── test_cost_analysis.py        # 生图成本对比分析
├── test_image_quality.py        # 生图质量对比测试
├── run_all_tests.py             # 主测试入口
├── results/                     # 测试结果输出目录
└── README.md                    # 本文件
```

## 测试说明

### 1. 检索召回率和准确率测试 (`test_retrieval_metrics.py`)

**测试目的：**
- 评估向量检索系统的召回率和准确率
- 检查新品查询是否能正确检索到相似品类/风格的爆款

**测试方法：**
- 使用 `new_products/` 目录中的新品图片作为查询
- 对于每个新品，执行 Top-3 混合检索（稠密向量 + 稀疏向量）
- 以品类匹配作为相关性判断标准
- 计算以下指标：
  - **品类匹配率**: 检索结果中与查询品类相同的产品比例
  - **NDCG@K**: 归一化折损累计增益
  - **Top-1 准确率**: 第一个结果正确的比例
  - **Top-3 召回率**: 前3个结果中有相关结果的比例

**运行方式：**
```bash
cd backend
python -m test.test_retrieval_metrics
```

### 2. 生图成本对比分析 (`test_cost_analysis.py`)

**测试目的：**
- 对比传统拍摄方式与 AI 方式的成本
- 计算不同规模下的节省金额和比例

**对比项目：**

| 项目 | 传统方式 | AI 方式 |
|------|----------|---------|
| 模特费用 | ¥500-2000 | ¥0 |
| 摄影师 | ¥500-1500 | ¥0 |
| 场地租赁 | ¥200-1000 | ¥0 |
| 造型师 | ¥200-500 | ¥0 |
| 后期修图 | ¥100-300 | ¥0 |
| **API 调用** | - | ¥2-5 |
| **时间周期** | 3-7 天 | 2-5 分钟 |
| **单张成本** | ¥3100 | ¥2-5 |

**运行方式：**
```bash
cd backend
python -m test.test_cost_analysis
```

### 3. 生图质量对比测试 (`test_image_quality.py`)

**测试目的：**
- 量化对比 "本方案"（带检索参考图）与 "基线方案"（仅新品图+prompt）的生成图片质量差异

**对比方案：**

| 方案 | 输入 | 说明 |
|------|------|------|
| 基线方案 | 新品图 + 通用 prompt | 直接生成，无风格参考 |
| 本方案 | 新品图 + 参考爆款图 + LLM 风格分析 | 使用爆款风格指导生成 |

**评分维度（使用 VLM 自动评分）：**
- **服装一致性** (1-5分): 是否正确穿上新品服装
- **人体自然度** (1-5分): 姿势、合身度是否自然
- **场景质量** (1-5分): 背景、场景是否专业
- **光影效果** (1-5分): 光线是否自然、专业
- **商业价值** (1-5分): 是否适合电商使用

**输出结果：**
- 两组生成图片保存到 `results/quality_comparison/`
- JSON 报告包含各维度评分对比和提升百分比

**运行方式：**
```bash
cd backend
python -m test.test_image_quality
```

**注意：** 此测试需要多次调用生成 API，成本较高。默认测试 2 个新品。

### 4. 运行所有测试 (`run_all_tests.py`)

**运行方式：**
```bash
cd backend
python -m test.run_all_tests
```

## 输出结果

所有测试结果保存在 `test/results/` 目录：

- `retrieval_metrics_YYYYMMDD_HHMMSS.json` - 检索指标详细结果
- `cost_analysis_YYYYMMDD_HHMMSS.json` - 成本分析报告
- `summary_report_YYYYMMDD_HHMMSS.json` - 汇总报告

## 前置条件

1. **Milvus 服务运行中**
   ```bash
   # 确保 Milvus 在 localhost:19530 运行
   ```

2. **数据库已初始化**
   ```bash
   cd backend
   python main.py  # 首次运行初始化数据库
   ```

3. **API Key 配置**
   - 确保 `.env` 文件中配置了 `OPENROUTER_API_KEY`
   - 账户有足够余额

4. **新品图片**
   - `backend/new_products/` 目录下有测试图片

## 成本估算 (基于 OpenRouter 价格)

- **嵌入模型** (nvidia/llama-nemotron-embed-vl-1b-v2): 免费
- **LLM 模型** (qwen/qwen3-vl-8b-instruct): ~$0.000055/次分析
- **生图模型** (bytedance-seed/seedream-4.5): ~$0.04/张

单个新品总成本: 约 ¥0.29 (约 $0.04)
