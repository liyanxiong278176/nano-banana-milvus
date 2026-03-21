# 测试模块说明

本目录包含电商 AI 生图流水线的评估测试脚本和优化模块测试。

---

## 目录结构

```
test/
├── __init__.py                      # 模块初始化
│
├── 【优化模块测试】
├── test_validator_simple.py         # 参数校验 + 并发控制测试
├── test_task_manager.py             # 任务管理器测试
├── test_full_integration.py         # 完整集成测试
├── test_optimization_integration.py # 优化模块展示测试
│
├── 【功能评估测试】
├── test_retrieval_metrics.py        # 检索召回率和准确率测试
├── test_cost_analysis.py            # 生图成本对比分析
├── test_image_quality.py            # 生图质量对比测试
├── test_cycle_retrieval.py          # 循环检索测试
├── test_cycle_simple.py             # 简单循环检索测试
├── test_cycle_low_score.py          # 低分循环检索测试
├── test_image_quality_improved.py   # 改进版质量测试
│
├── run_all_tests.py                 # 主测试入口
├── results/                         # 测试结果输出目录
└── README.md                        # 本文件
```

---

## 快速开始

### 运行优化模块测试

```bash
cd backend

# 测试参数校验和并发控制
python test/test_validator_simple.py

# 测试任务管理器（生命周期、状态转换）
python test/test_task_manager.py

# 测试完整集成
python test/test_full_integration.py

# 展示优化模块功能
python test/test_optimization_integration.py
```

### 运行功能评估测试

```bash
cd backend

# 检索指标测试
python test/test_retrieval_metrics.py

# 成本分析
python test/test_cost_analysis.py

# 图片质量对比
python test/test_image_quality.py
```

---

## 优化模块测试说明

### 1. 参数校验测试 (`test_validator_simple.py`)

**测试内容**：
- 有效参数校验
- 无效品类拒绝
- 无效风格拒绝
- 文件大小限制
- 并发执行

**运行**：
```bash
python test/test_validator_simple.py
```

### 2. 任务管理器测试 (`test_task_manager.py`)

**测试内容**：
- 任务注册
- 状态转换（pending → queued → running → completed）
- 进度更新
- 失败处理
- 任务取消
- 统计信息
- 任务列表
- 进度回调

**运行**：
```bash
python test/test_task_manager.py
```

**预期输出**：
```
总计: 8/8 测试通过
  [OK] 任务注册
  [OK] 任务状态转换
  [OK] 进度更新
  [OK] 失败处理
  [OK] 任务取消
  [OK] 统计信息
  [OK] 任务列表
  [OK] 进度回调
```

### 3. 完整集成测试 (`test_full_integration.py`)

**测试内容**：
- API 模块集成
- 任务注册流程
- 工作流执行模拟
- WebSocket 管理器
- 统计信息

**运行**：
```bash
python test/test_full_integration.py
```

### 4. 优化模块展示 (`test_optimization_integration.py`)

**测试内容**：
- Validator 参数校验演示
- Limiter 并发控制演示
- TaskManager 任务管理演示
- API 端点集成说明

**运行**：
```bash
python test/test_optimization_integration.py
```

---

## 功能评估测试说明

### 1. 检索召回率和准确率测试 (`test_retrieval_metrics.py`)

**测试目的：**
- 评估向量检索系统的召回率和准确率
- 检查新品查询是否能正确检索到相似品类/风格的爆款

**测试方法：**
- 使用 `new_products/` 目录中的新品图片作为查询
- 执行 Top-3 混合检索（稠密向量 + 稀疏向量）
- 计算品类匹配率、NDCG@K、Top-1 准确率

**运行**：
```bash
python test/test_retrieval_metrics.py
```

### 2. 生图成本对比分析 (`test_cost_analysis.py`)

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

**运行**：
```bash
python test/test_cost_analysis.py
```

### 3. 生图质量对比测试 (`test_image_quality.py`)

**对比方案：**

| 方案 | 输入 | 说明 |
|------|------|------|
| 基线方案 | 新品图 + 通用 prompt | 直接生成，无风格参考 |
| 本方案 | 新品图 + 参考爆款图 + LLM 风格分析 | 使用爆款风格指导生成 |

**评分维度：**
- 服装一致性 (1-5分)
- 人体自然度 (1-5分)
- 场景质量 (1-5分)
- 光影效果 (1-5分)
- 商业价值 (1-5分)

**运行**：
```bash
python test/test_image_quality.py
```

---

## 输出结果

所有测试结果保存在 `test/results/` 目录：

- `retrieval_metrics_YYYYMMDD_HHMMSS.json` - 检索指标详细结果
- `cost_analysis_YYYYMMDD_HHMMSS.json` - 成本分析报告
- `summary_report_YYYYMMDD_HHMMSS.json` - 汇总报告

---

## 前置条件

1. **Milvus 服务运行中**
   ```bash
   docker ps | grep milvus
   ```

2. **数据库已初始化**
   ```bash
   cd backend
   python -c "from main import FashionImagePipeline; FashionImagePipeline()._init_database()"
   ```

3. **API Key 配置**
   - `.env` 文件中配置了 `OPENROUTER_API_KEY`
   - 账户有足够余额

4. **新品图片**
   - `backend/new_products/` 目录下有测试图片

---

## 成本估算 (基于 OpenRouter 价格)

- **嵌入模型** (nvidia/llama-nemotron-embed-vl-1b-v2): 免费
- **LLM 模型** (qwen/qwen3-vl-8b-instruct): ~$0.000055/次分析
- **生图模型** (black-forest-labs/flux.2-klein-4b): ~$0.014/张

单个新品总成本: 约 ¥0.10 (约 $0.014)
