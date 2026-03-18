"""
配置文件 - Nano Banana + Milvus 电商生图流水线
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ==================== 项目路径 ====================
PROJECT_ROOT = Path(__file__).parent
IMAGE_DIR = PROJECT_ROOT / "images"
NEW_PRODUCT_DIR = PROJECT_ROOT / "new_products"
OUTPUT_DIR = PROJECT_ROOT / "output"
PRODUCT_CSV = PROJECT_ROOT / "products.csv"
NEW_PRODUCT_CSV = PROJECT_ROOT / "new_products.csv"

# 确保目录存在
for dir_path in [IMAGE_DIR, NEW_PRODUCT_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

# ==================== API 配置 ====================
# 从环境变量获取 API Key，如果没有设置则使用占位符
OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
)

# ==================== 模型配置 ====================
# 所有模型通过 OpenRouter API 调用，无需本地 GPU
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"  # 免费，支持图像+文本 → 2048维
EMBED_DIM = 2048
LLM_MODEL = "qwen/qwen3-vl-8b-instruct"  # 风格分析（免费，支持视觉+视频）
IMAGE_GEN_MODEL = "black-forest-labs/flux.2-klein-4b"  # FLUX.2 Klein (价格更低: 首$0.014/MP, 后续$0.001/MP)
LIGHT_LLM_MODEL = "qwen/qwen3-vl-8b-instruct"  # 免费模型，用于检索质量评估

# ==================== Milvus 配置 ====================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "fashion_products"
TOP_K_RETRIEVAL = 3  # 检索返回的相似爆款数量

# ==================== Neo4j 知识图谱配置 ====================
# 【新增】Neo4j 图数据库配置，用于结构化推理和知识图谱检索
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")  # 使用默认数据库（主流用法）
NEO4J_MAX_CONNECTION_POOL_SIZE = 50  # 连接池最大大小
NEO4J_CONNECTION_ACQUISITION_TIMEOUT = 60  # 连接获取超时（秒）
NEO4J_CONNECTION_TIMEOUT = 30  # 连接超时（秒）

# ==================== 图像生成配置 ====================
DEFAULT_ASPECT_RATIO = "3:4"  # 适合人像的宽高比
DEFAULT_IMAGE_SIZE = "2K"     # 分辨率: 512px, 1K, 2K, 4K
MAX_IMAGE_SIZE = 1024         # API 传输时图片最大尺寸

# ==================== 批处理配置 ====================
EMBED_BATCH_SIZE = 5          # 批量编码时的批次大小
RATE_LIMIT_DELAY = 0.5        # API 调用间隔(秒)

# ==================== 筛选条件 ====================
MIN_SALES_COUNT = 1500        # 只检索销量超过此值的爆款

# ==================== 模型分级配置 ====================
# 【新增】根据不同场景选择不同模型，平衡成本和质量
MODEL_TIERS = {
    # 高质量模式：用于最终图片生成的风格分析
    "high_quality": {
        "style_analysis": LLM_MODEL,      # 使用主模型进行风格分析
        "quality_judge": LLM_MODEL,       # 使用主模型进行质量评估
    },
    # 标准模式：默认配置
    "standard": {
        "style_analysis": LLM_MODEL,
        "quality_judge": LIGHT_LLM_MODEL,
    },
    # 经济模式：降低成本
    "economy": {
        "style_analysis": LIGHT_LLM_MODEL,
        "quality_judge": LIGHT_LLM_MODEL,
    }
}

# 当前使用的模型层级（可通过环境变量 OVERRUN_MODEL_TIER 覆盖）
CURRENT_MODEL_TIER = os.environ.get("OVERRUN_MODEL_TIER", "standard")

# ==================== 缓存配置 ====================
# 【新增】简单本地缓存，减少重复的 LLM 调用
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "true").lower() == "true"  # 默认启用
CACHE_MAX_SIZE_MB = 500  # 缓存目录最大大小（MB），超过后建议手动清理
# 注意：缓存文件会持续累积，建议定期清理 CACHE_DIR 中的旧文件

print(f"配置加载完成 | 项目路径: {PROJECT_ROOT}")
print(f"模型层级: {CURRENT_MODEL_TIER} | 缓存: {'启用' if ENABLE_CACHE else '禁用'}")
