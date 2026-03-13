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
    "your-openrouter-api-key-here"
)

# ==================== 模型配�� ====================
# 所有模型通过 OpenRouter API 调用，无需本地 GPU
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"  # 免费，支持图像+文本 → 2048维
EMBED_DIM = 2048
LLM_MODEL = "qwen/qwen3-vl-8b-instruct"  # 风格分析（支持视觉）
IMAGE_GEN_MODEL = "bytedance-seed/seedream-4.5"  # ByteDance Seedream (在中国可用)

# ==================== Milvus 配置 ====================
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "fashion_products"
TOP_K_RETRIEVAL = 3  # 检索返回的相似爆款数量

# ==================== 图像生成配置 ====================
DEFAULT_ASPECT_RATIO = "3:4"  # 适合人像的宽高比
DEFAULT_IMAGE_SIZE = "2K"     # 分辨率: 512px, 1K, 2K, 4K
MAX_IMAGE_SIZE = 1024         # API 传输时图片最大尺寸

# ==================== 批处理配置 ====================
EMBED_BATCH_SIZE = 5          # 批量编码时的批次大小
RATE_LIMIT_DELAY = 0.5        # API 调用间隔(秒)

# ==================== 筛选条件 ====================
MIN_SALES_COUNT = 1500        # 只检索销量超过此值的爆款

print(f"配置加载完成 | 项目路径: {PROJECT_ROOT}")
