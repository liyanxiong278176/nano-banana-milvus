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
# 从环境变量获取 API Key
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# 验证 API Key 是否存在
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY 环境变量未设置。\n"
        "请在 .env 文件中配置 OPENROUTER_API_KEY，或导出环境变量。\n"
        "获取 API Key: https://openrouter.ai/keys"
    )

# ==================== 模型配置 ====================
# 所有模型通过 OpenRouter API 调用，无需本地 GPU
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"  # 免费，支持图像+文本 → 2048维
EMBED_DIM = 2048
LLM_MODEL = "qwen/qwen3-vl-8b-instruct"  # 风格分析（免费，支持视觉+视频）
IMAGE_GEN_MODEL = "black-forest-labs/flux.2-klein-4b"  # FLUX.2 Klein
LIGHT_LLM_MODEL = "qwen/qwen3-vl-8b-instruct"  # 免费模型，用于检索质量评估

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
MIN_SALES_COUNT = 500         # 只检索销量超过此值的爆款（降低阈值提高召回率）

# ==================== 检索配置常量 ====================
# 检索候选数量配置
RETRIEVAL_CANDIDATE_MULTIPLIER = 4  # 候选数量 = top_k * 候选倍数
MIN_CANDIDATE_COUNT = 15            # 最小候选数量
MAX_SIMILARITY_THRESHOLD = 0.5     # RRF 距离��值（越小越相关，0-1之间）

# 参考图片数量限制
MAX_REFERENCE_IMAGES_FOR_SCORING = 2   # 质量评估时最多使用的参考图数量
MAX_REFERENCE_IMAGES_FOR_ANALYSIS = 3  # 风格分析时最多评估的参考图数量

# 循环检索配置
MAX_RETRIEVAL_ROUNDS = 3          # 最大检索轮数
QUALITY_SCORE_THRESHOLD = 7.0     # 质量评估阈值（0-10分）
RETRIEVAL_QUALITY_THRESHOLD = 6.0 # 检索质量评估阈值（0-10分）

# 查询重写配置
QUERY_REWRITE_SALES_HIGH = 1500  # 第1轮重写的高销量阈值
QUERY_REWRITE_SALES_MID = 1000   # 第1轮重写的中销量阈值
QUERY_REWRITE_SALES_LOW = 500    # 第2轮重写的低销量阈值

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
