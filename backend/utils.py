"""
工具函数模块
"""
import io
import base64
import time
import json
import hashlib
import requests as req
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Any, Optional
from tqdm import tqdm

from config import (
    OPENROUTER_API_KEY, EMBED_MODEL, MAX_IMAGE_SIZE, RATE_LIMIT_DELAY,
    CACHE_DIR, ENABLE_CACHE
)


def image_to_uri(img: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> str:
    """
    将 PIL 图片转换为 base64 data URI，用于 API 传输

    Args:
        img: PIL Image 对象
        max_size: 最大边长，超过会等比缩放

    Returns:
        base64 data URI 字符串
    """
    img = img.copy()
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def get_image_embeddings(images: List[Image.Image], batch_size: int = 5) -> np.ndarray:
    """
    通过 OpenRouter Embedding API 批量编码图片

    Args:
        images: PIL Image 列表
        batch_size: 批次大小

    Returns:
        numpy 数组，形状为 (n_images, embed_dim)
    """
    all_embeddings = []

    for i in tqdm(range(0, len(images), batch_size), desc="编码图片向量"):
        batch = images[i:i + batch_size]

        # 构建 API 请求
        inputs = [
            {
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": image_to_uri(img, max_size=512)}
                }]
            }
            for img in batch
        ]

        resp = req.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": EMBED_MODEL, "input": inputs},
            timeout=300,  # 增加到 5 分钟
        )

        data = resp.json()
        if "data" not in data:
            print(f"API 错误: {data}")
            continue

        # 按 index 排序确保顺序一致
        for item in sorted(data["data"], key=lambda x: x["index"]):
            all_embeddings.append(item["embedding"])

        time.sleep(RATE_LIMIT_DELAY)  # 礼貌地避免速率限制

    return np.array(all_embeddings, dtype=np.float32)


def get_text_embedding(text: str) -> np.ndarray:
    """
    通过 OpenRouter Embedding API 编码文本

    Args:
        text: 输入文本

    Returns:
        numpy 向量数组
    """
    resp = req.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": EMBED_MODEL, "input": text},
        timeout=180,  # 增加到 3 分钟
    )
    return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)


def sparse_to_dict(sparse_row) -> Dict[int, float]:
    """
    将 scipy 稀疏矩阵行转换为 Milvus 稀疏向量格式 {index: value}

    Args:
        sparse_row: scipy sparse 矩阵的一行

    Returns:
        字典格式的稀疏向量
    """
    coo = sparse_row.tocoo()
    return {int(i): float(v) for i, v in zip(coo.col, coo.data)}


def extract_images(response) -> List[Image.Image]:
    """
    从 OpenRouter API 响应中提取生成的图片

    Args:
        response: OpenAI SDK 的响应对象

    Returns:
        PIL Image 列表
    """
    images = []
    raw = response.model_dump()
    msg = raw["choices"][0]["message"]

    # 方法1: 从 images 字段提取 (OpenRouter 扩展)
    if "images" in msg and msg["images"]:
        for img_data in msg["images"]:
            url = img_data["image_url"]["url"]
            b64 = url.split(",", 1)[1]
            images.append(Image.open(io.BytesIO(base64.b64decode(b64))))

    # 方法2: 从 content parts 中的内联 base64 提取
    if not images and isinstance(msg.get("content"), list):
        for part in msg["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:image"):
                    b64 = url.split(",", 1)[1]
                    images.append(Image.open(io.BytesIO(base64.b64decode(b64))))

    return images


def load_image(image_path: str, mode: str = "RGB") -> Image.Image:
    """
    加载图片文件

    Args:
        image_path: 图片路径
        mode: 图片模式，默认 RGB

    Returns:
        PIL Image 对象
    """
    return Image.open(image_path).convert(mode)


def save_image(img: Image.Image, output_path: str):
    """
    保存图片到指定路径

    Args:
        img: PIL Image 对象
        output_path: 输出路径
    """
    img.save(output_path)
    print(f"已保存: {output_path}")


def display_comparison(original: Image.Image, reference: Image.Image, generated: Image.Image):
    """
    拼接展示三张图片对比图

    Args:
        original: 原始新品平铺图
        reference: 参考爆款图
        generated: 生成宣传图

    Returns:
        拼接后的 PIL Image
    """
    # 统一高度
    target_height = 400
    images = [original, reference, generated]
    resized = []

    for img in images:
        w, h = img.size
        new_w = int(w * target_height / h)
        resized.append(img.resize((new_w, target_height), Image.LANCZOS))

    # 横向拼接
    total_width = sum(img.size[0] for img in resized)
    result = Image.new("RGB", (total_width + 40, target_height + 60), (255, 255, 255))

    x_offset = 20
    labels = ["New Product", "Bestseller Reference", "Generated"]

    for i, (img, label) in enumerate(zip(resized, labels)):
        result.paste(img, (x_offset, 50))
        # 这里可以添加标签文字
        x_offset += img.size[0] + 10

    return result


# ==================== 缓存工具函数（新增） ====================

def get_cache_key(prefix: str, content: Union[str, bytes]) -> str:
    """
    生成缓存键

    使用 MD5 哈希 content 生成唯一键，避免文件名过长

    Args:
        prefix: 缓存前缀（如 "style_analysis", "embedding"）
        content: 用于生成键的内容（图片的 base64 或文本）

    Returns:
        唯一的缓存文件名（格式: prefix_md5hash.json）

    Examples:
        >>> get_cache_key("style", "some text content")
        'style_a94a8fe5ccb19ba61c4c0873d391e987982fbbd3.json'
    """
    if isinstance(content, str):
        content_bytes = content.encode('utf-8')
    else:
        content_bytes = content

    hash_obj = hashlib.md5(content_bytes)
    hash_str = hash_obj.hexdigest()
    return f"{prefix}_{hash_str}.json"


def save_to_cache(cache_key: str, data: Union[Dict, str, list]) -> bool:
    """
    保存数据到缓存

    Args:
        cache_key: 缓存键（由 get_cache_key 生成）
        data: 要缓存的数据（字典、字符串或列表）

    Returns:
        是否成功保存（异常时返回 False，不影响主流程）
    """
    if not ENABLE_CACHE:
        return False

    try:
        cache_path = CACHE_DIR / cache_key
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        # 缓存写入失败时，静默处理，不影响主流程
        print(f"  警告: 缓存写入失败 ({cache_key}): {e}")
        return False


def load_from_cache(cache_key: str) -> Optional[Dict]:
    """
    从缓存加载数据

    Args:
        cache_key: 缓存键（由 get_cache_key 生成）

    Returns:
        缓存的数据（未找到或读取失败返回 None）
    """
    if not ENABLE_CACHE:
        return None

    try:
        cache_path = CACHE_DIR / cache_key
        if not cache_path.exists():
            return None

        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        # 缓存读取失败时，静默处理，继续正常流程
        print(f"  警告: 缓存读取失败 ({cache_key}): {e}")
        return None


def clear_cache(prefix: str = None) -> int:
    """
    清理缓存文件

    Args:
        prefix: 可选，只清理指定前缀的缓存；为 None 则清理全部

    Returns:
        清理的文件数量

    Note:
        建议定期运行此函数清理缓存，避免缓存文件过多占用磁盘空间
    """
    count = 0
    try:
        for cache_file in CACHE_DIR.glob("*.json"):
            if prefix is None or cache_file.name.startswith(f"{prefix}_"):
                cache_file.unlink()
                count += 1
        print(f"已清理 {count} 个缓存文件")
    except Exception as e:
        print(f"清理缓存失败: {e}")
    return count


if __name__ == "__main__":
    print("工具函数模块测试")
    print(f"image_to_uri 函数: 将 PIL 图片转为 base64 URI")
    print(f"get_image_embeddings 函数: 批量编码图片为向量")
    print(f"get_text_embedding 函数: 编码文本为向量")
    print(f"sparse_to_dict 函数: 转换稀疏向量格式")
    print(f"extract_images 函数: 从 API 响应提取图片")
    # 新增缓存函数说明
    print(f"get_cache_key 函数: 生成缓存键")
    print(f"save_to_cache 函数: 保存数据到缓存")
    print(f"load_from_cache 函数: 从缓存加载数据")
    print(f"clear_cache 函数: 清理缓存文件")
