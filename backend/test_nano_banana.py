"""
测�� Nano Banana 图像生成
"""
import os
import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import OPENROUTER_API_KEY, IMAGE_GEN_MODEL
from openai import OpenAI
import httpx

API_TIMEOUT = 300

print("=" * 60)
print("  Nano Banana 图像生成测试")
print("=" * 60)

# 检查 API Key
if not OPENROUTER_API_KEY:
    print("\n[错误] OPENROUTER_API_KEY 未设置!")
    sys.exit(1)

print(f"\n当前配置模型: {IMAGE_GEN_MODEL}")
print(f"API Key: {OPENROUTER_API_KEY[:10]}...{OPENROUTER_API_KEY[-4:]}")

# 创建测试图片 (新品平铺图)
print("\n[1/4] 创建测试图片...")
test_img = Image.new("RGB", (800, 1200), color=(200, 100, 100))
print(f"  测试图片尺寸: {test_img.size}")

# 创建客户端
print("\n[2/4] 创建 OpenRouter 客户端...")
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=httpx.Timeout(API_TIMEOUT, connect=60)
)

# 准备生成请求
from utils.core import image_to_uri

style_prompt = "Professional e-commerce fashion photo, clean white background, soft studio lighting, full-body shot"

gen_content = [
    {"type": "image_url", "image_url": {"url": image_to_uri(test_img)}},
    {"type": "text", "text": f"""Generate a professional fashion product photograph.

IMPORTANT: Keep the original red dress design exactly as shown in the reference image. Only change the presentation style.

Style requirements: {style_prompt}

Output only the generated image."""}
]

print("\n[3/4] 调用 Nano Banana 生成图片...")
print(f"  模型: {IMAGE_GEN_MODEL}")
print(f"  提示词: {style_prompt}")

start_time = time.time()

try:
    response = client.chat.completions.create(
        model=IMAGE_GEN_MODEL,
        messages=[{"role": "user", "content": gen_content}],
        extra_body={
            "modalities": ["text", "image"],
            "image_config": {
                "aspect_ratio": "3:4",
                "image_size": "2K"
            },
        },
    )

    elapsed = time.time() - start_time
    print(f"\n  [OK] 响应时间: {elapsed:.2f}s")

    # 解析响应
    if response.choices:
        choice = response.choices[0]
        message = choice.message

        print(f"\n[4/4] 解析响应...")

        # 检查内容
        if hasattr(message, 'content'):
            print(f"  文本内容: {str(message.content)[:100]}...")

        # 检查是否有图片
        if hasattr(message, '__dict__'):
            keys = list(message.message.__dict__.keys()) if hasattr(message, 'message') else list(message.__dict__.keys())
            print(f"  响应字段: {keys}")

        # 尝试获取图片 URL
        image_url = None

        # 方法1: 检查 message.content 中的图片 URL
        if hasattr(message, 'content') and message.content:
            import re
            url_match = re.search(r'https://[^\s\)]+\.(?:png|jpg|jpeg)', str(message.content))
            if url_match:
                image_url = url_match.group()

        # 方法2: 检查 response_format 中的图片
        if not image_url and hasattr(message, 'message'):
            inner = message.message
            if hasattr(inner, 'content'):
                content_list = inner.content if isinstance(inner.content, list) else [inner.content]
                for item in content_list:
                    if isinstance(item, dict) and 'image_url' in item:
                        image_url = item['image_url'].get('url')
                        break

        if image_url:
            print(f"\n  [成功] 图片 URL: {image_url}")

            # 下载并保存图片
            import requests
            img_response = requests.get(image_url)
            output_path = Path("output/test_nano_banana_result.png")
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, 'wb') as f:
                f.write(img_response.content)

            print(f"  [保存] 图片已保存至: {output_path}")
            print(f"  [大小] {len(img_response.content)} bytes")

        else:
            print(f"\n  [警告] 未找到图片 URL")
            print(f"  完整响应: {message}")

    # 打印使用情况
    if hasattr(response, 'usage'):
        usage = response.usage
        print(f"\n[Token 使用]")
        print(f"  输入 tokens: {usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 'N/A'}")
        print(f"  输出 tokens: {usage.completion_tokens if hasattr(usage, 'completion_tokens') else 'N/A'}")
        if hasattr(usage, 'completion_tokens') and hasattr(usage, 'prompt_tokens'):
            # 计算成本
            input_cost = usage.prompt_tokens * 0.30 / 1_000_000
            output_cost = usage.completion_tokens * 2.50 / 1_000_000
            total_cost = input_cost + output_cost
            print(f"  预估成本: ${total_cost:.4f}")

    print("\n" + "=" * 60)
    print("  测试完成!")
    print("=" * 60)

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n[错误] 请求失败 (耗时 {elapsed:.2f}s)")
    print(f"  错误类型: {type(e).__name__}")
    print(f"  错误信息: {str(e)}")
    import traceback
    traceback.print_exc()
