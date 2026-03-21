"""
测试图像生成 API - 多模型测试
"""
import os
import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from config import OPENROUTER_API_KEY
from openai import OpenAI
import httpx

API_TIMEOUT = 300
TEST_MODELS = [
    "x-ai/grok-4.20-multi-agent-beta",
    "google/gemini-3.1-flash-image-preview",
    "openai/gpt-5-image-mini"
]

print("="*60)
print("  图像生成 API 测试")
print("="*60)

# 检查 API Key
if not OPENROUTER_API_KEY:
    print("\n[错误] OPENROUTER_API_KEY 未设置!")
    sys.exit(1)

print(f"\nAPI Key: {OPENROUTER_API_KEY[:10]}...{OPENROUTER_API_KEY[-4:]}")

# 创建测试图片
print("\n创建测试图片...")
test_img = Image.new("RGB", (512, 768), color=(100, 150, 200))

# 创建客户端
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=httpx.Timeout(API_TIMEOUT, connect=60)
)

# 准备请求
from utils.core import image_to_uri

gen_prompt = "Generate a professional product photo of a fashion model wearing a red dress, clean white background, studio lighting."

gen_content = [
    {"type": "image_url", "image_url": {"url": image_to_uri(test_img)}},
    {"type": "text", "text": gen_prompt}
]

print(f"\n测试 {len(TEST_MODELS)} 个模型...")
print("="*60)

# 测试每个模型
success = False
for i, test_model in enumerate(TEST_MODELS, 1):
    print(f"\n[{i}/{len(TEST_MODELS)}] {test_model}")
    print("-" * 50)

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=test_model,
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
        print(f"  [成功] 响应时间: {elapsed:.2f}s")

        if response.choices:
            choice = response.choices[0]
            content = choice.message.content

            print(f"  Content: {str(content)[:200]}")

            # 检查所有字段
            if hasattr(choice.message, '__dict__'):
                keys = list(choice.message.__dict__.keys())
                print(f"  字段: {keys}")

        print(f"\n  找到可用模型!")
        success = True
        break

    except Exception as e:
        elapsed = time.time() - start_time
        err_type = type(e).__name__
        err_msg = str(e)[:150]
        print(f"  [失败] {err_type}: {err_msg}")

print("\n" + "="*60)
if success:
    print("  测试通过!")
else:
    print("  所有模型都不可用")
print("="*60)
