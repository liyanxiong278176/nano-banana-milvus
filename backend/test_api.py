"""
测试后端API - 验证前后端联调

测试步骤：
1. 启动 Docker Desktop
2. 启动 Milvus 和 Neo4j
3. 初始化数据库
4. 测试图片上传
"""
import requests
import time
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查"""
    print("\n[1/5] 测试健康检查...")
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    print(f"  状态: {data['status']}")
    print(f"  版本: {data['version']}")
    print(f"  Milvus连接: {data['milvus_connected']}")
    print(f"  数据库就绪: {data['database_ready']}")
    return data

def test_categories():
    """测试获取品类列表"""
    print("\n[2/5] 测试获取品类列表...")
    response = requests.get(f"{BASE_URL}/api/categories")
    data = response.json()
    categories = data.get("categories", [])
    print(f"  可用品类: {len(categories)} 个")
    for cat in categories[:5]:
        print(f"    - {cat['value']}: {cat['label']}")
    return categories

def test_styles():
    """测试获取风格列表"""
    print("\n[3/5] 测试获取风格列表...")
    response = requests.get(f"{BASE_URL}/api/styles")
    data = response.json()
    styles = data.get("styles", [])
    print(f"  可用风格: {len(styles)} 个")
    for style in styles[:5]:
        print(f"    - {style['value']}: {style['label']}")
    return styles

def test_upload():
    """测试图片上传"""
    print("\n[4/5] 测试图片上传...")

    # 选择测试图片
    new_products_dir = Path(__file__).parent / "new_products"
    test_images = list(new_products_dir.glob("NEW_*.jpg"))

    if not test_images:
        print("  错误: new_products 目录中没有测试图片")
        return None

    # 选择一张较小的图片
    test_image = sorted(test_images, key=lambda p: p.stat().st_size)[0]
    print(f"  使用测试图片: {test_image.name} ({test_image.stat().st_size} bytes)")

    with open(test_image, "rb") as f:
        files = {"file": (test_image.name, f, "image/jpeg")}
        data = {
            "category": "midi_dress",
            "style": "elegant",
            "season": "summer",
            "scene_hint": "beach",
            "enable_quality_check": False,
            "retrieval_mode": "hybrid",  # 并行混合检索
            "enable_multi_hop": True,
            "max_hops": 3,
            "use_workflow": True  # 使用 LangGraph 工作流
        }

        response = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        task_id = result.get("task_id")
        print(f"  ✓ 上传成功!")
        print(f"  任务ID: {task_id}")
        print(f"  产品ID: {result.get('product_id')}")
        print(f"  消息: {result.get('message')}")
        return task_id
    else:
        print(f"  ✗ 上传失败: {response.status_code}")
        print(f"  错误: {response.text}")
        return None

def test_task_status(task_id):
    """测试任务状态查询"""
    print("\n[5/5] 测试任务状态查询...")

    max_wait = 60  # 最多等待60秒
    start_time = time.time()

    while time.time() - start_time < max_wait:
        response = requests.get(f"{BASE_URL}/api/tasks/{task_id}")

        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            progress = data.get("progress", 0) * 100

            print(f"  状态: {status} | 进度: {progress:.0f}%")

            if status == "completed":
                result = data.get("result", {})
                print(f"\n  ✓ 任务完成!")
                print(f"  生成图片: {result.get('generated_count', 0)} 张")
                print(f"  检索结果: {result.get('retrieved_count', 0)} 个")

                # 显示生成的图片URL
                images = result.get("generated_images", [])
                if images:
                    print(f"\n  生成的图片:")
                    for img_url in images:
                        print(f"    - {BASE_URL}{img_url}")

                # 显示检索结果
                retrieved = result.get("retrieved_products", [])
                if retrieved:
                    print(f"\n  参考的爆款:")
                    for r in retrieved[:3]:
                        print(f"    - {r.get('product_id')}: {r.get('style')} {r.get('category')}")

                return True

            elif status == "failed":
                error = data.get("error", "未知错误")
                print(f"\n  ✗ 任务失败: {error}")
                return False

        time.sleep(2)

    print(f"\n  ✗ 任务超时")
    return False

def main():
    print("="*60)
    print("  后端 API 测试")
    print("="*60)

    try:
        # 1. 健康检查
        health = test_health()

        if not health.get("milvus_connected") or not health.get("database_ready"):
            print("\n  ⚠️ 警告: Milvus 未连接或数据库未就绪")
            print("  请先启动 Docker Desktop，然后运行:")
            print("    docker start milvus-standalone")
            print("    docker start neo4j")
            print("\n  如果容器不存在，请创建:")
            print("    docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest")
            print("    docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
            return

        # 2. 获取品类和风格
        categories = test_categories()
        styles = test_styles()

        # 3. 测试上传
        task_id = test_upload()

        if task_id:
            # 4. 查询任务状态
            test_task_status(task_id)

    except requests.exceptions.ConnectionError:
        print("\n  ✗ 错误: 无法连接到后端服务")
        print("  请确保后端服务正在运行: cd backend && python api.py")
    except Exception as e:
        print(f"\n  ✗ 错误: {e}")

    print("\n" + "="*60)
    print("  测试完成")
    print("="*60)

if __name__ == "__main__":
    main()
