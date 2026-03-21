"""
测试优化模块集成 - 展示新加的模块如何工作

这个测试展示：
1. 参数校验（Validator）- 拦绝无效请求
2. 并发控制（Limiter）- 限制同时处理的任务数
3. 任务管理（TaskManager）- 统一的任务生命周期管理
"""
import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from console_utils import fix_console_encoding
fix_console_encoding()

from PIL import Image


def create_test_image(size_kb=50):
    """创建测试图片"""
    img = Image.new("RGB", (200, 200), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return buffer.getvalue()


def test_validator():
    """测试参数校验模块"""
    print("\n" + "=" * 60)
    print("测试1: 参数校验模块 (Validator)")
    print("=" * 60)

    from task.validator import get_validator

    validator = get_validator()

    # 测试1.1: 有效参数
    print("\n[测试1.1] 有效参数")
    file_bytes = create_test_image(50)
    passed, error = validator.validate_upload_request(
        file_bytes=file_bytes,
        filename="dress.jpg",
        category="midi_dress",
        style="elegant"
    )
    print(f"  结果: {'通过 [PASS]' if passed else f'失败 [FAIL] - {error.message}'}")

    # 测试1.2: 无效品类
    print("\n[测试1.2] 无效品类（应该被拒绝）")
    passed, error = validator.validate_upload_request(
        file_bytes=file_bytes,
        filename="dress.jpg",
        category="invalid_category",
        style="elegant"
    )
    print(f"  结果: {'通过 [FAIL]' if passed else f'被拒绝 [PASS] - {error.code}'}")

    # 测试1.3: 文件太大
    print("\n[测试1.3] 文件太大（应该被拒绝）")
    large_file = b"x" * (11 * 1024 * 1024)
    passed, error = validator.validate_upload_request(
        file_bytes=large_file,
        filename="large.jpg",
        category="midi_dress",
        style="elegant"
    )
    print(f"  结果: {'通过 [FAIL]' if passed else f'被拒绝 [PASS] - {error.code}'}")

    # 测试1.4: 查看允许的参数
    print("\n[测试1.4] 允许的参数")
    allowed = validator.get_allowed_values()
    print(f"  允许的品类: {len(allowed['categories'])} 个")
    print(f"  允许的风格: {len(allowed['styles'])} 个")
    print(f"  品类示例: {', '.join(list(allowed['categories'])[:3])}")


def test_limiter():
    """测试并发控制模块"""
    print("\n" + "=" * 60)
    print("测试2: 并发控制模块 (Limiter)")
    print("=" * 60)

    from task.limiter import get_limiter, reset_limiter

    reset_limiter()
    limiter = get_limiter()

    print(f"\n[测试2.1] 并发配置")
    stats = limiter.get_stats()
    print(f"  最大并发数: {stats.get('max_concurrent', 'N/A')}")
    print(f"  当前活跃: {stats.get('active', 0)}")
    print(f"  当前排队: {stats.get('queued', 0)}")


def test_task_manager():
    """测试任务管理模块"""
    print("\n" + "=" * 60)
    print("测试3: 任务管理模块 (TaskManager)")
    print("=" * 60)

    from api import setup_task_manager

    task_manager = setup_task_manager()

    # 测试3.1: 注册任务
    print("\n[测试3.1] 注册任务")
    file_bytes = create_test_image(50)
    task_id = task_manager.register_task(
        file_bytes=file_bytes,
        file_name="test_dress.jpg",
        category="midi_dress",
        style="elegant"
    )
    print(f"  任务ID: {task_id}")

    # 测试3.2: 查询任务状态
    print("\n[测试3.2] 查询任务状态")
    status = task_manager.get_task_status(task_id)
    if status:
        print(f"  状态: {status['status']}")
        print(f"  品类: {status['input']['category']}")
        print(f"  风格: {status['input']['style']}")

    # 测试3.3: 统计信息
    print("\n[测试3.3] 统计信息")
    stats = task_manager.get_stats()
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  活跃任务: {stats['active_tasks']}")
    print(f"  平均耗时: {stats['avg_duration_seconds']:.2f}秒")


def test_api_integration():
    """测试 API 集成"""
    print("\n" + "=" * 60)
    print("测试4: API 端点集成")
    print("=" * 60)

    from api import app

    print("\n[测试4.1] 可用的 API 端点")
    upload_endpoints = []
    for route in app.routes:
        if hasattr(route, 'path') and 'upload' in route.path:
            methods = getattr(route, 'methods', set())
            upload_endpoints.append(f"{list(methods)[0] if methods else 'GET'} {route.path}")

    for endpoint in sorted(upload_endpoints):
        print(f"  {endpoint}")

    print("\n[测试4.2] 集成说明")
    print("  POST /api/upload - 现在集成了:")
    print("    - Validator: 参数校验")
    print("    - Limiter: 并发控制")
    print("    - TaskManager: 任务管理")
    print("  请求无效参数时会返回 400 错误")
    print("  请求会被记录在 TaskManager 中")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("优化模块集成测试")
    print("=" * 60)
    print("\n这个测试展示新加的优化模块如何集成到主流程中")

    try:
        test_validator()
        test_limiter()
        test_task_manager()
        test_api_integration()

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        print("\n总结:")
        print("  1. Validator - 在 /api/upload 入口处校验参数")
        print("  2. Limiter - 限制并发任务数量")
        print("  3. TaskManager - 管理任务生命周期")
        print("\n下次调用 /api/upload 时，会看到类似的日志:")
        print('  2026-03-21 | INFO | === 上传请求 | file=dress.jpg, category=midi_dress ===')
        print('  2026-03-21 | INFO | 文件读取完成 | size=102400 bytes')
        print('  2026-03-21 | INFO | 参数校验通过 | Validator检查通过')
        print('  2026-03-21 | INFO | TaskManager 任务注册成功 | tm_task_id=task_xxx')

        return True

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
