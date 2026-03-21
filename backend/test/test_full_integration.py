"""
完整流程集成测试 - 从图片上传到生成图片

测试流程：
1. 上传图片（带参数校验）
2. 任务注册和异步执行
3. 进度跟踪
4. 结果查询
"""
import sys
import asyncio
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from console_utils import fix_console_encoding
fix_console_encoding()

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

from PIL import Image


def create_test_image(name="test", size_kb=50):
    """创建测试图片"""
    img = Image.new("RGB", (200, 200), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return buffer.getvalue()


def test_api_integration():
    """测试 API 模块集成"""
    print("\n" + "=" * 60)
    print("API 模块集成测试")
    print("=" * 60)

    try:
        # 导入 API 模块中的关键函数
        from api import (
            setup_task_manager,
            get_validator,
            workflow_executor_v2
        )

        print("[OK] API 模块导入成功")

        # 测试 TaskManager 设置
        task_manager = setup_task_manager()
        print(f"[OK] TaskManager 初始化成功 | workflow_executor={'已设置' if task_manager.workflow_executor else '未设置'}")

        # 测试 Validator
        validator = get_validator()
        allowed = validator.get_allowed_values()
        print(f"[OK] Validator 初始化成功 | categories={len(allowed['categories'])}, styles={len(allowed['styles'])}")

        return True

    except Exception as e:
        print(f"[FAIL] API 模块集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_registration():
    """测试任务注册流程"""
    print("\n" + "=" * 60)
    print("任务注册流程测试")
    print("=" * 60)

    try:
        from api import setup_task_manager, get_validator

        task_manager = setup_task_manager()
        validator = get_validator()

        # 创建测试图片
        file_bytes = create_test_image("test_dress", 50)
        print(f"[1/4] 测试图片已创建 | size={len(file_bytes)} bytes")

        # 参数校验
        passed, error = validator.validate_upload_request(
            file_bytes=file_bytes,
            filename="test_dress.jpg",
            category="midi_dress",
            style="elegant"
        )

        if not passed:
            print(f"[FAIL] 参数校验失败: {error.message}")
            return False

        print(f"[2/4] 参数校验通过")

        # 注册任务
        task_id = task_manager.register_task(
            file_bytes=file_bytes,
            file_name="test_dress.jpg",
            category="midi_dress",
            style="elegant"
        )
        print(f"[3/4] 任务已注册 | task_id={task_id}")

        # 查询任务状态
        status = task_manager.get_task_status(task_id)
        if not status:
            print(f"[FAIL] 无法查询任务状态")
            return False

        print(f"[4/4] 任务状态查询成功 | status={status['status']}")

        return True

    except Exception as e:
        print(f"[FAIL] 任务注册测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_workflow_execution_simulation():
    """测试工作流执行（模拟）"""
    print("\n" + "=" * 60)
    print("工作流执行模拟测试")
    print("=" * 60)

    try:
        from api import setup_task_manager

        task_manager = setup_task_manager()

        # 创建测试图片
        file_bytes = create_test_image("test_dress", 50)

        # 注册任务
        task_id = task_manager.register_task(
            file_bytes=file_bytes,
            file_name="test_dress.jpg",
            category="midi_dress",
            style="elegant"
        )
        print(f"[1/3] 任务已注册 | task_id={task_id}")

        # 注册进度回调
        progress_updates = []

        def progress_callback(tid, step, percent, message):
            progress_updates.append((step, percent, message))
            print(f"      进度: [{step}] {percent}% - {message}")

        task_manager.register_progress_callback(task_id, progress_callback)
        print(f"[2/3] 进度回调已注册")

        # 注意：由于需要实际的 Milvus 和 API 密钥，这里不执行真实的工作流
        # 只验证任务管理器状态管理是否正常
        record = task_manager.get_task(task_id)
        if record:
            print(f"[3/3] 任务记录获取成功 | status={record.status.value}")

        return True

    except Exception as e:
        print(f"[FAIL] 工作流执行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_websocket_manager():
    """测试 WebSocket 管理器"""
    print("\n" + "=" * 60)
    print("WebSocket 管理器测试")
    print("=" * 60)

    try:
        from task.websocket_manager import get_websocket_manager, reset_websocket_manager

        reset_websocket_manager()
        ws_manager = get_websocket_manager()

        print(f"[1/3] WebSocketManager 初始化成功")
        print(f"[2/3] 连接数: {ws_manager.get_connection_count()}")
        print(f"[3/3] 活跃任务: {ws_manager.get_active_tasks()}")

        return True

    except Exception as e:
        print(f"[FAIL] WebSocket 管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics():
    """测试统计信息"""
    print("\n" + "=" * 60)
    print("统计信息测试")
    print("=" * 60)

    try:
        from api import setup_task_manager

        task_manager = setup_task_manager()
        stats = task_manager.get_stats()

        print(f"[1/3] 总任务数: {stats['total_tasks']}")
        print(f"[2/3] 活跃任务: {stats['active_tasks']}")
        print(f"[3/3] 平均耗时: {stats['avg_duration_seconds']:.2f}s")

        return True

    except Exception as e:
        print(f"[FAIL] 统计信息测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("完整流程集成测试")
    print("=" * 60)

    tests = [
        ("API 模块集成", test_api_integration),
        ("任务注册流程", test_task_registration),
        ("工作流执行模拟", lambda: asyncio.run(test_workflow_execution_simulation())),
        ("WebSocket 管理器", test_websocket_manager),
        ("统计信息", test_statistics),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] {name} 测试异常: {e}")
            results.append((name, False))

    # 测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed_count = sum(1 for _, result in results if result)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n总计: {passed_count}/{len(results)} 测试通过")

    if passed_count == len(results):
        print("所有集成测试通过!")
        return True
    else:
        print("部分测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
