"""
任务管理器测试脚本 - 阶段2：生命周期管理
"""
import sys
import asyncio
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from console_utils import fix_console_encoding
fix_console_encoding()

from PIL import Image
from task.manager import TaskManager, get_task_manager, reset_task_manager
from task.record import TaskStatus, TaskPriority
from task.limiter import get_limiter


def create_test_image(size_kb=50):
    """创建测试图片"""
    img = Image.new("RGB", (200, 200), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return buffer.getvalue()


async def test_task_registration():
    """测试任务注册"""
    print("\n=== 测试任务注册 ===")

    manager = get_task_manager()

    # 创建测试数据
    file_bytes = create_test_image()

    # 注册任务
    task_id = manager.register_task(
        file_bytes=file_bytes,
        file_name="test.jpg",
        category="midi_dress",
        style="elegant",
        season="summer"
    )

    print(f"  [OK] 任务注册成功: {task_id}")

    # 查询任务
    record = manager.get_task(task_id)
    if record and record.task_id == task_id:
        print(f"  [OK] 任务查询成功: status={record.status.value}")
    else:
        print(f"  [FAIL] 任务查询失败")
        return False

    # 查询状态
    status = manager.get_task_status(task_id)
    if status and status["task_id"] == task_id:
        print(f"  [OK] 状态查询成功")
    else:
        print(f"  [FAIL] 状态查询失败")
        return False

    return True


async def test_task_status_transitions():
    """测试任务状态转换"""
    print("\n=== 测试任务状态转换 ===")

    manager = get_task_manager()
    manager.tasks.clear()  # 清空现有任务

    file_bytes = create_test_image()
    task_id = manager.register_task(
        file_bytes=file_bytes,
        file_name="test.jpg",
        category="midi_dress",
        style="elegant"
    )

    record = manager.get_task(task_id)

    # 测试状态转换
    print(f"  初始状态: {record.status.value}")

    record.mark_queued()
    print(f"  标记排队: {record.status.value}")
    if record.status != TaskStatus.QUEUED:
        print(f"  [FAIL] 状态转换失败")
        return False

    record.mark_running()
    print(f"  标记运行: {record.status.value}")
    if record.status != TaskStatus.RUNNING:
        print(f"  [FAIL] 状态转换失败")
        return False

    # 模拟完成
    record.mark_completed({"test": "result"})
    print(f"  标记完成: {record.status.value}")
    if record.status != TaskStatus.COMPLETED:
        print(f"  [FAIL] 状态转换失败")
        return False

    print(f"  [OK] 所有状态转换正常")
    return True


async def test_progress_tracking():
    """测试进度跟踪"""
    print("\n=== 测试进度跟踪 ===")

    manager = get_task_manager()
    manager.tasks.clear()

    file_bytes = create_test_image()
    task_id = manager.register_task(
        file_bytes=file_bytes,
        file_name="test.jpg",
        category="midi_dress",
        style="elegant"
    )

    record = manager.get_task(task_id)

    # 模拟进度更新
    record.update_progress("step1", 10, "Starting")
    print(f"  进度1: {record.progress.progress_percent}% - {record.progress.message}")

    record.update_progress("step2", 50, "Processing")
    print(f"  进度2: {record.progress.progress_percent}% - {record.progress.message}")

    record.update_progress("step3", 100, "Done")
    print(f"  进度3: {record.progress.progress_percent}% - {record.progress.message}")

    if record.progress.progress_percent == 100:
        print(f"  [OK] 进度跟踪正常")
        return True
    else:
        print(f"  [FAIL] 进度跟踪异常")
        return False


async def test_task_failure():
    """测试任务失败处理"""
    print("\n=== 测试任务失败处理 ===")

    manager = get_task_manager()
    manager.tasks.clear()

    file_bytes = create_test_image()
    task_id = manager.register_task(
        file_bytes=file_bytes,
        file_name="test.jpg",
        category="midi_dress",
        style="elegant"
    )

    record = manager.get_task(task_id)

    # 标记失败
    record.mark_failed("Test error", {"code": 500})

    if record.status == TaskStatus.FAILED:
        print(f"  [OK] 失败状态正确: {record.error}")
        return True
    else:
        print(f"  [FAIL] 失败状态错误")
        return False


async def test_task_cancellation():
    """测试任务取消"""
    print("\n=== 测试任务取消 ===")

    manager = get_task_manager()
    manager.tasks.clear()

    file_bytes = create_test_image()
    task_id = manager.register_task(
        file_bytes=file_bytes,
        file_name="test.jpg",
        category="midi_dress",
        style="elegant"
    )

    # 取消任务
    result = manager.cancel_task(task_id, "User cancelled")

    if result:
        record = manager.get_task(task_id)
        if record.status == TaskStatus.CANCELLED:
            print(f"  [OK] 任务取消成功")
            return True

    print(f"  [FAIL] 任务取消失败")
    return False


async def test_task_statistics():
    """测试任务统计"""
    print("\n=== 测试任务统计 ===")

    manager = get_task_manager()
    manager.tasks.clear()

    # 创建多个任务
    file_bytes = create_test_image()
    for i in range(5):
        task_id = manager.register_task(
            file_bytes=file_bytes,
            file_name=f"test{i}.jpg",
            category="midi_dress",
            style="elegant"
        )
        if i < 2:
            manager.get_task(task_id).mark_completed()
        elif i < 4:
            manager.get_task(task_id).mark_running()
        else:
            manager.get_task(task_id).mark_failed("Test")

    # 获取统计
    stats = manager.get_stats()
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  活跃任务: {stats['active_tasks']}")
    print(f"  状态分布: {stats['by_status']}")

    if stats["total_tasks"] == 5:
        print(f"  [OK] 统计信息正确")
        return True
    else:
        print(f"  [FAIL] 统计信息错误")
        return False


async def test_task_listing():
    """测试任务列表"""
    print("\n=== 测试任务列表 ===")

    manager = get_task_manager()
    manager.tasks.clear()

    # 创建多个任务
    file_bytes = create_test_image()
    for i in range(3):
        manager.register_task(
            file_bytes=file_bytes,
            file_name=f"test{i}.jpg",
            category="midi_dress",
            style="elegant"
        )

    # 列出所有任务
    all_tasks = manager.list_tasks()
    print(f"  所有任务: {len(all_tasks)} 个")

    # 按状态过滤
    pending_tasks = manager.list_tasks(status=TaskStatus.PENDING)
    print(f"  等待中任务: {len(pending_tasks)} 个")

    if len(all_tasks) == 3 and len(pending_tasks) == 3:
        print(f"  [OK] 任务列表正常")
        return True
    else:
        print(f"  [FAIL] 任务列表异常")
        return False


async def test_progress_callback():
    """测试进度回调"""
    print("\n=== 测试进度回调 ===")

    manager = get_task_manager()
    manager.tasks.clear()

    # 收集回调数据
    callback_data = []

    def test_callback(task_id, step, percent, message):
        callback_data.append((step, percent, message))

    file_bytes = create_test_image()
    task_id = manager.register_task(
        file_bytes=file_bytes,
        file_name="test.jpg",
        category="midi_dress",
        style="elegant"
    )

    # 注册回调
    manager.register_progress_callback(task_id, test_callback)

    # 触发进度通知
    manager._notify_progress(task_id, "test_step", 75, "Test message")

    # 等待回调执行
    await asyncio.sleep(0.1)

    if len(callback_data) == 1:
        step, percent, message = callback_data[0]
        if step == "test_step" and percent == 75:
            print(f"  [OK] 回调执行正常: {step} - {percent}% - {message}")
            return True

    print(f"  [FAIL] 回调执行异常")
    return False


def main():
    """运行所有测试"""
    print("=" * 50)
    print("阶段2测试：生命周期管理模块")
    print("=" * 50)

    # 重置管理器
    reset_task_manager()

    tests = [
        ("任务注册", test_task_registration),
        ("状态转换", test_task_status_transitions),
        ("进度跟踪", test_progress_tracking),
        ("失败处理", test_task_failure),
        ("任务取消", test_task_cancellation),
        ("任务统计", test_task_statistics),
        ("任务列表", test_task_listing),
        ("进度回调", test_progress_callback),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            result = asyncio.run(test_func())
            if result:
                passed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("✓ 阶段2测试全部通过!")
        return True
    else:
        print(f"✗ 阶段2测试有 {total - passed} 个失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
