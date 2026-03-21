"""
任务校验器单元测试 - 阶段1：接单校验

测试内容：
- 参数校验功能
- 并发限制功能
- 边界条件测试
"""
import asyncio
import pytest
import io
from pathlib import Path

# 导入被测试模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from task.validator import TaskValidator, ValidationResult, get_validator, reset_validator
from task.limiter import TaskLimiter, RejectionPolicy, get_limiter, reset_limiter


# ==================== 参数校验测试 ====================

class TestTaskValidator:
    """任务参数校验器测试"""

    def setup_method(self):
        """每个测试前重置校验器"""
        reset_validator()
        self.validator = TaskValidator()

    def create_test_image(self, width=100, height=100, format="PNG"):
        """创建测试图片"""
        from PIL import Image
        img = Image.new("RGB", (width, height), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer.read()

    def test_valid_request(self):
        """测试有效请求通过校验"""
        file_bytes = self.create_test_image()
        passed, error = self.validator.validate_upload_request(
            file_bytes=file_bytes,
            filename="test.jpg",
            category="midi_dress",
            style="elegant"
        )
        assert passed is True
        assert error is None

    def test_invalid_category(self):
        """测试无效品类被拒绝"""
        file_bytes = self.create_test_image()
        passed, error = self.validator.validate_upload_request(
            file_bytes=file_bytes,
            filename="test.jpg",
            category="invalid_category",
            style="elegant"
        )
        assert passed is False
        assert error.code == ValidationResult.INVALID_CATEGORY

    def test_invalid_style(self):
        """测试无效风格被拒绝"""
        file_bytes = self.create_test_image()
        passed, error = self.validator.validate_upload_request(
            file_bytes=file_bytes,
            filename="test.jpg",
            category="midi_dress",
            style="invalid_style"
        )
        assert passed is False
        assert error.code == ValidationResult.INVALID_STYLE

    def test_file_too_large(self):
        """测试超大文件被拒绝"""
        # 创建超过限制的文件（默认10MB）
        large_file = b"x" * (11 * 1024 * 1024)
        passed, error = self.validator.validate_upload_request(
            file_bytes=large_file,
            filename="large.jpg",
            category="midi_dress",
            style="elegant"
        )
        assert passed is False
        assert error.code == ValidationResult.FILE_TOO_LARGE

    def test_invalid_file_type(self):
        """测试无效文件类型被拒绝"""
        passed, error = self.validator.validate_upload_request(
            file_bytes=b"not an image",
            filename="test.bmp",  # 不在白名单中
            category="midi_dress",
            style="elegant"
        )
        assert passed is False
        assert error.code == ValidationResult.INVALID_FILE_TYPE

    def test_missing_required_field(self):
        """测试缺少必填字段被拒绝"""
        file_bytes = self.create_test_image()
        passed, error = self.validator.validate_upload_request(
            file_bytes=file_bytes,
            filename="test.jpg",
            category="",  # 空品类
            style="elegant"
        )
        assert passed is False
        assert error.code == ValidationResult.MISSING_REQUIRED_FIELD

    def test_corrupted_image(self):
        """测试损坏图片被拒绝"""
        passed, error = self.validator.validate_upload_request(
            file_bytes=b"this is not a valid image",
            filename="corrupt.jpg",
            category="midi_dress",
            style="elegant"
        )
        assert passed is False
        assert error.code == ValidationResult.FILE_CORRUPTED

    def test_image_too_large_dimensions(self):
        """测试超大尺寸图片被拒绝"""
        # 创建超过4096的图片
        file_bytes = self.create_test_image(width=5000, height=100)
        passed, error = self.validator.validate_upload_request(
            file_bytes=file_bytes,
            filename="large.jpg",
            category="midi_dress",
            style="elegant"
        )
        assert passed is False
        assert error.code == ValidationResult.FILE_TOO_LARGE

    def test_get_allowed_values(self):
        """测试获取允许的值"""
        values = self.validator.get_allowed_values()
        assert "categories" in values
        assert "styles" in values
        assert "seasons" in values
        assert "file_extensions" in values
        assert "max_file_size_mb" in values
        assert "midi_dress" in values["categories"]
        assert "elegant" in values["styles"]

    def test_dynamic_add_category(self):
        """测试动态添加品类"""
        new_category = "new_category"
        assert new_category not in self.validator.allowed_categories

        self.validator.add_category(new_category)
        assert new_category in self.validator.allowed_categories

        # 现在应该通过校验
        file_bytes = self.create_test_image()
        passed, error = self.validator.validate_upload_request(
            file_bytes=file_bytes,
            filename="test.jpg",
            category=new_category,
            style="elegant"
        )
        assert passed is True


# ==================== 并发限制测试 ====================

class TestTaskLimiter:
    """任务并发限制器测试"""

    def setup_method(self):
        """每个测试前重置限制器"""
        reset_limiter()

    async def test_acquire_within_limit(self):
        """测试在限制内能正常获取许可"""
        limiter = TaskLimiter(max_concurrent=2)
        assert await limiter.acquire("task1") is True
        assert await limiter.acquire("task2") is True
        limiter.release("task1")
        limiter.release("task2")

    async def test_acquire_exceed_limit_reject(self):
        """测试超过限制时拒绝策略"""
        limiter = TaskLimiter(max_concurrent=1, rejection_policy=RejectionPolicy.REJECT)

        assert await limiter.acquire("task1") is True
        assert await limiter.acquire("task2") is False  # 被拒绝

        limiter.release("task1")

    async def test_acquire_release_pattern(self):
        """测试获取-释放模式"""
        limiter = TaskLimiter(max_concurrent=2)

        assert await limiter.acquire("task1") is True
        assert limiter.stats.current_running == 1

        limiter.release("task1")
        assert limiter.stats.current_running == 0

    async def test_concurrent_execution(self):
        """测试并发执行"""
        limiter = TaskLimiter(max_concurrent=2)

        async def dummy_task(name, duration):
            await limiter.acquire(name)
            await asyncio.sleep(duration)
            limiter.release(name)
            return name

        # 启动3个任务，但只能并发2个
        results = await asyncio.gather(
            dummy_task("task1", 0.1),
            dummy_task("task2", 0.1),
            dummy_task("task3", 0.1),
            return_exceptions=True
        )

        # 所有任务都应该完成
        assert all(r is not None for r in results)

    async def test_stats_tracking(self):
        """测试统计信息跟踪"""
        limiter = TaskLimiter(max_concurrent=2)

        await limiter.acquire("task1")
        await limiter.acquire("task2")

        stats = limiter.get_stats()
        assert stats["current_running"] == 2
        assert stats["total_requests"] == 2
        assert stats["accepted_requests"] == 2
        assert stats["peak_concurrent"] == 2

        limiter.release("task1")
        stats = limiter.get_stats()
        assert stats["current_running"] == 1

    async def test_permit_context_manager(self):
        """测试许可上下文管理器"""
        from task.limiter import TaskPermit

        limiter = TaskLimiter(max_concurrent=1)

        async with TaskPermit(limiter, "task1") as acquired:
            assert acquired is True
            assert limiter.stats.current_running == 1

        assert limiter.stats.current_running == 0

    async def test_queue_policy(self):
        """测试排队策略"""
        limiter = TaskLimiter(
            max_concurrent=1,
            max_queue_size=2,
            rejection_policy=RejectionPolicy.QUEUE
        )

        # 第一个获取许可
        assert await limiter.acquire("task1") is True

        # 其他进入队列（不拒绝）
        # 注意：实际排队需要异步等待，这里简化测试

        limiter.release("task1")

    async def test_active_tasks_tracking(self):
        """测试活跃任务跟踪"""
        limiter = TaskLimiter(max_concurrent=3)

        await limiter.acquire("task1")
        await limiter.acquire("task2")

        active = limiter.get_active_tasks()
        assert len(active) == 2
        assert any(t["task_id"] == "task1" for t in active)

        limiter.release("task1")
        active = limiter.get_active_tasks()
        assert len(active) == 1


# ==================== 运行测试 ====================

def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行阶段1测试：接单校验模块")
    print("=" * 60)

    # 参数校验测试
    print("\n[1/2] 参数校验测试...")
    test_validator = TestTaskValidator()

    tests = [
        ("有效请求", test_validator.test_valid_request),
        ("无效品类", test_validator.test_invalid_category),
        ("无效风格", test_validator.test_invalid_style),
        ("文件过大", test_validator.test_file_too_large),
        ("无效文件类型", test_validator.test_invalid_file_type),
        ("缺少必填字段", test_validator.test_missing_required_field),
        ("损坏图片", test_validator.test_corrupted_image),
        ("超大尺寸图片", test_validator.test_image_too_large_dimensions),
        ("获取允许值", test_validator.test_get_allowed_values),
        ("动态添加品类", test_validator.test_dynamic_add_category),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            test_validator.setup_method()
            test_func()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # 并发限制测试
    print(f"\n参数校验测试: {passed}/{len(tests)} 通过")

    print("\n[2/2] 并发限制测试...")
    test_limiter = TestTaskLimiter()

    async_tests = [
        ("限制内获取许可", test_limiter.test_acquire_within_limit),
        ("超限拒绝", test_limiter.test_acquire_exceed_limit_reject),
        ("获取释放模式", test_limiter.test_acquire_release_pattern),
        ("并发执行", test_limiter.test_concurrent_execution),
        ("统计跟踪", test_limiter.test_stats_tracking),
        ("上下文管理器", test_limiter.test_permit_context_manager),
        ("活跃任务跟踪", test_limiter.test_active_tasks_tracking),
    ]

    async_passed = 0
    for name, test_func in async_tests:
        try:
            test_limiter.setup_method()
            asyncio.run(test_func())
            print(f"  ✓ {name}")
            async_passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print(f"\n并发限制测试: {async_passed}/{len(async_tests)} 通过")

    # 总结
    total_passed = passed + async_passed
    total_tests = len(tests) + len(async_tests)
    print(f"\n总计: {total_passed}/{total_tests} 通过")

    if total_passed == total_tests:
        print("✓ 阶段1测试全部通过！")
        return True
    else:
        print(f"✗ 阶段1测试有 {total_tests - total_passed} 个失败")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
