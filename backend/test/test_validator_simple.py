"""
简单测试脚本 - 阶段1：接单校验
"""
import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from console_utils import fix_console_encoding
fix_console_encoding()

from PIL import Image
from task.validator import TaskValidator, ValidationResult
from task.limiter import TaskLimiter, RejectionPolicy, get_limiter
import asyncio


def test_validator():
    """测试参数校验器"""
    print("\n=== 测试参数校验器 ===")
    
    validator = TaskValidator()
    
    # 创建测试图片
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    file_bytes = buffer.getvalue()
    
    tests_passed = 0
    tests_total = 0
    
    # 测试1: 有效请求
    tests_total += 1
    passed, error = validator.validate_upload_request(
        file_bytes=file_bytes,
        filename="test.jpg",
        category="midi_dress",
        style="elegant"
    )
    if passed and error is None:
        print("  [PASS] 有效请求通过校验")
        tests_passed += 1
    else:
        print(f"  [FAIL] 有效请求未通过: {error}")
    
    # 测试2: 无效品类
    tests_total += 1
    passed, error = validator.validate_upload_request(
        file_bytes=file_bytes,
        filename="test.jpg",
        category="invalid_category",
        style="elegant"
    )
    if not passed and error.code == ValidationResult.INVALID_CATEGORY:
        print("  [PASS] 无效品类被拒绝")
        tests_passed += 1
    else:
        print(f"  [FAIL] 无效品类未正确拒绝")
    
    # 测试3: 无效风格
    tests_total += 1
    passed, error = validator.validate_upload_request(
        file_bytes=file_bytes,
        filename="test.jpg",
        category="midi_dress",
        style="invalid_style"
    )
    if not passed and error.code == ValidationResult.INVALID_STYLE:
        print("  [PASS] 无效风格被拒绝")
        tests_passed += 1
    else:
        print(f"  [FAIL] 无效风格未正确拒绝")
    
    # 测试4: 文件太大
    tests_total += 1
    large_file = b"x" * (11 * 1024 * 1024)
    passed, error = validator.validate_upload_request(
        file_bytes=large_file,
        filename="large.jpg",
        category="midi_dress",
        style="elegant"
    )
    if not passed and error.code == ValidationResult.FILE_TOO_LARGE:
        print("  [PASS] 大文件被拒绝")
        tests_passed += 1
    else:
        print(f"  [FAIL] 大文件未正确拒绝")
    
    print(f"\n参数校验器: {tests_passed}/{tests_total} 测试通过")
    return tests_passed == tests_total


async def test_limiter():
    """测试并发限制器"""
    print("\n=== 测试并发限制器 ===")
    
    limiter = TaskLimiter(max_concurrent=2)
    
    tests_passed = 0
    tests_total = 0
    
    # 测试1: 在限制内获取许可
    tests_total += 1
    if await limiter.acquire("task1"):
        limiter.release("task1")
        print("  [PASS] 在限制内成功获取许可")
        tests_passed += 1
    else:
        print("  [FAIL] 在限制内未能获取许可")
    
    # 测试2: 超过限制被拒绝
    tests_total += 1
    limiter2 = TaskLimiter(max_concurrent=1, rejection_policy=RejectionPolicy.REJECT)
    await limiter2.acquire("task1")
    acquired = await limiter2.acquire("task2")
    limiter2.release("task1")
    if not acquired:
        print("  [PASS] 超过限制被拒绝")
        tests_passed += 1
    else:
        print("  [FAIL] 超过限制未被拒绝")
    
    # 测试3: 统计信息
    tests_total += 1
    stats = limiter.get_stats()
    if "current_running" in stats and "max_concurrent" in stats:
        print("  [PASS] 统计信息正确")
        tests_passed += 1
    else:
        print("  [FAIL] 统计信息不正确")
    
    print(f"\n并发限制器: {tests_passed}/{tests_total} 测试通过")
    return tests_passed == tests_total


def main():
    """运行所有测试"""
    print("=" * 50)
    print("阶段1测试：接单校验模块")
    print("=" * 50)
    
    # 测试参数校验器
    validator_ok = test_validator()
    
    # 测试并发限制器
    limiter_ok = asyncio.run(test_limiter())
    
    # 总结
    print("\n" + "=" * 50)
    if validator_ok and limiter_ok:
        print("所有测试通过!")
        print("=" * 50)
        return True
    else:
        print("有测试失败，请检查")
        print("=" * 50)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
