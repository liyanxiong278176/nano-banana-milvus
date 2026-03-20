"""
控制台编码修复工具模块

统一处理 Windows 控制台 UTF-8 编码问题，避免在多个文件中重复代码。
"""
import sys


def fix_console_encoding():
    """
    ���复 Windows 控制台编码问题

    在 Windows 平台上，将 stdout 和 stderr 包装为 UTF-8 编码的 TextIOWrapper，
    以正确显示中文等非 ASCII 字符。

    注意：此函数会检查是否已经包装过，避免重复包装导致问题。

    使用示例：
        from console_utils import fix_console_encoding
        fix_console_encoding()
    """
    if sys.platform != "win32":
        return

    import io as _io

    # 检查 stdout 是否可用，避免重复包装
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, _io.TextIOWrapper):
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    # 检查 stderr 是否可用，避免重复包装
    if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, _io.TextIOWrapper):
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


if __name__ == "__main__":
    print("控制台编码修复工具模块")
    if sys.platform == "win32":
        print("当前平台: Windows")
    else:
        print(f"当前平台: {sys.platform}")
