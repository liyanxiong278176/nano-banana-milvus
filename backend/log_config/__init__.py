"""
日志配置模块

功能：
- 配置日志输出到文件和控制台
- 按日期自动切分日志文件
- 支持不同级别的日志输出
"""
import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime


# 日志目录
LOG_DIR = Path(__file__).parent
LOG_DIR.mkdir(exist_ok=True)


# 日志格式
DETAILED_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
SIMPLE_FORMAT = "[%(levelname)s] %(message)s"

# 日期格式
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str = "app",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_to_file: 是否输出到文件
        log_to_console: 是否输出到控制台

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的处理器
    logger.handlers.clear()

    # 日志格式器
    formatter = logging.Formatter(DETAILED_FORMAT, datefmt=DATE_FORMAT)

    # 文件处理器（按天切分）
    if log_to_file:
        log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=30,  # 保留30天
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.suffix = "%Y%m%d"
        logger.addHandler(file_handler)

    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        # 控制台使用简单格式
        console_formatter = logging.Formatter(SIMPLE_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    # 如果已经配置过，直接返回
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # 否则使用默认配置
    return setup_logging(name)


# 预配置的日志记录器
api_logger = setup_logging("api", level=logging.INFO)
task_logger = setup_logging("task", level=logging.INFO)
limiter_logger = setup_logging("limiter", level=logging.INFO)
workflow_logger = setup_logging("workflow", level=logging.INFO)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """记录函数调用"""
    args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[函数调用] {func_name}({args_str})")


def log_function_result(logger: logging.Logger, func_name: str, result, duration: float = None):
    """记录函数返回结果"""
    duration_str = f", 耗时={duration:.3f}秒" if duration else ""
    logger.info(f"[函数返回] {func_name} → {result}{duration_str}")


def log_function_error(logger: logging.Logger, func_name: str, error: Exception):
    """记录函数异常"""
    logger.error(f"[函数异常] {func_name} → {type(error).__name__}: {error}", exc_info=True)
