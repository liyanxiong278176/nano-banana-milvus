"""
任务并发限制器 - 基于 OpenClaw 阶段1：接单校验之并发配额

功能：
- 全局并发控制
- 任务队列管理
- 并发状态统计

企业考虑：
- 可配置的并发限制
- 并发状态可监控
- 拒绝策略（拒绝/排队）
"""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum
from collections import deque

# 使用配置好的日志系统
from log_config import get_logger
logger = get_logger("limiter")


class RejectionPolicy(str, Enum):
    """拒绝策略"""
    REJECT = "reject"      # 直接拒绝新请求
    QUEUE = "queue"        # 排队等待


@dataclass
class ConcurrencyStats:
    """并发统计信息"""
    total_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0
    queued_requests: int = 0
    current_running: int = 0
    peak_concurrent: int = 0
    total_wait_time_seconds: float = 0


class TaskLimiter:
    """
    任务并发限制器

    使用 Semaphore 实现并发控制，支持拒绝策略和统计
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        max_queue_size: int = 10,
        rejection_policy: RejectionPolicy = RejectionPolicy.REJECT,
        acquire_timeout: Optional[float] = None
    ):
        """
        初始化并发限制器

        Args:
            max_concurrent: 最大并发任务数
            max_queue_size: 最大队列长度（0表示不排队）
            rejection_policy: 拒绝策略（reject=拒绝, queue=排队）
            acquire_timeout: 获取许可的超时时间（秒），None表示无限等待
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.rejection_policy = rejection_policy
        self.acquire_timeout = acquire_timeout

        # 并发信号量
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 统计信息
        self.stats = ConcurrencyStats()

        # 等待队列（用于排队策略）
        self.wait_queue: deque = deque()

        # 当前活跃任务跟踪
        self.active_tasks: Dict[str, float] = {}  # task_id -> start_time

        logger.info(
            f"[并发限制器] 初始化完成 | 最大并发={max_concurrent}, "
            f"队列大小={max_queue_size}, 拒绝策略={rejection_policy.value}"
        )

    async def acquire(self, task_id: str) -> bool:
        """
        获取执行许可

        Args:
            task_id: 任务ID

        Returns:
            是否获取成功
        """
        self.stats.total_requests += 1
        start_time = time.time()

        # 检查当前并发数
        current_running = self.stats.current_running

        if current_running >= self.max_concurrent:
            # 达到并发上限
            if self.rejection_policy == RejectionPolicy.REJECT:
                self.stats.rejected_requests += 1
                logger.warning(
                    f"[并发限制器] 任务被拒绝（达到并发上限）| task_id={task_id}, "
                    f"当前并发={current_running}/{self.max_concurrent}"
                )
                return False
            elif self.max_queue_size == 0:
                # 不支持排队
                self.stats.rejected_requests += 1
                logger.warning(
                    f"[并发限制器] 任务被拒绝（无队列）| task_id={task_id}"
                )
                return False
            elif len(self.wait_queue) >= self.max_queue_size:
                # 队列已满
                self.stats.rejected_requests += 1
                logger.warning(
                    f"[并发限制器] 任务被拒绝（队列已满）| task_id={task_id}, "
                    f"队列长度={len(self.wait_queue)}/{self.max_queue_size}"
                )
                return False

        # 加入等待队列（用于统计）
        self.wait_queue.append(task_id)
        self.stats.queued_requests = len(self.wait_queue)

        logger.info(f"[并发限制器] 任务进入等待队列 | task_id={task_id}, 队列位置={len(self.wait_queue)}")

        try:
            # 尝试获取许可
            if self.acquire_timeout is not None:
                acquired = await asyncio.wait_for(
                    self.semaphore.acquire(),
                    timeout=self.acquire_timeout
                )
                if not acquired:
                    raise asyncio.TimeoutError()
            else:
                await self.semaphore.acquire()

            # 获取成功
            wait_time = time.time() - start_time
            self.stats.total_wait_time_seconds += wait_time
            self.stats.accepted_requests += 1
            self.stats.current_running += 1

            # 更新峰值
            if self.stats.current_running > self.stats.peak_concurrent:
                self.stats.peak_concurrent = self.stats.current_running

            # 记录活跃任务
            self.active_tasks[task_id] = time.time()

            # 从等待队列移除
            try:
                self.wait_queue.remove(task_id)
            except ValueError:
                pass

            logger.info(
                f"[并发限制器] 获取许可成功 | task_id={task_id}, "
                f"等待时间={wait_time:.2f}秒, "
                f"当前并发={self.stats.current_running}/{self.max_concurrent}"
            )
            return True

        except asyncio.TimeoutError:
            # 超时
            self.stats.rejected_requests += 1
            try:
                self.wait_queue.remove(task_id)
            except ValueError:
                pass
            logger.warning(f"[并发限制器] 获取许可超时 | task_id={task_id}")
            return False

        except Exception as e:
            # 其他异常
            self.stats.rejected_requests += 1
            try:
                self.wait_queue.remove(task_id)
            except ValueError:
                pass
            logger.error(f"[并发限制器] 获取许可异常 | task_id={task_id}, 错误={e}")
            return False

    def release(self, task_id: str):
        """
        释放执行许可

        Args:
            task_id: 任务ID
        """
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]

        self.semaphore.release()
        self.stats.current_running -= 1

        logger.info(
            f"[并发限制器] 释放许可 | task_id={task_id}, "
            f"当前并发={self.stats.current_running}/{self.max_concurrent}"
        )

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        accept_rate = (
            self.stats.accepted_requests / self.stats.total_requests
            if self.stats.total_requests > 0 else 1.0
        )
        avg_wait = (
            self.stats.total_wait_time_seconds / self.stats.accepted_requests
            if self.stats.accepted_requests > 0 else 0
        )

        stats = {
            "max_concurrent": self.max_concurrent,
            "current_running": self.stats.current_running,
            "peak_concurrent": self.stats.peak_concurrent,
            "total_requests": self.stats.total_requests,
            "accepted_requests": self.stats.accepted_requests,
            "rejected_requests": self.stats.rejected_requests,
            "queued_requests": len(self.wait_queue),
            "accept_rate": accept_rate,
            "avg_wait_time_seconds": avg_wait
        }

        logger.debug(
            f"[并发限制器] 统计信息 | 总请求={self.stats.total_requests}, "
            f"接受={self.stats.accepted_requests}, 拒绝={self.stats.rejected_requests}, "
            f"接受率={accept_rate:.1%}, 平均等待={avg_wait:.2f}秒"
        )

        return stats

    def get_active_tasks(self) -> list:
        """
        获取当前活跃任务列表

        Returns:
            任务ID列表及其运行时长
        """
        now = time.time()
        active_list = [
            {
                "task_id": task_id,
                "running_time_seconds": now - start_time
            }
            for task_id, start_time in self.active_tasks.items()
        ]
        logger.debug(f"[并发限制器] 活跃任务 | 数量={len(active_list)}")
        return active_list

    def set_max_concurrent(self, new_max: int):
        """
        动态调整最大并发数

        Args:
            new_max: 新的最大并发数
        """
        old_max = self.max_concurrent
        self.max_concurrent = new_max
        logger.info(f"[并发限制器] 最大并发数调整 | {old_max} → {new_max}")

        # 注意：已经创建的 Semaphore 无法动态调整
        # 如果需要动态调整，需要重新创建 Semaphore（更复杂）


class TaskPermit:
    """
    任务许可上下文管理器

    使用方式：
        async with limiter.permit(task_id) as acquired:
            if acquired:
                # 执行任务
                pass
    """

    def __init__(self, limiter: TaskLimiter, task_id: str):
        self.limiter = limiter
        self.task_id = task_id
        self.acquired = False

    async def __aenter__(self) -> bool:
        self.acquired = await self.limiter.acquire(self.task_id)
        return self.acquired

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.limiter.release(self.task_id)


# 全局单例
_default_limiter: Optional[TaskLimiter] = None


def get_limiter() -> TaskLimiter:
    """获取全局并发限制器单例"""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = TaskLimiter()
        logger.info("[并发限制器] 全局单例已创建")
    return _default_limiter


def reset_limiter():
    """重置全局并发限制器（主要用于测试）"""
    global _default_limiter
    _default_limiter = None
    logger.info("[并发限制器] 全局单例已重置")
