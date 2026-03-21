"""
任务管理器 - 基于 OpenClaw 阶段4：生命周期管理

功能：
- 任务注册表
- 异步任务执行
- 状态查询
- 进度回调

企业考虑：
- 线程安全
- 异常隔离（单个任务失败不影响其他任务）
- 可观测性（日志、统计）
"""
import asyncio
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List

from .record import TaskRecord, TaskStatus, TaskPriority, TaskInput
from .limiter import TaskLimiter, get_limiter

# 使用配置好的日志系统
from log_config import get_logger
logger = get_logger("task")


class TaskManager:
    """
    任务管理器 - 核心控制器

    负责：
    1. 任务注册和跟踪
    2. 异步任务调度
    3. 状态管理
    4. 进度通知
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        output_base_dir: str = "output",
        enable_auto_cleanup: bool = True,
        cleanup_delay_seconds: int = 300
    ):
        """
        初始化任务管理器

        Args:
            max_concurrent: 最大并发任务数
            output_base_dir: 输出目录基础路径
            enable_auto_cleanup: 是否启用自动清理
            cleanup_delay_seconds: 清理延迟时间（秒）
        """
        self.max_concurrent = max_concurrent
        self.output_base_dir = Path(output_base_dir)
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_delay_seconds = cleanup_delay_seconds

        # 任务注册表 {task_id: TaskRecord}
        self.tasks: Dict[str, TaskRecord] = {}

        # 并发限制器
        self.limiter = get_limiter()

        # 进度回调注册表 {task_id: [callbacks]}
        self.progress_callbacks: Dict[str, List[Callable]] = {}

        # 工作流执行器（延迟注入）
        self.workflow_executor = None

        # 确保输出目录存在
        self.output_base_dir.mkdir(exist_ok=True, parents=True)

        logger.info(
            f"[任务管理器] 初始化完成 | 最大并发={max_concurrent}, 自动清理={enable_auto_cleanup}"
        )

    def set_workflow_executor(self, executor: Callable):
        """设置工作流执行器"""
        self.workflow_executor = executor
        logger.info("[任务管理器] 工作流执行器已设置")

    def register_task(
        self,
        file_bytes: bytes,
        file_name: str,
        category: str,
        style: str,
        season: str = "all_season",
        scene_hint: str = "",
        enable_quality_check: bool = False,
        judge_model: str = "",
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        注册新任务

        Returns:
            task_id: 任务ID

        Raises:
            ValueError: 参数不合法
        """
        # 创建任务记录
        import uuid
        task_id = f"task_{uuid.uuid4().hex[:10]}"

        task_input = TaskInput(
            category=category,
            style=style,
            season=season,
            scene_hint=scene_hint,
            enable_quality_check=enable_quality_check,
            judge_model=judge_model
        )

        record = TaskRecord(
            task_id=task_id,
            priority=priority,
            input=task_input,
            file_name=file_name,
            file_size=len(file_bytes)
        )

        # 创建任务专用输出目录
        task_output_dir = self.output_base_dir / task_id
        task_output_dir.mkdir(exist_ok=True, parents=True)
        record.output_dir = str(task_output_dir)

        # 保存原始文件
        original_path = task_output_dir / f"{task_id}_original{Path(file_name).suffix}"
        original_path.write_bytes(file_bytes)

        # 注册任务
        self.tasks[task_id] = record

        logger.info(
            f"[任务管理器] 任务已注册 | task_id={task_id}, 品类={category}, 风格={style}, "
            f"文件大小={len(file_bytes)}字节, 优先级={priority.value}"
        )
        return task_id

    async def submit_task(self, task_id: str) -> bool:
        """
        提交任务执行

        Args:
            task_id: 任务ID

        Returns:
            是否成功提交
        """
        record = self.tasks.get(task_id)
        if not record:
            logger.error(f"[任务管理器] 提交失败：任务不存在 | task_id={task_id}")
            return False

        if record.is_terminal():
            logger.warning(f"[任务管理器] 任务已处于终止状态 | task_id={task_id}, 状态={record.status.value}")
            return False

        # 获取并发许可
        logger.info(f"[任务管理器] 正在获取执行许可 | task_id={task_id}")
        acquired = await self.limiter.acquire(task_id)
        if not acquired:
            record.mark_failed("无法获取执行许可（并发限制）")
            logger.error(f"[任务管理器] 获取许可失败（达到并发限制）| task_id={task_id}")
            return False

        # 标记为排队
        record.mark_queued()
        logger.info(f"[任务管理器] 任务已排队 | task_id={task_id}, 状态={record.status.value}")

        # 启动异步执行
        asyncio.create_task(self._execute_task(task_id))
        return True

    async def _execute_task(self, task_id: str):
        """
        执行任务（内部方法）

        Args:
            task_id: 任务ID
        """
        record = self.tasks.get(task_id)
        if not record:
            logger.warning(f"[任务管理器] 执行失败：任务不存在 | task_id={task_id}")
            return

        try:
            # 标记为运行中
            record.mark_running()
            logger.info(f"[任务管理器] 任务开始执行 | task_id={task_id}, 状态={record.status.value}")
            self._notify_progress(task_id, "started", 0, "任务已开始")

            # 执行工作流
            if self.workflow_executor is None:
                raise ValueError("工作流执行器未设置")

            # 定义进度回调
            def progress_callback(step: str, percent: int, message: str = ""):
                record.update_progress(step, percent, message)
                logger.debug(f"[任务管理器] 进度更新 | task_id={task_id}, 步骤={step}, 进度={percent}%, 消息={message}")
                self._notify_progress(task_id, step, percent, message)

            # 执行工作流
            logger.info(f"[任务管理器] 正在执行工作流 | task_id={task_id}")
            result = await self.workflow_executor(
                task_id=task_id,
                record=record,
                progress_callback=progress_callback
            )

            # 标记完成
            record.mark_completed(result)
            self._notify_progress(task_id, "completed", 100, "任务已完成")

            logger.info(f"[任务管理器] 任务完成 | task_id={task_id}, 耗时={record.duration_seconds:.2f}秒")

        except asyncio.TimeoutError:
            record.mark_failed("任务执行超时")
            logger.error(f"[任务管理器] 任务超时 | task_id={task_id}")

        except Exception as e:
            record.mark_failed(str(e))
            logger.error(f"[任务管理器] 任务失败 | task_id={task_id}, 错误={e}", exc_info=True)

        finally:
            # 释放并发许可
            self.limiter.release(task_id)
            logger.debug(f"[任务管理器] 已释放执行许可 | task_id={task_id}")

            # 启动延迟清理
            if self.enable_auto_cleanup:
                asyncio.create_task(self._delayed_cleanup(task_id))

    def _notify_progress(self, task_id: str, step: str, percent: int, message: str):
        """通知进度更新"""
        callbacks = self.progress_callbacks.get(task_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(task_id, step, percent, message))
                else:
                    callback(task_id, step, percent, message)
            except Exception as e:
                logger.error(f"[任务管理器] 进度回调错误 | task_id={task_id}, 错误={e}")

    async def _delayed_cleanup(self, task_id: str):
        """延迟清理任务资源"""
        await asyncio.sleep(self.cleanup_delay_seconds)

        record = self.tasks.get(task_id)
        if record and record.output_dir:
            try:
                # 只清理临时文件，保留任务记录
                output_path = Path(record.output_dir)
                if output_path.exists():
                    # 保留最终结果，清理中间文件
                    for file in output_path.glob("*"):
                        if "generated" in file.name or "original" in file.name:
                            # 保留重要文件
                            continue
                        file.unlink(missing_ok=True)

                    # 如果目录为空，删除目录
                    try:
                        output_path.rmdir()
                    except:
                        pass

                logger.info(f"[任务管理器] 资源清理完成 | task_id={task_id}")
            except Exception as e:
                logger.error(f"[任务管理器] 资源清理错误 | task_id={task_id}, 错误={e}")

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """获取任务记录"""
        return self.tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态（用于API响应）"""
        record = self.tasks.get(task_id)
        if not record:
            logger.debug(f"[任务管理器] 查询状态：任务不存在 | task_id={task_id}")
            return None
        return record.to_dict()

    def cancel_task(self, task_id: str, reason: str = "") -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID
            reason: 取消原因

        Returns:
            是否成功取消
        """
        record = self.tasks.get(task_id)
        if not record:
            logger.warning(f"[任务管理器] 取消失败：任务不存在 | task_id={task_id}")
            return False

        if record.is_terminal():
            logger.warning(f"[任务管理器] 取消失败：任务已终止 | task_id={task_id}")
            return False

        record.mark_cancelled(reason)
        logger.info(f"[任务管理器] 任务已取消 | task_id={task_id}, 原因={reason}")
        return True

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 50
    ) -> List[dict]:
        """
        列出任务

        Args:
            status: 状态过滤
            limit: 最大返回数量

        Returns:
            任务列表
        """
        tasks = list(self.tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        # 按创建时间倒序
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        logger.debug(f"[任务管理器] 列出任务 | 过滤状态={status.value if status else '全部'}, 返回数量={min(limit, len(tasks))}")
        return [t.to_dict() for t in tasks[:limit]]

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = len(self.tasks)
        by_status = {}
        for status in TaskStatus:
            by_status[status.value] = sum(1 for t in self.tasks.values() if t.status == status)

        # 计算平均耗时
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        avg_duration = 0
        if completed_tasks:
            avg_duration = sum(t.duration_seconds for t in completed_tasks) / len(completed_tasks)

        stats = {
            "total_tasks": total,
            "by_status": by_status,
            "active_tasks": sum(1 for t in self.tasks.values() if t.is_active()),
            "avg_duration_seconds": avg_duration,
            "limiter_stats": self.limiter.get_stats()
        }

        logger.debug(f"[任务管理器] 统计信息 | 总任务={total}, 活跃任务={stats['active_tasks']}, 平均耗时={avg_duration:.2f}秒")
        return stats

    def register_progress_callback(self, task_id: str, callback: Callable):
        """注册进度回调"""
        if task_id not in self.progress_callbacks:
            self.progress_callbacks[task_id] = []
        self.progress_callbacks[task_id].append(callback)
        logger.debug(f"[任务管理器] 注册进度回调 | task_id={task_id}")

    def unregister_progress_callback(self, task_id: str, callback: Callable):
        """注销进度回调"""
        callbacks = self.progress_callbacks.get(task_id, [])
        if callback in callbacks:
            callbacks.remove(callback)
            logger.debug(f"[任务管理器] 注销进度回调 | task_id={task_id}")

    def cleanup_old_tasks(self, max_age_days: int = 7, keep_count: int = 100):
        """
        清理旧任务记录

        Args:
            max_age_days: 保留最近N天的任务
            keep_count: 至少保留N个任务
        """
        import time
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        # 按更新时间排序
        sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.updated_at.timestamp(), reverse=True)

        # 始终保留最新的 keep_count 个任务
        to_remove = []
        for i, task in enumerate(sorted_tasks):
            if i >= keep_count and task.updated_at.timestamp() < cutoff_time:
                to_remove.append(task.task_id)

        for task_id in to_remove:
            del self.tasks[task_id]

        if to_remove:
            logger.info(f"[任务管理器] 清理旧任务 | 清理数量={len(to_remove)}")


# 全局单例
_default_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """获取全局任务管理器单例"""
    global _default_manager
    if _default_manager is None:
        _default_manager = TaskManager()
        logger.info("[任务管理器] 全局单例已创建")
    return _default_manager


def reset_task_manager():
    """重置全局任务管理器（主要用于测试）"""
    global _default_manager
    _default_manager = None
    logger.info("[任务管理器] 全局单例已重置")
