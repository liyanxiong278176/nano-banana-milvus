"""
任务记录模型 - 基于 OpenClaw 阶段4：生命周期管理

功能：
- 任务状态定义
- 任务元数据存储
- ��度跟踪

企业考虑：
- 可序列化（支持持久化）
- 完整的审计信息
- 时间戳记录
"""
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import uuid


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"       # 等待执行
    QUEUED = "queued"         # 排队中
    RUNNING = "running"       # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消
    TIMEOUT = "timeout"       # 超时

    @classmethod
    def is_terminal(cls, status: 'TaskStatus') -> bool:
        """是否是终止状态"""
        return status in {cls.COMPLETED, cls.FAILED, cls.CANCELLED, cls.TIMEOUT}

    @classmethod
    def is_active(cls, status: 'TaskStatus') -> bool:
        """是否是活跃状态"""
        return status in {cls.PENDING, cls.QUEUED, cls.RUNNING}


class TaskPriority(str, Enum):
    """任务优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TaskInput:
    """任务输入参数"""
    category: str
    style: str
    season: str = "all_season"
    scene_hint: str = ""
    enable_quality_check: bool = False
    judge_model: str = ""

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class TaskProgress:
    """任务进度信息"""
    current_step: str = ""
    progress_percent: int = 0
    message: str = ""
    timestamp: float = 0

    def update(self, step: str, percent: int, msg: str = ""):
        """更新进度"""
        self.current_step = step
        self.progress_percent = max(0, min(100, percent))
        self.message = msg
        self.timestamp = datetime.now().timestamp()


@dataclass
class TaskRecord:
    """
    任务记录 - 对应 OpenClaw 的 SubagentRunRecord

    存储任务的完整生命周期信息
    """
    # 基本信息
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 输入参数
    input: Optional[TaskInput] = None
    file_name: str = ""
    file_size: int = 0

    # 执行信息
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0

    # 进度跟踪
    progress: TaskProgress = field(default_factory=TaskProgress)

    # 结果信息
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # 资源信息
    output_dir: Optional[str] = None
    output_files: list = field(default_factory=list)

    # 统计信息
    api_calls_count: int = 0
    tokens_used: int = 0
    estimated_cost_usd: float = 0

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.task_id:
            self.task_id = uuid.uuid4().hex[:12]

    def mark_queued(self):
        """标记为排队"""
        self.status = TaskStatus.QUEUED
        self.updated_at = datetime.now()

    def mark_running(self):
        """标记为运行中"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self.updated_at = datetime.now()

    def mark_completed(self, result: Optional[Dict[str, Any]] = None):
        """标记为已完成"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        if result:
            self.result = result
        self.progress.update("completed", 100, "Task completed")

    def mark_failed(self, error: str, details: Optional[Dict] = None):
        """标记为失败"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.error = error
        self.error_details = details

    def mark_cancelled(self, reason: str = ""):
        """标记为已取消"""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        self.error = f"Cancelled: {reason}" if reason else "Cancelled"

    def update_progress(self, step: str, percent: int, message: str = ""):
        """更新进度"""
        self.progress.update(step, percent, message)
        self.updated_at = datetime.now()

    def add_output_file(self, file_path: str):
        """添加输出文件"""
        self.output_files.append(file_path)

    def is_terminal(self) -> bool:
        """是否已终止"""
        return TaskStatus.is_terminal(self.status)

    def is_active(self) -> bool:
        """是否活跃"""
        return TaskStatus.is_active(self.status)

    def to_dict(self) -> dict:
        """转换为字典（用于API响应）"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "progress": {
                "step": self.progress.current_step,
                "percent": self.progress.progress_percent,
                "message": self.progress.message
            },
            "input": self.input.to_dict() if self.input else None,
            "file_name": self.file_name,
            "result": self.result,
            "error": self.error,
            "output_files": self.output_files,
            "stats": {
                "api_calls_count": self.api_calls_count,
                "tokens_used": self.tokens_used,
                "estimated_cost_usd": self.estimated_cost_usd
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskRecord':
        """从字典创建（用于从持久化存储恢复）"""
        # 简化版本，实际可能需要更复杂的处理
        return cls(
            task_id=data.get("task_id", ""),
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", "normal")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            result=data.get("result"),
            error=data.get("error")
        )
