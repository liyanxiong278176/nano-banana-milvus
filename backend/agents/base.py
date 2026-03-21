"""
Agent基类定义 - BaseAgent

所有具体Agent的父类，定义统一的接口和公共方法。

【v2.2 更新】集成提示词工程：
- 提示词版本管理
- 提示词执行追踪
- 多语言支持

【面试讲解要点】
1. 模板方法模式：run()是抽象方法，子类实现具体逻辑
2. _add_evidence和_log_metric是通用能力，所有子类复用
3. time_decorator自动记录每个Agent的执行时间
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from functools import wraps
import time
from .state import PipelineState

# ==================== 【v2.2新增】提示词工程导入 ====================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from prompts.prompts import PromptVersionManager
    from prompts.v2 import get_metrics, record_prompt_execution
    from prompts.v3 import PromptLocale
    PROMPTS_ENGINE_AVAILABLE = True
except ImportError:
    print("[BaseAgent] 提示词工程模块不可用，使用基础功能")
    PROMPTS_ENGINE_AVAILABLE = False


def time_decorator(metric_key: str):
    """
    装饰器：自动记录函数执行时间到metrics

    Args:
        metric_key: 记录到metrics中的键名（如"upload_time"）

    使用示例：
        @time_decorator("upload_time")
        def run(self, state: PipelineState) -> PipelineState:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, state: PipelineState) -> PipelineState:
            start_time = time.time()
            result = func(self, state)
            elapsed = time.time() - start_time
            # 自动记录耗时
            result["metrics"][metric_key] = round(elapsed, 3)
            return result
        return wrapper
    return decorator


class BaseAgent(ABC):
    """
    Agent抽象基类

    所有具体Agent都必须继承此类并实现run()方法。

    【设计原则】
    1. 每个Agent只负责一个明确的任务（单一职责）
    2. Agent之间通过PipelineState解耦，不直接依赖
    3. 所有Agent都支持证据链追踪和指标埋点
    4. 执行时间自动记录（通过time_decorator）

    【继承示例】
        class UploadAgent(BaseAgent):
            def __init__(self, ...):
                super().__init__("UploadAgent")
                # 初始化特定组件

            @time_decorator("upload_time")
            def run(self, state: PipelineState) -> PipelineState:
                # 实现具体逻辑
                state["product_id"] = "NEW_123"
                self._add_evidence(state, "生成商品ID: NEW_123")
                return state
    """

    def __init__(self, name: str):
        """
        初始化Agent

        Args:
            name: Agent名称（用于日志和证据链）
        """
        self.name = name
        self._step_count = 0  # 内部计数器，用于证据链编号

        # 【v2.2新增】提示词版本追踪
        self.prompt_version = "2.0"  # 默认提示词版本
        self.prompt_locale = "zh"   # 默认语言

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """
        Agent的主执行方法（抽象方法，子类必须实现）

        Args:
            state: 当前工作流状态

        Returns:
            更新后的状态
        """
        pass

    def _add_evidence(self, state: PipelineState, message: str):
        """
        向证据链追加一条记录

        证据链用途：
        1. 调试：追踪每一步的决策依据
        2. 面试：展示可解释性，讲清楚"为什么这样设计"
        3. 监控：线上出问题时快速定位

        Args:
            state: 当前状态
            message: 证据消息（建议格式："步骤名称: 具体内容"）

        Example:
            self._add_evidence(state, "检索完成: 返回5个结果，Top1为SKU001")
        """
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        evidence = f"[{timestamp}] [{self.name}] {message}"
        state["evidence_chain"].append(evidence)

    def _log_metric(self, state: PipelineState, key: str, value: float):
        """
        记录业务指标到metrics

        指标用途：
        1. 性能分析：找出耗时瓶颈
        2. 业务监控：跟踪关键业务指标（如检索数量、评分）
        3. 面试展示：展示数据驱动优化能力

        Args:
            state: 当前状态
            key: 指标名称（如"result_count", "best_score"）
            value: 指标值（数值类型）

        Example:
            self._log_metric(state, "result_count", 5)
            self._log_metric(state, "best_score", 4.2)
        """
        state["metrics"][key] = value

    def _update_status(self, state: PipelineState, status: str, step: str = ""):
        """
        更新任务状态和当前步骤

        Args:
            state: 当前状态
            status: 新状态（pending/processing/completed/failed）
            step: 当前步骤名称（用于兜底Agent判断）
        """
        state["status"] = status
        if step:
            state["current_step"] = step
            self._add_evidence(state, f"进入步骤: {step}")

    def _handle_error(self, state: PipelineState, error_msg: str) -> PipelineState:
        """
        标准错误处理

        Args:
            state: 当前状态
            error_msg: 错误消息

        Returns:
            更新后的状态（标记为failed）
        """
        state["status"] = "failed"
        state["error_msg"] = error_msg
        state["current_step"] = self.name
        self._add_evidence(state, f"错误: {error_msg}")

        # 【v2.2新增】记录提示词执行失败
        self._record_prompt_execution(state, success=False, error=error_msg)

        return state

    # ==================== 【v2.2新增】提示词工程方法 ====================

    def set_prompt_version(self, version: str):
        """
        设置提示词版本

        Args:
            version: 版本号 (如 "2.0", "2.1")
        """
        self.prompt_version = version
        if PROMPTS_ENGINE_AVAILABLE:
            versions = PromptVersionManager.get_all_versions()
            if self.name.lower() in ["qualityjudgeagent", "quality_judge"]:
                pt = "quality_judge"
            elif self.name.lower() in ["styleanalysisagent", "style_analysis"]:
                pt = "style_analysis"
            elif self.name.lower() in ["hybridretrievalagent", "hybrid_retrieval"]:
                pt = "retrieval_judge"
            else:
                pt = None

            if pt and pt in versions:
                self.prompt_version = versions[pt]["current"]

    def set_prompt_locale(self, locale: str):
        """
        设置提示词语言

        Args:
            locale: 语言代码 ("zh" 或 "en")
        """
        if PROMPTS_ENGINE_AVAILABLE:
            if locale in PromptLocale.get_supported_locales():
                self.prompt_locale = locale

    def get_prompt(self, prompt_type: str, key: str = "system") -> str:
        """
        获取本地化提示词

        Args:
            prompt_type: 提示词类型
            key: 提示词部分

        Returns:
            本地化的提示词文本
        """
        if PROMPTS_ENGINE_AVAILABLE:
            return PromptLocale.get_prompt(prompt_type, self.prompt_locale, key)
        return ""

    def _record_prompt_execution(
        self,
        state: PipelineState,
        success: bool,
        execution_time: float = 0,
        error: str = "",
        metadata: Dict[str, Any] = None
    ):
        """
        记录提示词执行（内部方法）

        Args:
            state: 当前状态
            success: 是否成功
            execution_time: 执行时间
            error: 错误信息
            metadata: 额外元数据
        """
        if not PROMPTS_ENGINE_AVAILABLE:
            return

        try:
            # 确定提示词类型
            if self.name.lower() in ["qualityjudgeagent", "quality_judge"]:
                pt = "quality_judge"
            elif self.name.lower() in ["styleanalysisagent", "style_analysis"]:
                pt = "style_analysis"
            elif self.name.lower() in ["hybridretrievalagent", "hybrid_retrieval"]:
                pt = "retrieval_judge"
            elif self.name.lower() in ["imagegenagent", "image_gen"]:
                pt = "image_generation"
            else:
                return  # 其他类型暂不追踪

            # 获取模型信息
            model = state.get("model", "qwen/qwen3-vl-8b-instruct")

            # 记录执行
            record_prompt_execution(
                prompt_type=pt,
                prompt_version=self.prompt_version,
                model=model,
                input_summary=f"{self.name} on task {state.get('task_id', 'unknown')[:8]}",
                output_summary=f"success={success}",
                execution_time=execution_time,
                success=success,
                error_message=error,
                metadata=metadata or {}
            )

        except Exception as e:
            # 记录失败不影响主流程
            print(f"[{self.name}] 记录提示词执行失败: {e}")

    def _log_prompt_version(self, state: PipelineState):
        """
        记录提示词版本到证据链

        Args:
            state: 当前状态
        """
        if PROMPTS_ENGINE_AVAILABLE:
            self._add_evidence(
                state,
                f"提示词版本: {self.prompt_version} | 语言: {self.prompt_locale}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', prompt_v='{self.prompt_version}')"


if __name__ == "__main__":
    # 测试基类
    from .state import create_initial_state

    class TestAgent(BaseAgent):
        @time_decorator("test_time")
        def run(self, state: PipelineState) -> PipelineState:
            self._update_status(state, "processing", "TestAgent")
            self._add_evidence(state, "测试证据: 执行成功")
            self._log_metric(state, "test_metric", 42.0)
            return state

    # 创建测试状态
    state = create_initial_state(
        task_id="test_001",
        file_bytes=b"test",
        category="midi_dress",
        style="elegant"
    )

    # 执行测试Agent
    agent = TestAgent("TestAgent")
    result = agent.run(state)

    print("\n测试Agent执行结果：")
    print(f"  status: {result['status']}")
    print(f"  current_step: {result['current_step']}")
    print(f"  metrics: {result['metrics']}")
    print(f"  evidence_chain:")
    for evidence in result['evidence_chain']:
        print(f"    - {evidence}")
