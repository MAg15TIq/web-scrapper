"""
Error Recovery Agent with predictive failure detection and intelligent recovery strategies.
This agent monitors system health, predicts failures, and implements recovery actions.
"""
import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.langchain_base import EnhancedAgent
from models.langchain_models import (
    AgentConfig, AgentType, TaskRequest, TaskResponse, ErrorCorrection
)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(str, Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    RESTART = "restart"
    ALTERNATIVE_APPROACH = "alternative_approach"


class ErrorPattern(BaseModel):
    """Pattern for error detection and classification."""
    pattern_id: str
    error_type: str
    pattern_regex: str
    severity: ErrorSeverity
    frequency_threshold: int = Field(default=3, description="Occurrences before triggering")
    time_window_minutes: int = Field(default=60, description="Time window for frequency check")
    recovery_strategy: RecoveryStrategy
    description: str


class SystemHealthMetrics(BaseModel):
    """System health metrics for predictive analysis."""
    timestamp: datetime = Field(default_factory=datetime.now)
    cpu_usage: float = Field(ge=0.0, le=100.0)
    memory_usage: float = Field(ge=0.0, le=100.0)
    active_agents: int = Field(ge=0)
    pending_tasks: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)
    response_time_avg: float = Field(ge=0.0)
    success_rate: float = Field(ge=0.0, le=1.0)


class RecoveryAction(BaseModel):
    """Recovery action to be executed."""
    action_id: str = Field(default_factory=lambda: f"recovery-{int(time.time())}")
    strategy: RecoveryStrategy
    target_agent: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, description="1=highest, 5=lowest")
    estimated_duration: int = Field(default=30, description="Estimated duration in seconds")
    description: str


@tool
def analyze_error_patterns(error_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze historical errors to identify patterns and predict future failures."""
    if not error_history:
        return {"patterns": [], "predictions": []}
    
    # Group errors by type and time
    error_groups = defaultdict(list)
    time_series = defaultdict(list)
    
    for error in error_history:
        error_type = error.get("type", "unknown")
        timestamp = error.get("timestamp", datetime.now())
        
        error_groups[error_type].append(error)
        time_series[error_type].append(timestamp)
    
    patterns = []
    predictions = []
    
    # Analyze each error type
    for error_type, errors in error_groups.items():
        if len(errors) >= 3:  # Need at least 3 occurrences for pattern
            # Calculate frequency
            timestamps = [e.get("timestamp", datetime.now()) for e in errors]
            if timestamps:
                time_diffs = []
                for i in range(1, len(timestamps)):
                    if isinstance(timestamps[i], datetime) and isinstance(timestamps[i-1], datetime):
                        diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                        time_diffs.append(diff)
                
                if time_diffs:
                    avg_interval = sum(time_diffs) / len(time_diffs)
                    
                    patterns.append({
                        "error_type": error_type,
                        "frequency": len(errors),
                        "avg_interval_seconds": avg_interval,
                        "severity": "high" if len(errors) > 10 else "medium",
                        "trend": "increasing" if len(errors[-3:]) > len(errors[:3]) else "stable"
                    })
                    
                    # Predict next occurrence
                    if avg_interval > 0:
                        last_occurrence = max(timestamps)
                        if isinstance(last_occurrence, datetime):
                            next_predicted = last_occurrence + timedelta(seconds=avg_interval)
                            predictions.append({
                                "error_type": error_type,
                                "predicted_time": next_predicted.isoformat(),
                                "confidence": min(0.9, len(errors) / 20)  # Higher confidence with more data
                            })
    
    return {
        "patterns": patterns,
        "predictions": predictions,
        "total_errors": len(error_history),
        "unique_error_types": len(error_groups)
    }


@tool
def assess_system_health(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall system health and identify potential issues."""
    health_score = 100.0
    issues = []
    recommendations = []
    
    # CPU usage assessment
    cpu_usage = metrics.get("cpu_usage", 0)
    if cpu_usage > 90:
        health_score -= 20
        issues.append("Critical CPU usage")
        recommendations.append("Scale up compute resources or optimize agent workloads")
    elif cpu_usage > 70:
        health_score -= 10
        issues.append("High CPU usage")
        recommendations.append("Monitor CPU usage and consider scaling")
    
    # Memory usage assessment
    memory_usage = metrics.get("memory_usage", 0)
    if memory_usage > 90:
        health_score -= 20
        issues.append("Critical memory usage")
        recommendations.append("Increase memory allocation or optimize memory usage")
    elif memory_usage > 70:
        health_score -= 10
        issues.append("High memory usage")
        recommendations.append("Monitor memory usage trends")
    
    # Error rate assessment
    error_rate = metrics.get("error_rate", 0)
    if error_rate > 0.1:  # 10% error rate
        health_score -= 25
        issues.append("High error rate")
        recommendations.append("Investigate error causes and implement fixes")
    elif error_rate > 0.05:  # 5% error rate
        health_score -= 10
        issues.append("Elevated error rate")
        recommendations.append("Monitor error trends closely")
    
    # Response time assessment
    response_time = metrics.get("response_time_avg", 0)
    if response_time > 30:  # 30 seconds
        health_score -= 15
        issues.append("Slow response times")
        recommendations.append("Optimize agent performance or increase resources")
    elif response_time > 10:  # 10 seconds
        health_score -= 5
        issues.append("Elevated response times")
    
    # Success rate assessment
    success_rate = metrics.get("success_rate", 1.0)
    if success_rate < 0.8:  # 80% success rate
        health_score -= 20
        issues.append("Low success rate")
        recommendations.append("Review and improve agent reliability")
    elif success_rate < 0.9:  # 90% success rate
        health_score -= 10
        issues.append("Reduced success rate")
    
    # Determine overall health status
    if health_score >= 90:
        status = "excellent"
    elif health_score >= 75:
        status = "good"
    elif health_score >= 60:
        status = "fair"
    elif health_score >= 40:
        status = "poor"
    else:
        status = "critical"
    
    return {
        "health_score": max(0, health_score),
        "status": status,
        "issues": issues,
        "recommendations": recommendations,
        "assessment_time": datetime.now().isoformat()
    }


@tool
def generate_recovery_plan(error_info: Dict[str, Any], system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate a comprehensive recovery plan based on error analysis and system state."""
    recovery_actions = []
    
    error_type = error_info.get("type", "unknown")
    severity = error_info.get("severity", "medium")
    affected_agent = error_info.get("agent_id")
    
    # Basic retry strategy for transient errors
    if severity in ["low", "medium"]:
        recovery_actions.append({
            "strategy": "retry",
            "target_agent": affected_agent,
            "parameters": {
                "max_attempts": 3,
                "backoff_factor": 2,
                "initial_delay": 5
            },
            "priority": 1,
            "description": f"Retry failed operation for {error_type}"
        })
    
    # Fallback strategy for persistent errors
    if severity in ["medium", "high"]:
        recovery_actions.append({
            "strategy": "fallback",
            "target_agent": affected_agent,
            "parameters": {
                "fallback_method": "basic_mode",
                "disable_advanced_features": True
            },
            "priority": 2,
            "description": f"Switch to fallback mode for {error_type}"
        })
    
    # Escalation for critical errors
    if severity == "critical":
        recovery_actions.append({
            "strategy": "escalate",
            "parameters": {
                "notification_level": "immediate",
                "include_system_state": True,
                "auto_restart": True
            },
            "priority": 1,
            "description": f"Escalate critical error: {error_type}"
        })
    
    # Resource-based recovery
    cpu_usage = system_state.get("cpu_usage", 0)
    memory_usage = system_state.get("memory_usage", 0)
    
    if cpu_usage > 90 or memory_usage > 90:
        recovery_actions.append({
            "strategy": "restart",
            "target_agent": affected_agent,
            "parameters": {
                "graceful_shutdown": True,
                "clear_cache": True,
                "reduce_concurrency": True
            },
            "priority": 2,
            "description": "Restart agent to free resources"
        })
    
    # Alternative approach for repeated failures
    error_frequency = error_info.get("frequency", 1)
    if error_frequency > 5:
        recovery_actions.append({
            "strategy": "alternative_approach",
            "target_agent": affected_agent,
            "parameters": {
                "switch_strategy": True,
                "use_different_tools": True,
                "reduce_complexity": True
            },
            "priority": 3,
            "description": "Try alternative approach due to repeated failures"
        })
    
    return recovery_actions


@tool
def predict_failure_probability(metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Predict the probability of system failures based on historical metrics."""
    if len(metrics_history) < 5:
        return {"probability": 0.1, "confidence": 0.3, "factors": []}
    
    # Analyze trends in key metrics
    recent_metrics = metrics_history[-5:]  # Last 5 data points
    older_metrics = metrics_history[-10:-5] if len(metrics_history) >= 10 else []
    
    risk_factors = []
    failure_probability = 0.0
    
    # CPU usage trend
    recent_cpu = [m.get("cpu_usage", 0) for m in recent_metrics]
    if recent_cpu and max(recent_cpu) > 85:
        failure_probability += 0.2
        risk_factors.append("High CPU usage trend")
    
    # Memory usage trend
    recent_memory = [m.get("memory_usage", 0) for m in recent_metrics]
    if recent_memory and max(recent_memory) > 85:
        failure_probability += 0.2
        risk_factors.append("High memory usage trend")
    
    # Error rate trend
    recent_errors = [m.get("error_rate", 0) for m in recent_metrics]
    if recent_errors and max(recent_errors) > 0.1:
        failure_probability += 0.3
        risk_factors.append("Increasing error rate")
    
    # Response time trend
    recent_response_times = [m.get("response_time_avg", 0) for m in recent_metrics]
    if recent_response_times and max(recent_response_times) > 20:
        failure_probability += 0.15
        risk_factors.append("Degrading response times")
    
    # Success rate trend
    recent_success = [m.get("success_rate", 1.0) for m in recent_metrics]
    if recent_success and min(recent_success) < 0.85:
        failure_probability += 0.25
        risk_factors.append("Declining success rate")
    
    # Calculate confidence based on data availability
    confidence = min(0.9, len(metrics_history) / 20)
    
    return {
        "probability": min(1.0, failure_probability),
        "confidence": confidence,
        "risk_factors": risk_factors,
        "recommendation": "immediate_action" if failure_probability > 0.7 else "monitor_closely" if failure_probability > 0.4 else "normal_monitoring"
    }


class ErrorRecoveryAgent(EnhancedAgent):
    """
    Error Recovery Agent that monitors system health, predicts failures,
    and implements intelligent recovery strategies using AI-powered analysis.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the Error Recovery Agent.
        
        Args:
            config: Agent configuration
            llm: Language model for recovery planning
        """
        if config is None:
            config = AgentConfig(
                agent_id="error-recovery-agent",
                agent_type=AgentType.ERROR_RECOVERY,
                capabilities=[
                    "error_pattern_analysis",
                    "failure_prediction",
                    "recovery_planning",
                    "system_health_monitoring",
                    "automated_recovery"
                ]
            )
        
        if llm is None:
            try:
                llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.2,  # Low temperature for consistent recovery decisions
                    max_tokens=1500
                )
            except Exception as e:
                llm = None
                logging.warning(f"Could not initialize OpenAI LLM: {e}. Using rule-based recovery.")
        
        # Define recovery tools
        tools = [
            analyze_error_patterns,
            assess_system_health,
            generate_recovery_plan,
            predict_failure_probability
        ]
        
        # Create prompt template for recovery tasks
        prompt_template = PromptTemplate(
            input_variables=["input", "task_parameters"],
            template="""
You are an Error Recovery Agent responsible for maintaining system reliability and implementing recovery strategies.
Your role is to analyze errors, predict failures, and execute appropriate recovery actions.

Error/System Information: {input}
Recovery Parameters: {task_parameters}

Please analyze the situation and:
1. Identify error patterns and root causes
2. Assess system health and stability
3. Predict potential failure scenarios
4. Generate appropriate recovery strategies
5. Prioritize recovery actions based on impact and urgency

Use the available tools to perform comprehensive analysis and create actionable recovery plans.

Available tools:
- analyze_error_patterns: Identify patterns in error history
- assess_system_health: Evaluate overall system health
- generate_recovery_plan: Create specific recovery actions
- predict_failure_probability: Forecast potential failures

Focus on preventing cascading failures and maintaining system stability while minimizing downtime.
"""
        )
        
        super().__init__(
            config=config,
            llm=llm,
            tools=tools,
            prompt_template=prompt_template
        )
        
        # Error tracking and recovery state
        self.error_history: deque = deque(maxlen=1000)  # Keep last 1000 errors
        self.recovery_actions: List[RecoveryAction] = []
        self.system_metrics_history: deque = deque(maxlen=100)  # Keep last 100 metric snapshots
        
        # Load default error patterns
        self.error_patterns = self._load_default_error_patterns()
        
        self.logger.info("Error Recovery Agent initialized with predictive capabilities")
    
    def _load_default_error_patterns(self) -> List[ErrorPattern]:
        """Load default error patterns for common issues."""
        return [
            ErrorPattern(
                pattern_id="timeout_pattern",
                error_type="timeout",
                pattern_regex=r"timeout|timed out|connection timeout",
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.RETRY,
                description="Network or operation timeout errors"
            ),
            ErrorPattern(
                pattern_id="memory_pattern",
                error_type="memory_error",
                pattern_regex=r"out of memory|memory error|allocation failed",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.RESTART,
                description="Memory allocation and usage errors"
            ),
            ErrorPattern(
                pattern_id="rate_limit_pattern",
                error_type="rate_limit",
                pattern_regex=r"rate limit|too many requests|quota exceeded",
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                description="API rate limiting errors"
            ),
            ErrorPattern(
                pattern_id="authentication_pattern",
                error_type="auth_error",
                pattern_regex=r"authentication|unauthorized|invalid token",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.ESCALATE,
                description="Authentication and authorization errors"
            )
        ]
    
    async def handle_error(
        self,
        error_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an error and implement recovery strategy.
        
        Args:
            error_info: Information about the error
            context: Additional context for recovery
            
        Returns:
            Recovery result with actions taken
        """
        self.logger.info(f"Handling error: {error_info.get('type', 'unknown')}")
        
        # Add error to history
        error_record = {
            **error_info,
            "timestamp": datetime.now(),
            "context": context or {}
        }
        self.error_history.append(error_record)
        
        try:
            # Create error handling task
            recovery_task = TaskRequest(
                task_type="handle_error",
                parameters={
                    "error_info": error_info,
                    "context": context or {},
                    "error_history": list(self.error_history)[-20:]  # Last 20 errors
                }
            )
            
            # Use LangChain reasoning for recovery planning
            response = await self.execute_with_reasoning(recovery_task, context)
            
            if response.status == "completed" and response.result:
                return await self._execute_recovery_plan(error_info, response.result, context)
            else:
                # Fallback to rule-based recovery
                return await self._handle_error_basic(error_info, context)
                
        except Exception as e:
            self.logger.error(f"Error in error handling: {e}")
            # Fallback to basic recovery
            return await self._handle_error_basic(error_info, context)
    
    async def _execute_recovery_plan(
        self,
        error_info: Dict[str, Any],
        recovery_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the generated recovery plan."""
        
        # Generate recovery actions
        system_state = {
            "cpu_usage": 50,  # Mock values - would be real in production
            "memory_usage": 60,
            "error_rate": 0.05
        }
        
        recovery_actions = generate_recovery_plan(error_info, system_state)
        
        executed_actions = []
        for action in recovery_actions:
            try:
                # Simulate action execution
                action_result = await self._execute_recovery_action(action)
                executed_actions.append({
                    "action": action,
                    "result": action_result,
                    "status": "completed"
                })
            except Exception as e:
                executed_actions.append({
                    "action": action,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "recovery_plan": recovery_actions,
            "executed_actions": executed_actions,
            "total_actions": len(recovery_actions),
            "successful_actions": len([a for a in executed_actions if a["status"] == "completed"]),
            "recovery_time": datetime.now().isoformat()
        }
    
    async def _execute_recovery_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific recovery action."""
        strategy = action.get("strategy")
        
        # Simulate different recovery strategies
        if strategy == "retry":
            await asyncio.sleep(1)  # Simulate retry delay
            return {"action": "retry", "attempts": 1, "success": True}
        
        elif strategy == "fallback":
            return {"action": "fallback", "mode": "basic", "success": True}
        
        elif strategy == "restart":
            await asyncio.sleep(2)  # Simulate restart time
            return {"action": "restart", "downtime_seconds": 2, "success": True}
        
        elif strategy == "escalate":
            return {"action": "escalate", "notification_sent": True, "success": True}
        
        else:
            return {"action": strategy, "success": True}
    
    async def _handle_error_basic(
        self,
        error_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Basic error handling fallback."""
        self.logger.info("Using basic error handling fallback")
        
        error_type = error_info.get("type", "unknown")
        severity = error_info.get("severity", "medium")
        
        # Simple rule-based recovery
        if severity == "critical":
            action = "escalate"
        elif "timeout" in error_type.lower():
            action = "retry"
        elif "memory" in error_type.lower():
            action = "restart"
        else:
            action = "retry"
        
        return {
            "recovery_strategy": action,
            "reason": f"Rule-based recovery for {error_type}",
            "executed": True,
            "fallback_mode": True
        }
    
    async def execute_task_basic(self, task: TaskRequest) -> TaskResponse:
        """Basic task execution for error recovery operations."""
        if task.task_type == "handle_error":
            error_info = task.parameters.get("error_info", {})
            context = task.parameters.get("context", {})
            
            # Perform basic error handling
            result = await self._handle_error_basic(error_info, context)
            
            return TaskResponse(
                task_id=task.id,
                status="completed",
                result={"recovery": result},
                agent_id=self.agent_id
            )
        else:
            return TaskResponse(
                task_id=task.id,
                status="failed",
                error={"message": f"Unknown task type: {task.task_type}"},
                agent_id=self.agent_id
            )
