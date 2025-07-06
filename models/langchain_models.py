"""
Enhanced Pydantic models for LangChain integration.
These models provide structured data validation and type safety for the enhanced agent system.
"""
import uuid
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import BaseModel, Field, validator, ConfigDict


class AgentType(str, Enum):
    """Enhanced agent types for the LangChain-integrated system."""
    # Core agents
    COORDINATOR = "coordinator"
    SCRAPER = "scraper"
    PARSER = "parser"
    STORAGE = "storage"
    JAVASCRIPT = "javascript"
    
    # Intelligence agents
    MASTER_INTELLIGENCE = "master_intelligence"
    NLU_AGENT = "nlu_agent"
    PLANNING_AGENT = "planning_agent"
    
    # Specialized agents
    AUTHENTICATION = "authentication"
    ANTI_DETECTION = "anti_detection"
    DATA_TRANSFORMATION = "data_transformation"
    DATA_VALIDATION = "data_validation"
    VISUALIZATION = "visualization"
    
    # Processing agents
    NLP_PROCESSING = "nlp_processing"
    IMAGE_PROCESSING = "image_processing"
    DOCUMENT_PROCESSING = "document_processing"
    
    # Quality & monitoring
    QUALITY_ASSURANCE = "quality_assurance"
    MONITORING = "monitoring"
    ERROR_RECOVERY = "error_recovery"


class ActionType(str, Enum):
    """Types of actions that can be requested."""
    SCRAPE = "scrape"
    MONITOR = "monitor"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    ANALYZE = "analyze"
    VALIDATE = "validate"
    VISUALIZE = "visualize"
    CONFIGURE = "configure"
    EXPORT = "export"
    SCHEDULE = "schedule"


class OutputFormat(str, Enum):
    """Supported output formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    HTML = "html"
    XML = "xml"


class Priority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class ScrapingRequest(BaseModel):
    """Structured request for web scraping operations."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    product_query: str = Field(..., description="Natural language description of what to scrape")
    target_sites: List[str] = Field(..., description="List of target websites or URLs")
    data_points: List[str] = Field(..., description="Specific data fields to extract")
    action: ActionType = Field(default=ActionType.SCRAPE)
    timeframe: Optional[str] = Field(None, description="Time constraints for the operation")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    output_format: OutputFormat = Field(default=OutputFormat.JSON)
    priority: Priority = Field(default=Priority.NORMAL)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('target_sites')
    def validate_sites(cls, v):
        if not v:
            raise ValueError("At least one target site must be specified")
        return v
    
    @validator('data_points')
    def validate_data_points(cls, v):
        if not v:
            raise ValueError("At least one data point must be specified")
        return v


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    model_config = ConfigDict(extra='allow')
    
    agent_id: str
    agent_type: AgentType
    capabilities: List[str] = Field(default_factory=list)
    max_concurrent_tasks: int = Field(default=5)
    timeout_seconds: int = Field(default=300)
    retry_attempts: int = Field(default=3)
    memory_limit_mb: int = Field(default=512)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    """Enhanced task request with LangChain integration."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    priority: Priority = Field(default=Priority.NORMAL)
    timeout: int = Field(default=300)
    retry_policy: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class TaskResponse(BaseModel):
    """Enhanced task response with detailed metadata."""
    model_config = ConfigDict(extra='allow')
    
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    execution_time: float = Field(default=0.0)
    agent_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=datetime.now)


class AgentPerformanceMetrics(BaseModel):
    """Performance metrics for agent monitoring."""
    agent_id: str
    agent_type: AgentType
    tasks_completed: int = Field(default=0)
    tasks_failed: int = Field(default=0)
    average_execution_time: float = Field(default=0.0)
    memory_usage_mb: float = Field(default=0.0)
    cpu_usage_percent: float = Field(default=0.0)
    success_rate: float = Field(default=0.0)
    last_activity: datetime = Field(default_factory=datetime.now)
    
    @validator('success_rate')
    def calculate_success_rate(cls, v, values):
        completed = values.get('tasks_completed', 0)
        failed = values.get('tasks_failed', 0)
        total = completed + failed
        return (completed / total * 100) if total > 0 else 0.0


class WorkflowState(BaseModel):
    """State management for complex workflows."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    current_step: str
    completed_steps: List[str] = Field(default_factory=list)
    pending_steps: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def mark_step_completed(self, step: str):
        """Mark a step as completed and update state."""
        if step in self.pending_steps:
            self.pending_steps.remove(step)
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        self.updated_at = datetime.now()


class ScrapeStrategy(BaseModel):
    """Strategy configuration for scraping operations."""
    site_url: str
    strategy_type: str = Field(default="standard")
    use_javascript: bool = Field(default=False)
    use_proxy: bool = Field(default=False)
    rate_limit_delay: float = Field(default=1.0)
    user_agent: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    cookies: Dict[str, str] = Field(default_factory=dict)
    selectors: Dict[str, str] = Field(default_factory=dict)
    pagination_config: Optional[Dict[str, Any]] = None
    anti_detection_config: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """Detailed execution plan for complex operations."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    strategies: List[ScrapeStrategy]
    estimated_duration: int = Field(default=0)  # seconds
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    fallback_plans: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class DataValidationRule(BaseModel):
    """Rules for data validation."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    field_name: str
    rule_type: str  # required, format, range, custom
    parameters: Dict[str, Any] = Field(default_factory=dict)
    error_message: str
    severity: str = Field(default="error")  # error, warning, info


class QualityReport(BaseModel):
    """Data quality assessment report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data_source: str
    total_records: int
    valid_records: int
    invalid_records: int
    quality_score: float = Field(ge=0.0, le=100.0)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


class ErrorCorrection(BaseModel):
    """Error correction suggestions."""
    error_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    error_type: str
    error_message: str
    suggested_fix: str
    confidence: float = Field(ge=0.0, le=1.0)
    auto_fixable: bool = Field(default=False)
    fix_parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Enhanced message format for agent communication."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: AgentType
    recipient: AgentType
    message_type: str
    payload: Union[TaskRequest, TaskResponse, Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    priority: Priority = Field(default=Priority.NORMAL)
    context: Dict[str, Any] = Field(default_factory=dict)
