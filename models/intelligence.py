"""
Intelligence models for the self-aware web scraping system.
"""
import time
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Union
from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Types of content that can be processed by the system."""
    HTML = "html"
    JSON = "json"
    XML = "xml"
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    XLS = "xls"
    XLSX = "xlsx"
    CSV = "csv"
    TXT = "txt"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"


class InputType(str, Enum):
    """Types of inputs that can be processed by the system."""
    URL = "url"
    FILE = "file"
    TEXT = "text"
    API = "api"
    DATABASE = "database"
    UNKNOWN = "unknown"


class WebsiteType(str, Enum):
    """Types of websites that can be processed by the system."""
    ECOMMERCE = "ecommerce"
    NEWS = "news"
    BLOG = "blog"
    SOCIAL_MEDIA = "social_media"
    FORUM = "forum"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    CORPORATE = "corporate"
    PERSONAL = "personal"
    UNKNOWN = "unknown"


class AgentCapability(str, Enum):
    """Capabilities that agents can have."""
    HTML_PARSING = "html_parsing"
    JSON_PARSING = "json_parsing"
    XML_PARSING = "xml_parsing"
    PDF_PROCESSING = "pdf_processing"
    DOCUMENT_PROCESSING = "document_processing"
    IMAGE_PROCESSING = "image_processing"
    VIDEO_PROCESSING = "video_processing"
    AUDIO_PROCESSING = "audio_processing"
    JAVASCRIPT_RENDERING = "javascript_rendering"
    AUTHENTICATION = "authentication"
    ANTI_DETECTION = "anti_detection"
    DATA_TRANSFORMATION = "data_transformation"
    NLP_PROCESSING = "nlp_processing"
    API_INTEGRATION = "api_integration"
    PAGINATION_HANDLING = "pagination_handling"
    PROXY_MANAGEMENT = "proxy_management"
    ERROR_RECOVERY = "error_recovery"
    DATA_VALIDATION = "data_validation"
    DATA_STORAGE = "data_storage"
    URL_INTELLIGENCE = "url_intelligence"
    CONTENT_RECOGNITION = "content_recognition"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    QUALITY_ASSURANCE = "quality_assurance"


class AgentPerformanceMetrics(BaseModel):
    """Performance metrics for an agent."""
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    error_rate: float = 0.0
    task_count: int = 0
    last_execution_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_error_time: Optional[float] = None


class AgentProfile(BaseModel):
    """Profile of an agent with its capabilities and performance metrics."""
    agent_id: str
    agent_type: str
    capabilities: Set[AgentCapability] = Field(default_factory=set)
    performance: AgentPerformanceMetrics = Field(default_factory=AgentPerformanceMetrics)
    specializations: Dict[str, float] = Field(default_factory=dict)  # Mapping of specialization to proficiency (0.0-1.0)
    last_updated: float = Field(default_factory=time.time)


class ContentAnalysisResult(BaseModel):
    """Result of content analysis."""
    content_type: ContentType
    input_type: InputType
    website_type: Optional[WebsiteType] = None
    complexity: float = 0.0  # 0.0-1.0 scale
    requires_javascript: bool = False
    requires_authentication: bool = False
    has_anti_bot: bool = False
    has_pagination: bool = False
    estimated_pages: Optional[int] = None
    language: Optional[str] = None
    confidence: float = 0.0  # 0.0-1.0 scale
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentAssignment(BaseModel):
    """Assignment of an agent to a task."""
    agent_id: str
    agent_type: str
    task_id: str
    task_type: str
    priority: int = 1
    assigned_at: float = Field(default_factory=time.time)
    estimated_completion_time: Optional[float] = None
    capabilities_used: List[AgentCapability] = Field(default_factory=list)
    confidence: float = 0.0  # 0.0-1.0 scale


class IntelligenceDecision(BaseModel):
    """Decision made by the Master Intelligence Agent."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_analysis: ContentAnalysisResult
    agent_assignments: List[AgentAssignment] = Field(default_factory=list)
    workflow_steps: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_completion_time: Optional[float] = None
    created_at: float = Field(default_factory=time.time)
    confidence: float = 0.0  # 0.0-1.0 scale
    reasoning: str = ""


class LearningEvent(BaseModel):
    """Event for the self-learning system."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class AgentSelectionCriteria(BaseModel):
    """Criteria for selecting agents for a task."""
    required_capabilities: Set[AgentCapability] = Field(default_factory=set)
    preferred_capabilities: Set[AgentCapability] = Field(default_factory=set)
    min_success_rate: float = 0.0
    max_error_rate: float = 1.0
    content_type: Optional[ContentType] = None
    website_type: Optional[WebsiteType] = None
    task_complexity: float = 0.0  # 0.0-1.0 scale
