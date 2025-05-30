"""
Task models for the web scraping system.
"""
import uuid
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, HttpUrl


class TaskStatus(str, Enum):
    """Status of a task in the system."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskType(str, Enum):
    """Types of tasks that can be performed by agents."""
    # Basic tasks
    FETCH_URL = "fetch_url"
    PARSE_CONTENT = "parse_content"
    STORE_DATA = "store_data"
    RENDER_JS = "render_js"
    EXTRACT_LINKS = "extract_links"
    FOLLOW_PAGINATION = "follow_pagination"
    AGGREGATE_RESULTS = "aggregate_results"

    # JavaScript rendering tasks
    RENDER_PAGE = "render_page"
    INTERACT_WITH_PAGE = "interact_with_page"
    SCROLL_PAGE = "scroll_page"
    TAKE_SCREENSHOT = "take_screenshot"
    EXECUTE_SCRIPT = "execute_script"

    # Authentication tasks
    AUTHENTICATE = "authenticate"
    REFRESH_SESSION = "refresh_session"
    SOLVE_CAPTCHA = "solve_captcha"
    MFA_AUTHENTICATE = "mfa_authenticate"
    VERIFY_SESSION = "verify_session"
    ROTATE_CREDENTIALS = "rotate_credentials"
    MAP_AUTH_FLOW = "map_auth_flow"

    # Anti-detection tasks
    GENERATE_FINGERPRINT = "generate_fingerprint"
    CHECK_BLOCKING = "check_blocking"
    OPTIMIZE_REQUEST_PATTERN = "optimize_request_pattern"
    SIMULATE_HUMAN_BEHAVIOR = "simulate_human_behavior"
    DETECT_HONEYPOT = "detect_honeypot"
    CUSTOMIZE_HEADERS = "customize_headers"
    ANALYZE_TRAFFIC_PATTERN = "analyze_traffic_pattern"

    # Data transformation tasks
    CLEAN_DATA = "clean_data"
    TRANSFORM_SCHEMA = "transform_schema"
    ENRICH_DATA = "enrich_data"
    ANALYZE_TEXT = "analyze_text"
    DETECT_ANOMALIES = "detect_anomalies"
    INFER_FIELD_TYPES = "infer_field_types"
    RECONCILE_ENTITIES = "reconcile_entities"
    IMPUTE_MISSING_DATA = "impute_missing_data"
    NORMALIZE_DATA = "normalize_data"

    # API Integration tasks
    API_REQUEST = "api_request"
    API_PAGINATE = "api_paginate"
    API_AUTHENTICATE = "api_authenticate"
    API_TRANSFORM = "api_transform"
    API_LEARN_SCHEMA = "api_learn_schema"
    API_MANAGE_QUOTA = "api_manage_quota"
    API_CACHE_RESPONSE = "api_cache_response"
    API_FALLBACK = "api_fallback"

    # NLP Processing tasks
    NLP_ENTITY_EXTRACTION = "nlp_entity_extraction"
    NLP_SENTIMENT_ANALYSIS = "nlp_sentiment_analysis"
    NLP_TEXT_CLASSIFICATION = "nlp_text_classification"
    NLP_KEYWORD_EXTRACTION = "nlp_keyword_extraction"
    NLP_TEXT_SUMMARIZATION = "nlp_text_summarization"
    NLP_LANGUAGE_DETECTION = "nlp_language_detection"
    NLP_TRAIN_DOMAIN_MODEL = "nlp_train_domain_model"
    NLP_CROSS_LANGUAGE_PROCESS = "nlp_cross_language_process"
    NLP_CONTEXT_ANALYSIS = "nlp_context_analysis"
    NLP_CONTENT_DEDUPLICATION = "nlp_content_deduplication"

    # Image Processing tasks
    IMAGE_DOWNLOAD = "image_download"
    IMAGE_OCR = "image_ocr"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_EXTRACTION = "image_extraction"
    IMAGE_COMPARISON = "image_comparison"
    IMAGE_SOLVE_CAPTCHA = "image_solve_captcha"
    IMAGE_CONTENT_CROP = "image_content_crop"
    IMAGE_TO_DATA = "image_to_data"
    IMAGE_SIMILARITY_DETECTION = "image_similarity_detection"
    IMAGE_WATERMARK_REMOVAL = "image_watermark_removal"

    # Monitoring & Alerting tasks
    MONITOR_SYSTEM_HEALTH = "monitor_system_health"
    TRACK_PERFORMANCE = "track_performance"
    GENERATE_ALERT = "generate_alert"
    MONITOR_RESOURCES = "monitor_resources"
    GENERATE_REPORT = "generate_report"

    # Compliance tasks
    PARSE_ROBOTS_TXT = "parse_robots_txt"
    CHECK_RATE_LIMITS = "check_rate_limits"
    MONITOR_TOS = "monitor_tos"
    CHECK_LEGAL_COMPLIANCE = "check_legal_compliance"
    ENFORCE_ETHICAL_SCRAPING = "enforce_ethical_scraping"

    # Data Quality tasks
    VALIDATE_DATA = "validate_data"
    DETECT_DATA_ANOMALIES = "detect_data_anomalies"
    CHECK_COMPLETENESS = "check_completeness"
    VERIFY_CONSISTENCY = "verify_consistency"
    SCORE_DATA_QUALITY = "score_data_quality"

    # Self-Learning tasks
    ANALYZE_PATTERNS = "analyze_patterns"
    OPTIMIZE_STRATEGY = "optimize_strategy"
    TUNE_PARAMETERS = "tune_parameters"
    DETECT_SITE_CHANGES = "detect_site_changes"
    SUGGEST_IMPROVEMENTS = "suggest_improvements"

    # Site-Specific Specialist tasks
    APPLY_SITE_RULES = "apply_site_rules"
    NAVIGATE_SITE = "navigate_site"
    HANDLE_SITE_AUTH = "handle_site_auth"
    APPLY_SITE_ANTI_DETECTION = "apply_site_anti_detection"
    OPTIMIZE_SITE_SCRAPING = "optimize_site_scraping"

    # Intelligence tasks
    ANALYZE_INPUT = "analyze_input"
    SELECT_AGENTS = "select_agents"
    ANALYZE_URL = "analyze_url"
    DETECT_TECHNOLOGIES = "detect_technologies"
    CHECK_ROBOTS_TXT = "check_robots_txt"
    RECOGNIZE_CONTENT = "recognize_content"
    DETECT_CONTENT_TYPE = "detect_content_type"
    ANALYZE_STRUCTURE = "analyze_structure"
    PROCESS_DOCUMENT = "process_document"
    EXTRACT_TEXT = "extract_text"
    EXTRACT_TABLES = "extract_tables"
    EXTRACT_METADATA = "extract_metadata"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    QUALITY_ASSURANCE = "quality_assurance"

    # Coordinator Enhancement tasks
    ADAPTIVE_SCHEDULE = "adaptive_schedule"
    RECOVER_FROM_FAILURE = "recover_from_failure"
    ANALYZE_AGENT_PERFORMANCE = "analyze_agent_performance"
    BALANCE_LOAD = "balance_load"
    CREATE_WORKFLOW_TEMPLATE = "create_workflow_template"

    # Scraper Enhancement tasks
    SMART_RATE_LIMIT = "smart_rate_limit"
    DIFF_CONTENT = "diff_content"
    PROGRESSIVE_RENDER = "progressive_render"
    MANAGE_PROXIES = "manage_proxies"
    STREAM_EXTRACT = "stream_extract"

    # Storage Enhancement tasks
    VERSION_DATA = "version_data"
    VALIDATE_SCHEMA = "validate_schema"
    UPDATE_INCREMENTALLY = "update_incrementally"
    COMPRESS_DATA = "compress_data"
    OPTIMIZE_QUERY = "optimize_query"


class Task(BaseModel):
    """
    Base task class for all operations in the system.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    assigned_to: Optional[str] = None
    priority: int = 1
    parameters: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    dependencies: List[str] = []

    def update_status(self, status: TaskStatus) -> None:
        """Update the status of the task and the updated_at timestamp."""
        self.status = status
        self.updated_at = time.time()

    def assign(self, agent_id: str) -> None:
        """Assign the task to an agent."""
        self.assigned_to = agent_id
        self.update_status(TaskStatus.ASSIGNED)

    def complete(self, result: Dict[str, Any]) -> None:
        """Mark the task as completed with the given result."""
        self.result = result
        self.update_status(TaskStatus.COMPLETED)

    def fail(self, error_type: str, error_message: str, traceback: Optional[str] = None) -> None:
        """Mark the task as failed with the given error information."""
        self.error = {
            "type": error_type,
            "message": error_message,
            "traceback": traceback
        }
        self.update_status(TaskStatus.FAILED)


class FetchUrlTask(Task):
    """
    Task for fetching content from a URL.
    """
    type: TaskType = TaskType.FETCH_URL

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com",
                    "method": "GET",
                    "headers": {"User-Agent": "Mozilla/5.0"},
                    "timeout": 30,
                    "verify_ssl": True,
                    "proxy": None,
                    "cookies": {}
                }
            }
        }


class ParseContentTask(Task):
    """
    Task for parsing content and extracting structured data.
    """
    type: TaskType = TaskType.PARSE_CONTENT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "content": "<!DOCTYPE html><html>...</html>",
                    "selectors": {
                        "title": "h1.title",
                        "price": "span.price",
                        "description": "div.description"
                    },
                    "use_xpath": False,
                    "normalize": True
                }
            }
        }


class StoreDataTask(Task):
    """
    Task for storing extracted data.
    """
    type: TaskType = TaskType.STORE_DATA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"title": "Example", "price": "$10.99"}],
                    "format": "json",
                    "path": "output/data.json",
                    "append": False,
                    "schema": {
                        "title": "string",
                        "price": "string"
                    }
                }
            }
        }


class RenderJsTask(Task):
    """
    Task for rendering JavaScript content.
    """
    type: TaskType = TaskType.RENDER_JS

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com",
                    "wait_for": "div.content",
                    "timeout": 30,
                    "script": "window.scrollTo(0, document.body.scrollHeight);",
                    "browser": "chromium",
                    "headers": {"User-Agent": "Mozilla/5.0"}
                }
            }
        }


class ExtractLinksTask(Task):
    """
    Task for extracting links from content.
    """
    type: TaskType = TaskType.EXTRACT_LINKS

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "content": "<!DOCTYPE html><html>...</html>",
                    "base_url": "https://example.com",
                    "selector": "a.product-link",
                    "attribute": "href",
                    "filter_pattern": r"product/\\d+",
                    "limit": 10
                }
            }
        }


class FollowPaginationTask(Task):
    """
    Task for following pagination links.
    """
    type: TaskType = TaskType.FOLLOW_PAGINATION

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com/products",
                    "max_pages": 5,
                    "current_page": 1,
                    "wait_between_requests": 2.0,
                    "use_browser": False,
                    # Optional parameters for fallback
                    "next_page_selector": "a.next-page"
                }
            }
        }


# JavaScript Rendering Tasks

class RenderPageTask(Task):
    """
    Task for rendering a page with a headless browser.
    """
    type: TaskType = TaskType.RENDER_PAGE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com",
                    "wait_for": "div.content",
                    "timeout": 30,
                    "viewport": {"width": 1920, "height": 1080},
                    "browser": "chromium",
                    "device": "Desktop",
                    "headers": {"User-Agent": "Mozilla/5.0"},
                    "cookies": {},
                    "wait_until": "networkidle"
                }
            }
        }


class InteractWithPageTask(Task):
    """
    Task for interacting with elements on a page.
    """
    type: TaskType = TaskType.INTERACT_WITH_PAGE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "page_id": "page-123",  # ID of a previously rendered page
                    "actions": [
                        {"type": "click", "selector": "button.submit"},
                        {"type": "fill", "selector": "input[name=search]", "value": "example"},
                        {"type": "select", "selector": "select#dropdown", "value": "option1"},
                        {"type": "wait", "duration": 2.0}
                    ],
                    "timeout": 30
                }
            }
        }


class ScrollPageTask(Task):
    """
    Task for scrolling a page to load lazy content.
    """
    type: TaskType = TaskType.SCROLL_PAGE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "page_id": "page-123",  # ID of a previously rendered page
                    "scroll_behavior": "smooth",  # or "auto"
                    "max_scrolls": 10,
                    "scroll_delay": 1.0,
                    "scroll_amount": 800,  # pixels per scroll
                    "target_selector": "div.load-more",  # Optional target to scroll to
                    "wait_for_selector": ".item:nth-child(50)"  # Wait until this appears
                }
            }
        }


class TakeScreenshotTask(Task):
    """
    Task for taking a screenshot of a page.
    """
    type: TaskType = TaskType.TAKE_SCREENSHOT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "page_id": "page-123",  # ID of a previously rendered page
                    "output_path": "screenshots/example.png",
                    "full_page": True,
                    "clip": None,  # or {"x": 0, "y": 0, "width": 500, "height": 500}
                    "omit_background": False,
                    "quality": 90  # For JPEG format
                }
            }
        }


class ExecuteScriptTask(Task):
    """
    Task for executing JavaScript on a page.
    """
    type: TaskType = TaskType.EXECUTE_SCRIPT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "page_id": "page-123",  # ID of a previously rendered page
                    "script": "return document.title;",
                    "args": [],  # Arguments to pass to the script
                    "timeout": 30
                }
            }
        }


# Authentication Tasks

class AuthenticateTask(Task):
    """
    Task for authenticating with a website.
    """
    type: TaskType = TaskType.AUTHENTICATE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com/login",
                    "credentials": {
                        "username": "user",
                        "password": "pass"
                    },
                    "method": "form",  # or "basic", "oauth", etc.
                    "form_selectors": {
                        "username": "input[name=username]",
                        "password": "input[name=password]",
                        "submit": "button[type=submit]"
                    },
                    "success_indicator": ".logged-in",
                    "failure_indicator": ".error-message",
                    "store_cookies": True
                }
            }
        }


class RefreshSessionTask(Task):
    """
    Task for refreshing an authentication session.
    """
    type: TaskType = TaskType.REFRESH_SESSION

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "session_id": "session-123",
                    "method": "cookie",  # or "token", "oauth", etc.
                    "refresh_url": "https://example.com/refresh",
                    "verify_url": "https://example.com/profile"
                }
            }
        }


class SolveCaptchaTask(Task):
    """
    Task for solving a CAPTCHA challenge.
    """
    type: TaskType = TaskType.SOLVE_CAPTCHA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "page_id": "page-123",  # ID of a previously rendered page
                    "captcha_type": "recaptcha",  # or "hcaptcha", "image", etc.
                    "captcha_selector": "#recaptcha",
                    "api_key": "your-captcha-solving-api-key",
                    "timeout": 60
                }
            }
        }


# Anti-Detection Tasks

class GenerateFingerprintTask(Task):
    """
    Task for generating a browser fingerprint.
    """
    type: TaskType = TaskType.GENERATE_FINGERPRINT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "target_site": "example.com",
                    "device_type": "desktop",  # or "mobile", "tablet"
                    "os": "windows",  # or "macos", "linux", "android", "ios"
                    "browser": "chrome",  # or "firefox", "safari", "edge"
                    "browser_version": "latest",
                    "language": "en-US",
                    "timezone": "America/New_York",
                    "screen_resolution": "1920x1080"
                }
            }
        }


class CheckBlockingTask(Task):
    """
    Task for checking if a site is blocking the scraper.
    """
    type: TaskType = TaskType.CHECK_BLOCKING

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com",
                    "check_methods": ["status_code", "content_analysis", "redirect"],
                    "reference_content": "Welcome to Example",
                    "blocking_indicators": [
                        "captcha",
                        "unusual traffic",
                        "blocked",
                        "too many requests"
                    ]
                }
            }
        }


class OptimizeRequestPatternTask(Task):
    """
    Task for optimizing request patterns to avoid detection.
    """
    type: TaskType = TaskType.OPTIMIZE_REQUEST_PATTERN

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "current_success_rate": 0.75,
                    "current_request_interval": 2.0,
                    "current_concurrency": 3,
                    "history_length": 100,
                    "optimization_goal": "success_rate"  # or "throughput", "balance"
                }
            }
        }


# Data Transformation Tasks

class CleanDataTask(Task):
    """
    Task for cleaning and normalizing data.
    """
    type: TaskType = TaskType.CLEAN_DATA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"title": " Example Title ", "price": "$10.99"}],
                    "operations": [
                        {"field": "title", "operation": "strip_whitespace"},
                        {"field": "price", "operation": "extract_number"},
                        {"field": "*", "operation": "remove_empty"}
                    ],
                    "add_metadata": True
                }
            }
        }


class TransformSchemaTask(Task):
    """
    Task for transforming data from one schema to another.
    """
    type: TaskType = TaskType.TRANSFORM_SCHEMA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"title": "Example", "price": 10.99}],
                    "source_schema": {
                        "title": "string",
                        "price": "number"
                    },
                    "target_schema": {
                        "name": "string",
                        "cost": "number"
                    },
                    "mapping": {
                        "name": "title",
                        "cost": "price"
                    }
                }
            }
        }


class EnrichDataTask(Task):
    """
    Task for enriching data with additional information.
    """
    type: TaskType = TaskType.ENRICH_DATA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"product_id": "123", "name": "Example Product"}],
                    "enrichment_sources": [
                        {
                            "type": "api",
                            "url": "https://api.example.com/products/{product_id}",
                            "method": "GET",
                            "headers": {"Authorization": "Bearer token"},
                            "mapping": {
                                "description": "description",
                                "category": "category.name"
                            }
                        }
                    ],
                    "fallback_behavior": "keep_original"
                }
            }
        }


class AnalyzeTextTask(Task):
    """
    Task for analyzing text content.
    """
    type: TaskType = TaskType.ANALYZE_TEXT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "text": "This is a sample product description. It's very good quality.",
                    "analyses": ["sentiment", "entities", "keywords", "language"],
                    "language": "en",
                    "sentiment_model": "default",
                    "entity_types": ["PERSON", "ORG", "PRODUCT"]
                }
            }
        }


# Monitoring & Alerting Tasks

class MonitorSystemHealthTask(Task):
    """
    Task for monitoring the overall system health.
    """
    type: TaskType = TaskType.MONITOR_SYSTEM_HEALTH

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "components": ["scraper", "parser", "storage", "api_integration"],
                    "metrics": ["cpu", "memory", "response_time", "error_rate"],
                    "interval": 60,  # seconds
                    "thresholds": {
                        "cpu": 80,  # percent
                        "memory": 70,  # percent
                        "response_time": 2000,  # ms
                        "error_rate": 0.05  # 5%
                    }
                }
            }
        }


class TrackPerformanceTask(Task):
    """
    Task for tracking performance metrics of agents.
    """
    type: TaskType = TaskType.TRACK_PERFORMANCE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "agent_id": "scraper-1",
                    "metrics": ["tasks_completed", "success_rate", "avg_execution_time"],
                    "time_period": "1h",  # 1 hour
                    "aggregation": "avg"  # average
                }
            }
        }


class GenerateAlertTask(Task):
    """
    Task for generating alerts based on system conditions.
    """
    type: TaskType = TaskType.GENERATE_ALERT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "alert_type": "error",  # or "warning", "info"
                    "component": "scraper",
                    "message": "High error rate detected",
                    "details": {
                        "error_rate": 0.15,
                        "threshold": 0.05,
                        "time_period": "10m"
                    },
                    "notification_channels": ["email", "slack"],
                    "priority": "high"  # or "medium", "low"
                }
            }
        }


class MonitorResourcesTask(Task):
    """
    Task for monitoring system resource usage.
    """
    type: TaskType = TaskType.MONITOR_RESOURCES

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "resources": ["cpu", "memory", "disk", "network"],
                    "interval": 30,  # seconds
                    "detailed": True,
                    "process_specific": True,
                    "include_os_metrics": True
                }
            }
        }


class GenerateReportTask(Task):
    """
    Task for generating performance and status reports.
    """
    type: TaskType = TaskType.GENERATE_REPORT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "report_type": "performance",  # or "status", "error", "usage"
                    "time_period": "24h",  # 24 hours
                    "components": ["all"],  # or specific components
                    "format": "html",  # or "json", "csv", "pdf"
                    "include_charts": True,
                    "include_recommendations": True
                }
            }
        }


# Compliance Tasks

class ParseRobotsTxtTask(Task):
    """
    Task for parsing and enforcing robots.txt rules.
    """
    type: TaskType = TaskType.PARSE_ROBOTS_TXT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "user_agent": "MyScraperBot",
                    "cache_duration": 86400,  # 24 hours in seconds
                    "respect_crawl_delay": True,
                    "follow_sitemaps": True
                }
            }
        }


class CheckRateLimitsTask(Task):
    """
    Task for checking and enforcing rate limits.
    """
    type: TaskType = TaskType.CHECK_RATE_LIMITS

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "max_requests_per_minute": 10,
                    "max_concurrent_requests": 2,
                    "respect_retry_after": True,
                    "adaptive": True,
                    "backoff_factor": 2.0
                }
            }
        }


class MonitorTosTask(Task):
    """
    Task for monitoring terms of service changes.
    """
    type: TaskType = TaskType.MONITOR_TOS

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "tos_url": "https://example.com/terms",
                    "check_frequency": "weekly",
                    "notify_on_change": True,
                    "keywords_to_monitor": [
                        "scraping",
                        "crawling",
                        "automated",
                        "bot",
                        "data collection"
                    ]
                }
            }
        }


class CheckLegalComplianceTask(Task):
    """
    Task for checking legal compliance of scraping activities.
    """
    type: TaskType = TaskType.CHECK_LEGAL_COMPLIANCE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "jurisdiction": "US",
                    "data_types": ["public", "personal"],
                    "check_copyright": True,
                    "check_gdpr": True,
                    "check_ccpa": True
                }
            }
        }


class EnforceEthicalScrapingTask(Task):
    """
    Task for enforcing ethical scraping guidelines.
    """
    type: TaskType = TaskType.ENFORCE_ETHICAL_SCRAPING

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "guidelines": [
                        "respect_robots_txt",
                        "limit_request_rate",
                        "identify_bot",
                        "minimize_impact",
                        "respect_copyright"
                    ],
                    "identify_as": "MyScraperBot (contact@example.com)",
                    "abort_on_violation": True
                }
            }
        }


# Data Quality Tasks

class ValidateDataTask(Task):
    """
    Task for validating data against schemas.
    """
    type: TaskType = TaskType.VALIDATE_DATA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"title": "Example", "price": 10.99}],
                    "schema": {
                        "title": {"type": "string", "required": True, "min_length": 1},
                        "price": {"type": "number", "required": True, "min": 0}
                    },
                    "strict": True,
                    "report_all_errors": True,
                    "coerce_types": False
                }
            }
        }


class DetectDataAnomaliesTask(Task):
    """
    Task for detecting anomalies in scraped data.
    """
    type: TaskType = TaskType.DETECT_DATA_ANOMALIES

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"title": "Example", "price": 10.99}],
                    "fields_to_check": ["price"],
                    "methods": ["statistical", "isolation_forest", "one_class_svm"],
                    "reference_data": "historical_prices.json",
                    "sensitivity": 0.8,  # 0.0 to 1.0
                    "include_explanation": True
                }
            }
        }


class CheckCompletenessTask(Task):
    """
    Task for checking data completeness.
    """
    type: TaskType = TaskType.CHECK_COMPLETENESS

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"title": "Example", "price": 10.99, "description": None}],
                    "required_fields": ["title", "price", "description", "image_url"],
                    "completeness_threshold": 0.8,  # 80% of fields must be present
                    "field_weights": {
                        "title": 1.0,
                        "price": 1.0,
                        "description": 0.8,
                        "image_url": 0.5
                    }
                }
            }
        }


class VerifyConsistencyTask(Task):
    """
    Task for verifying data consistency.
    """
    type: TaskType = TaskType.VERIFY_CONSISTENCY

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"title": "Example", "price": 10.99, "sale_price": 8.99}],
                    "consistency_rules": [
                        {
                            "rule": "price >= sale_price",
                            "error_message": "Regular price must be greater than or equal to sale price"
                        },
                        {
                            "rule": "len(title) > 0",
                            "error_message": "Title cannot be empty"
                        }
                    ],
                    "check_cross_record_consistency": True
                }
            }
        }


class ScoreDataQualityTask(Task):
    """
    Task for scoring overall data quality.
    """
    type: TaskType = TaskType.SCORE_DATA_QUALITY

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"title": "Example", "price": 10.99, "description": "A good product"}],
                    "dimensions": ["completeness", "accuracy", "consistency", "timeliness"],
                    "weights": {
                        "completeness": 0.3,
                        "accuracy": 0.3,
                        "consistency": 0.2,
                        "timeliness": 0.2
                    },
                    "reference_data": "reference_dataset.json",
                    "generate_report": True
                }
            }
        }


# Self-Learning Tasks

class AnalyzePatternsTask(Task):
    """
    Task for analyzing success/failure patterns.
    """
    type: TaskType = TaskType.ANALYZE_PATTERNS

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "time_period": "7d",  # 7 days
                    "task_types": ["FETCH_URL", "PARSE_CONTENT"],
                    "success_threshold": 0.8,  # 80% success rate
                    "factors_to_analyze": [
                        "time_of_day",
                        "request_interval",
                        "user_agent",
                        "proxy_used"
                    ],
                    "min_sample_size": 100
                }
            }
        }


class OptimizeStrategyTask(Task):
    """
    Task for optimizing scraping strategies.
    """
    type: TaskType = TaskType.OPTIMIZE_STRATEGY

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "optimization_goal": "success_rate",  # or "throughput", "stealth"
                    "strategies_to_test": [
                        "request_interval",
                        "concurrency",
                        "user_agent_rotation",
                        "proxy_rotation"
                    ],
                    "test_iterations": 10,
                    "apply_immediately": False
                }
            }
        }


class TuneParametersTask(Task):
    """
    Task for tuning scraping parameters.
    """
    type: TaskType = TaskType.TUNE_PARAMETERS

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "parameters_to_tune": {
                        "request_interval": {"min": 1.0, "max": 5.0, "step": 0.5},
                        "concurrency": {"min": 1, "max": 5, "step": 1},
                        "timeout": {"min": 10, "max": 60, "step": 5}
                    },
                    "optimization_metric": "success_rate",
                    "test_duration": "1h",  # 1 hour
                    "apply_best_parameters": True
                }
            }
        }


class DetectSiteChangesTask(Task):
    """
    Task for detecting changes in website structure.
    """
    type: TaskType = TaskType.DETECT_SITE_CHANGES

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com/products",
                    "check_frequency": "daily",
                    "elements_to_monitor": [
                        {"selector": ".product-item", "attribute": "structure"},
                        {"selector": ".pagination", "attribute": "structure"},
                        {"selector": "form#search", "attribute": "structure"}
                    ],
                    "notify_on_change": True,
                    "update_selectors_automatically": False
                }
            }
        }


class SuggestImprovementsTask(Task):
    """
    Task for suggesting improvements to the scraping system.
    """
    type: TaskType = TaskType.SUGGEST_IMPROVEMENTS

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "performance_data_period": "30d",  # 30 days
                    "areas_to_analyze": [
                        "success_rate",
                        "speed",
                        "resource_usage",
                        "data_quality"
                    ],
                    "max_suggestions": 5,
                    "include_implementation_details": True
                }
            }
        }


# Site-Specific Specialist Tasks

class ApplySiteRulesTask(Task):
    """
    Task for applying site-specific extraction rules.
    """
    type: TaskType = TaskType.APPLY_SITE_RULES

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "content": "<!DOCTYPE html><html>...</html>",
                    "rule_set": "example_com_rules",
                    "extraction_targets": ["products", "categories", "prices"],
                    "fallback_to_generic": True
                }
            }
        }


class NavigateSiteTask(Task):
    """
    Task for navigating a specific site using custom patterns.
    """
    type: TaskType = TaskType.NAVIGATE_SITE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "start_url": "https://example.com",
                    "navigation_pattern": "category_listing_product_detail",
                    "max_depth": 3,
                    "max_pages_per_level": 5,
                    "custom_selectors": {
                        "category_links": ".category-item a",
                        "product_links": ".product-item a",
                        "next_page": ".pagination .next"
                    }
                }
            }
        }


class HandleSiteAuthTask(Task):
    """
    Task for handling site-specific authentication.
    """
    type: TaskType = TaskType.HANDLE_SITE_AUTH

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "auth_type": "custom_form",
                    "credentials": {
                        "username": "user",
                        "password": "pass"
                    },
                    "auth_flow": [
                        {"action": "navigate", "url": "https://example.com/login"},
                        {"action": "wait_for", "selector": "form#login"},
                        {"action": "fill", "selector": "input#username", "value": "{username}"},
                        {"action": "fill", "selector": "input#password", "value": "{password}"},
                        {"action": "click", "selector": "button[type=submit]"},
                        {"action": "wait_for", "selector": ".logged-in-indicator"}
                    ],
                    "session_indicators": [".user-menu", ".logout-button"]
                }
            }
        }


class ApplySiteAntiDetectionTask(Task):
    """
    Task for applying site-specific anti-detection measures.
    """
    type: TaskType = TaskType.APPLY_SITE_ANTI_DETECTION

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "anti_detection_profile": "example_com_profile",
                    "measures": [
                        "custom_headers",
                        "browser_fingerprint",
                        "request_pattern",
                        "javascript_behavior"
                    ],
                    "custom_headers": {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://www.google.com/"
                    },
                    "browser_behavior": {
                        "mouse_movements": True,
                        "scroll_behavior": "natural",
                        "typing_speed": "human"
                    }
                }
            }
        }


class OptimizeSiteScrapingTask(Task):
    """
    Task for optimizing scraping for a specific site.
    """
    type: TaskType = TaskType.OPTIMIZE_SITE_SCRAPING

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "optimization_profile": "example_com_optimizations",
                    "areas": [
                        "request_timing",
                        "resource_loading",
                        "selector_efficiency",
                        "data_extraction"
                    ],
                    "request_timing": {
                        "min_interval": 2.0,
                        "max_interval": 5.0,
                        "variance": 0.3
                    },
                    "resource_loading": {
                        "block_resources": ["images", "fonts", "analytics"],
                        "load_essential_only": True
                    }
                }
            }
        }


# Intelligence Tasks

class AnalyzeInputTask(Task):
    """
    Task for analyzing input data to determine its type and characteristics.
    """
    type: TaskType = TaskType.ANALYZE_INPUT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "input_data": "https://example.com/products/123",
                    "analyze_content": True,
                    "detect_type": True,
                    "estimate_complexity": True
                }
            }
        }


class SelectAgentsTask(Task):
    """
    Task for selecting appropriate agents based on content analysis.
    """
    type: TaskType = TaskType.SELECT_AGENTS

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "analysis_result": {
                        "content_type": "html",
                        "input_type": "url",
                        "website_type": "ecommerce",
                        "complexity": 0.7,
                        "requires_javascript": True,
                        "requires_authentication": False,
                        "has_anti_bot": False,
                        "has_pagination": True
                    },
                    "min_success_rate": 0.8,
                    "max_error_rate": 0.2
                }
            }
        }


class AnalyzeUrlTask(Task):
    """
    Task for analyzing a URL to determine its characteristics.
    """
    type: TaskType = TaskType.ANALYZE_URL

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com/products/123",
                    "check_robots_txt": True,
                    "detect_technologies": True,
                    "analyze_structure": True
                }
            }
        }


class DetectTechnologiesTask(Task):
    """
    Task for detecting technologies used on a website.
    """
    type: TaskType = TaskType.DETECT_TECHNOLOGIES

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "content": "<!DOCTYPE html><html><head><script src='react.js'></script></head><body></body></html>",
                    "url": "https://example.com"
                }
            }
        }


class CheckRobotsTxtTask(Task):
    """
    Task for checking robots.txt for a URL.
    """
    type: TaskType = TaskType.CHECK_ROBOTS_TXT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com",
                    "user_agent": "MyScraperBot"
                }
            }
        }


class RecognizeContentTask(Task):
    """
    Task for recognizing and analyzing input content.
    """
    type: TaskType = TaskType.RECOGNIZE_CONTENT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "content": "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>",
                    "extract_metadata": True,
                    "analyze_structure": True
                }
            }
        }


class DetectContentTypeTask(Task):
    """
    Task for detecting the content type of input data.
    """
    type: TaskType = TaskType.DETECT_CONTENT_TYPE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "content": "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>",
                    "content_type_header": "text/html; charset=utf-8"
                }
            }
        }


class AnalyzeStructureTask(Task):
    """
    Task for analyzing the structure of content.
    """
    type: TaskType = TaskType.ANALYZE_STRUCTURE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "content": "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>",
                    "content_type": "html"
                }
            }
        }


class ProcessDocumentTask(Task):
    """
    Task for processing a document and extracting its content.
    """
    type: TaskType = TaskType.PROCESS_DOCUMENT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "document_data": "https://example.com/document.pdf",
                    "document_type": "pdf"
                }
            }
        }


class ExtractTextTask(Task):
    """
    Task for extracting text from a document.
    """
    type: TaskType = TaskType.EXTRACT_TEXT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "document_data": "https://example.com/document.pdf",
                    "document_type": "pdf"
                }
            }
        }


class ExtractTablesTask(Task):
    """
    Task for extracting tables from a document.
    """
    type: TaskType = TaskType.EXTRACT_TABLES

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "document_data": "https://example.com/document.xlsx",
                    "document_type": "xlsx"
                }
            }
        }


class ExtractMetadataTask(Task):
    """
    Task for extracting metadata from a document.
    """
    type: TaskType = TaskType.EXTRACT_METADATA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "document_data": "https://example.com/document.pdf",
                    "document_type": "pdf"
                }
            }
        }


class OptimizePerformanceTask(Task):
    """
    Task for optimizing system performance.
    """
    type: TaskType = TaskType.OPTIMIZE_PERFORMANCE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "target_components": ["scraper", "parser", "storage"],
                    "optimization_goals": ["speed", "memory", "accuracy"],
                    "max_resource_usage": {
                        "cpu": 0.8,  # 80% CPU usage
                        "memory": 0.7  # 70% memory usage
                    }
                }
            }
        }


class QualityAssuranceTask(Task):
    """
    Task for ensuring high-quality output.
    """
    type: TaskType = TaskType.QUALITY_ASSURANCE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"title": "Product 1", "price": 10.99}],
                    "schema": {
                        "title": {"type": "string", "required": True},
                        "price": {"type": "number", "required": True},
                        "description": {"type": "string", "required": False}
                    },
                    "quality_thresholds": {
                        "completeness": 0.9,
                        "accuracy": 0.95,
                        "consistency": 0.9
                    }
                }
            }
        }


# Coordinator Enhancement Tasks

class AdaptiveScheduleTask(Task):
    """
    Task for adaptive scheduling of scraping tasks.
    """
    type: TaskType = TaskType.ADAPTIVE_SCHEDULE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "task_types": ["FETCH_URL", "PARSE_CONTENT"],
                    "priority_factors": {
                        "time_sensitivity": 0.4,
                        "value": 0.3,
                        "difficulty": 0.2,
                        "dependencies": 0.1
                    },
                    "resource_constraints": {
                        "max_concurrent_tasks": 10,
                        "max_tasks_per_minute": 30
                    },
                    "adaptive_period": "1h"  # Adjust scheduling every hour
                }
            }
        }


class RecoverFromFailureTask(Task):
    """
    Task for recovering from failures with exponential backoff.
    """
    type: TaskType = TaskType.RECOVER_FROM_FAILURE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "failed_task_id": "task-123",
                    "failure_type": "timeout",  # or "connection_error", "parsing_error", etc.
                    "retry_count": 3,
                    "backoff_factor": 2.0,
                    "max_retry_delay": 300,  # seconds
                    "jitter": True,
                    "alternative_strategies": ["use_different_proxy", "change_user_agent"]
                }
            }
        }


class AnalyzeAgentPerformanceTask(Task):
    """
    Task for analyzing agent performance.
    """
    type: TaskType = TaskType.ANALYZE_AGENT_PERFORMANCE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "agent_types": ["scraper", "parser", "storage"],
                    "metrics": ["success_rate", "execution_time", "resource_usage"],
                    "time_period": "24h",
                    "granularity": "1h",
                    "include_task_breakdown": True,
                    "generate_visualization": True
                }
            }
        }


class BalanceLoadTask(Task):
    """
    Task for balancing load across agents.
    """
    type: TaskType = TaskType.BALANCE_LOAD

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "agent_types": ["scraper", "parser", "storage"],
                    "balancing_strategy": "performance_based",  # or "round_robin", "least_busy"
                    "performance_metrics": ["cpu_usage", "memory_usage", "task_queue_length"],
                    "rebalance_threshold": 0.2,  # 20% imbalance triggers rebalancing
                    "max_tasks_to_reassign": 10
                }
            }
        }


class CreateWorkflowTemplateTask(Task):
    """
    Task for creating reusable workflow templates.
    """
    type: TaskType = TaskType.CREATE_WORKFLOW_TEMPLATE

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "template_name": "product_catalog_scraper",
                    "description": "Template for scraping product catalogs",
                    "workflow_steps": [
                        {
                            "task_type": "FETCH_URL",
                            "parameters": {
                                "url": "{catalog_url}",
                                "method": "GET",
                                "headers": {"User-Agent": "{user_agent}"}
                            }
                        },
                        {
                            "task_type": "PARSE_CONTENT",
                            "parameters": {
                                "selectors": {
                                    "products": "{product_selector}",
                                    "next_page": "{next_page_selector}"
                                }
                            },
                            "depends_on": 0  # Index of the previous task
                        }
                    ],
                    "variables": ["catalog_url", "user_agent", "product_selector", "next_page_selector"],
                    "save_to_library": True
                }
            }
        }


# Scraper Enhancement Tasks

class SmartRateLimitTask(Task):
    """
    Task for implementing smart rate limiting.
    """
    type: TaskType = TaskType.SMART_RATE_LIMIT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "initial_rate": 10,  # requests per minute
                    "min_rate": 1,
                    "max_rate": 30,
                    "adjustment_factors": {
                        "response_time": {"weight": 0.3, "threshold": 2000},  # ms
                        "error_rate": {"weight": 0.4, "threshold": 0.05},
                        "success_rate": {"weight": 0.3, "threshold": 0.95}
                    },
                    "adjustment_period": "5m",  # 5 minutes
                    "learning_rate": 0.1
                }
            }
        }


class DiffContentTask(Task):
    """
    Task for diffing content between scrapes.
    """
    type: TaskType = TaskType.DIFF_CONTENT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com/products",
                    "previous_content_id": "content-123",
                    "diff_method": "html_structure",  # or "text_only", "semantic"
                    "selector_focus": ".product-container",
                    "ignore_elements": [".ad-banner", ".timestamp", ".view-count"],
                    "process_only_changes": True
                }
            }
        }


class ProgressiveRenderTask(Task):
    """
    Task for progressive rendering of web pages.
    """
    type: TaskType = TaskType.PROGRESSIVE_RENDER

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com/products",
                    "critical_selectors": [".product-grid", ".pagination"],
                    "timeout_per_element": 2000,  # ms
                    "max_total_wait": 10000,  # ms
                    "extract_as_available": True,
                    "continue_on_partial": True
                }
            }
        }


class ManageProxiesTask(Task):
    """
    Task for managing distributed proxies.
    """
    type: TaskType = TaskType.MANAGE_PROXIES

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "domain": "example.com",
                    "proxy_pool": ["proxy1.example.com", "proxy2.example.com"],
                    "rotation_strategy": "performance",  # or "round_robin", "random"
                    "test_proxies": True,
                    "max_consecutive_failures": 3,
                    "proxy_cooldown_period": "10m",  # 10 minutes
                    "geolocation_requirements": ["US", "UK", "CA"]
                }
            }
        }


class StreamExtractTask(Task):
    """
    Task for streaming extraction of content.
    """
    type: TaskType = TaskType.STREAM_EXTRACT

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "url": "https://example.com/products",
                    "extraction_targets": [
                        {"selector": ".product-item", "batch_size": 10},
                        {"selector": ".category-item", "batch_size": 5}
                    ],
                    "process_immediately": True,
                    "max_buffer_size": 100,
                    "timeout": 30  # seconds
                }
            }
        }


# Storage Enhancement Tasks

class VersionDataTask(Task):
    """
    Task for versioning stored data.
    """
    type: TaskType = TaskType.VERSION_DATA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"id": "123", "title": "Example", "price": 10.99}],
                    "dataset_id": "products",
                    "version_strategy": "timestamp",  # or "incremental", "semantic"
                    "keep_versions": 10,
                    "diff_with_previous": True,
                    "metadata": {
                        "source": "example.com",
                        "scrape_date": "2023-05-15"
                    }
                }
            }
        }


class ValidateSchemaTask(Task):
    """
    Task for validating data against a schema.
    """
    type: TaskType = TaskType.VALIDATE_SCHEMA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "data": [{"id": "123", "title": "Example", "price": 10.99}],
                    "schema": {
                        "type": "object",
                        "required": ["id", "title", "price"],
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string", "minLength": 1},
                            "price": {"type": "number", "minimum": 0}
                        }
                    },
                    "validation_level": "strict",  # or "relaxed"
                    "handle_invalid": "reject"  # or "fix", "flag"
                }
            }
        }


class UpdateIncrementallyTask(Task):
    """
    Task for incrementally updating stored data.
    """
    type: TaskType = TaskType.UPDATE_INCREMENTALLY

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "dataset_id": "products",
                    "new_data": [{"id": "123", "title": "Example", "price": 10.99}],
                    "key_field": "id",
                    "update_strategy": "merge",  # or "replace", "append"
                    "conflict_resolution": "newest_wins",  # or "keep_existing", "manual"
                    "track_changes": True,
                    "update_timestamp": True
                }
            }
        }


class CompressDataTask(Task):
    """
    Task for compressing stored data.
    """
    type: TaskType = TaskType.COMPRESS_DATA

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "dataset_id": "products",
                    "compression_algorithm": "gzip",  # or "lzma", "brotli"
                    "compression_level": 6,  # 1-9
                    "field_specific_compression": {
                        "description": "text_specific",
                        "image_data": "image_specific"
                    },
                    "compress_metadata": False
                }
            }
        }


class OptimizeQueryTask(Task):
    """
    Task for optimizing data queries.
    """
    type: TaskType = TaskType.OPTIMIZE_QUERY

    class Config:
        schema_extra = {
            "example": {
                "parameters": {
                    "dataset_id": "products",
                    "query_patterns": [
                        {"fields": ["category", "price"], "frequency": "high"},
                        {"fields": ["title", "description"], "frequency": "medium"}
                    ],
                    "index_strategy": "automatic",  # or "manual"
                    "manual_indexes": ["category", "price_range"],
                    "analyze_query_performance": True
                }
            }
        }
