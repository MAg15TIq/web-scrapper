"""
Web Configuration for Multi-Agent Web Scraping System
Handles configuration for the web interface and API.
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    url: str = Field(default="sqlite:///./webscraper.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")


class RedisConfig(BaseModel):
    """Redis configuration."""
    
    url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    password: Optional[str] = Field(default=None, description="Redis password")
    max_connections: int = Field(default=10, description="Maximum connections")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, description="Socket connect timeout")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    secret_key: str = Field(default="your-secret-key-change-in-production", description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration")
    password_min_length: int = Field(default=8, description="Minimum password length")
    require_email_verification: bool = Field(default=False, description="Require email verification")
    max_login_attempts: int = Field(default=5, description="Maximum login attempts")
    lockout_duration_minutes: int = Field(default=15, description="Account lockout duration")


class CORSConfig(BaseModel):
    """CORS configuration."""
    
    allow_origins: List[str] = Field(default=["*"], description="Allowed origins")
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    allow_methods: List[str] = Field(default=["*"], description="Allowed methods")
    allow_headers: List[str] = Field(default=["*"], description="Allowed headers")
    expose_headers: List[str] = Field(default=[], description="Exposed headers")
    max_age: int = Field(default=600, description="Preflight cache duration")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    
    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_minute: int = Field(default=60, description="Requests per minute")
    requests_per_hour: int = Field(default=1000, description="Requests per hour")
    burst_size: int = Field(default=10, description="Burst size")
    storage_url: str = Field(default="memory://", description="Rate limit storage URL")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    
    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_endpoint: str = Field(default="/metrics", description="Metrics endpoint")
    health_endpoint: str = Field(default="/health", description="Health endpoint")
    update_interval: int = Field(default=5, description="Update interval in seconds")
    retention_days: int = Field(default=30, description="Metrics retention in days")
    alert_thresholds: Dict[str, float] = Field(
        default={
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0
        },
        description="Alert thresholds"
    )


class WebSocketConfig(BaseModel):
    """WebSocket configuration."""
    
    enabled: bool = Field(default=True, description="Enable WebSocket support")
    max_connections: int = Field(default=100, description="Maximum concurrent connections")
    ping_interval: int = Field(default=20, description="Ping interval in seconds")
    ping_timeout: int = Field(default=10, description="Ping timeout in seconds")
    close_timeout: int = Field(default=10, description="Close timeout in seconds")
    max_message_size: int = Field(default=1024 * 1024, description="Maximum message size in bytes")


class WebConfig(BaseSettings):
    """Main web configuration."""
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Application configuration
    title: str = Field(default="Multi-Agent Web Scraping System", description="Application title")
    description: str = Field(default="Advanced web scraping with AI agents", description="Application description")
    version: str = Field(default="2.0.0", description="Application version")
    
    # API configuration
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    docs_url: str = Field(default="/docs", description="API documentation URL")
    redoc_url: str = Field(default="/redoc", description="ReDoc URL")
    openapi_url: str = Field(default="/openapi.json", description="OpenAPI schema URL")
    
    # Static files
    static_dir: str = Field(default="web/frontend/dist", description="Static files directory")
    templates_dir: str = Field(default="web/templates", description="Templates directory")
    
    # Upload configuration
    upload_dir: str = Field(default="uploads", description="Upload directory")
    max_upload_size: int = Field(default=10 * 1024 * 1024, description="Maximum upload size in bytes")
    allowed_extensions: List[str] = Field(
        default=[".json", ".csv", ".xlsx", ".pdf", ".txt", ".html"],
        description="Allowed file extensions"
    )
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    
    # Agent system configuration
    agent_timeout: int = Field(default=300, description="Agent operation timeout")
    max_concurrent_jobs: int = Field(default=10, description="Maximum concurrent jobs")
    job_cleanup_interval: int = Field(default=3600, description="Job cleanup interval in seconds")
    
    # Feature flags
    enable_auth: bool = Field(default=True, description="Enable authentication")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    enable_websockets: bool = Field(default=True, description="Enable WebSocket support")
    enable_file_uploads: bool = Field(default=True, description="Enable file uploads")
    
    class Config:
        env_prefix = "WEB_"
        env_file = ".env"
        case_sensitive = False
    
    @validator('port')
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator('workers')
    def validate_workers(cls, v):
        """Validate number of workers."""
        if v < 1:
            raise ValueError("Workers must be at least 1")
        return v
    
    @validator('max_upload_size')
    def validate_max_upload_size(cls, v):
        """Validate maximum upload size."""
        if v < 1024:  # Minimum 1KB
            raise ValueError("Maximum upload size must be at least 1KB")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    def get_database_url(self) -> str:
        """Get database URL with environment variable override."""
        return os.getenv("DATABASE_URL", self.database.url)
    
    def get_redis_url(self) -> str:
        """Get Redis URL with environment variable override."""
        return os.getenv("REDIS_URL", self.redis.url)
    
    def get_secret_key(self) -> str:
        """Get secret key with environment variable override."""
        return os.getenv("SECRET_KEY", self.security.secret_key)
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins with environment variable override."""
        origins_env = os.getenv("CORS_ORIGINS")
        if origins_env:
            return [origin.strip() for origin in origins_env.split(",")]
        return self.cors.allow_origins
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.upload_dir,
            "logs",
            "data",
            "temp"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Global configuration instance
_config: Optional[WebConfig] = None


def get_web_config() -> WebConfig:
    """Get the web configuration instance."""
    global _config
    
    if _config is None:
        _config = WebConfig()
        _config.setup_directories()
    
    return _config


def reload_config():
    """Reload the configuration."""
    global _config
    _config = None
    return get_web_config()


# Configuration for different environments
def get_development_config() -> WebConfig:
    """Get development configuration."""
    config = WebConfig(
        debug=True,
        reload=True,
        log_level="DEBUG",
        enable_auth=False,
        enable_rate_limiting=False
    )
    config.setup_directories()
    return config


def get_production_config() -> WebConfig:
    """Get production configuration."""
    config = WebConfig(
        debug=False,
        reload=False,
        log_level="INFO",
        enable_auth=True,
        enable_rate_limiting=True,
        workers=4
    )
    config.setup_directories()
    return config


def get_testing_config() -> WebConfig:
    """Get testing configuration."""
    config = WebConfig(
        debug=True,
        log_level="DEBUG",
        database=DatabaseConfig(url="sqlite:///:memory:"),
        redis=RedisConfig(url="redis://localhost:6379/1"),
        enable_auth=False,
        enable_rate_limiting=False
    )
    config.setup_directories()
    return config
