"""
Configuration settings for the LangChain & Pydantic AI enhanced system.
"""
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class LangChainConfig(PydanticBaseSettings):
    """Configuration for LangChain integration."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    openai_temperature: float = Field(0.1, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(2000, env="OPENAI_MAX_TOKENS")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    
    # LangChain Settings
    langchain_tracing: bool = Field(False, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field("web-scraping-agents", env="LANGCHAIN_PROJECT")
    langchain_endpoint: Optional[str] = Field(None, env="LANGCHAIN_ENDPOINT")
    
    # Agent Configuration
    max_iterations: int = Field(10, env="AGENT_MAX_ITERATIONS")
    agent_timeout: int = Field(300, env="AGENT_TIMEOUT")
    enable_memory: bool = Field(True, env="ENABLE_AGENT_MEMORY")
    memory_window_size: int = Field(10, env="MEMORY_WINDOW_SIZE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class RedisConfig(PydanticBaseSettings):
    """Configuration for Redis message broker."""
    
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_ssl: bool = Field(False, env="REDIS_SSL")
    
    # Connection pool settings
    redis_max_connections: int = Field(20, env="REDIS_MAX_CONNECTIONS")
    redis_retry_on_timeout: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT")
    redis_socket_timeout: int = Field(30, env="REDIS_SOCKET_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DatabaseConfig(PydanticBaseSettings):
    """Configuration for database connections."""
    
    # PostgreSQL Configuration
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    postgres_db: str = Field("web_scraping", env="POSTGRES_DB")
    postgres_user: str = Field("postgres", env="POSTGRES_USER")
    postgres_password: str = Field("password", env="POSTGRES_PASSWORD")
    
    # Connection pool settings
    postgres_pool_size: int = Field(10, env="POSTGRES_POOL_SIZE")
    postgres_max_overflow: int = Field(20, env="POSTGRES_MAX_OVERFLOW")
    postgres_pool_timeout: int = Field(30, env="POSTGRES_POOL_TIMEOUT")
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class MonitoringConfig(PydanticBaseSettings):
    """Configuration for monitoring and observability."""
    
    # Prometheus Configuration
    prometheus_enabled: bool = Field(True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(8000, env="PROMETHEUS_PORT")
    prometheus_host: str = Field("0.0.0.0", env="PROMETHEUS_HOST")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Metrics Collection
    collect_agent_metrics: bool = Field(True, env="COLLECT_AGENT_METRICS")
    collect_workflow_metrics: bool = Field(True, env="COLLECT_WORKFLOW_METRICS")
    metrics_retention_days: int = Field(30, env="METRICS_RETENTION_DAYS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class AgentSystemConfig(PydanticBaseSettings):
    """Configuration for the agent system."""
    
    # System Settings
    max_concurrent_workflows: int = Field(10, env="MAX_CONCURRENT_WORKFLOWS")
    default_timeout: int = Field(300, env="DEFAULT_TIMEOUT")
    max_retry_attempts: int = Field(3, env="MAX_RETRY_ATTEMPTS")
    
    # Agent Pool Configuration
    min_agents_per_type: int = Field(1, env="MIN_AGENTS_PER_TYPE")
    max_agents_per_type: int = Field(5, env="MAX_AGENTS_PER_TYPE")
    agent_idle_timeout: int = Field(600, env="AGENT_IDLE_TIMEOUT")  # 10 minutes
    
    # Workflow Configuration
    enable_workflow_persistence: bool = Field(True, env="ENABLE_WORKFLOW_PERSISTENCE")
    workflow_checkpoint_interval: int = Field(30, env="WORKFLOW_CHECKPOINT_INTERVAL")  # seconds
    
    # Performance Settings
    enable_performance_optimization: bool = Field(True, env="ENABLE_PERFORMANCE_OPTIMIZATION")
    adaptive_rate_limiting: bool = Field(True, env="ADAPTIVE_RATE_LIMITING")
    intelligent_retry: bool = Field(True, env="INTELLIGENT_RETRY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class SecurityConfig(PydanticBaseSettings):
    """Configuration for security settings."""
    
    # API Security
    api_key_required: bool = Field(False, env="API_KEY_REQUIRED")
    api_key: Optional[str] = Field(None, env="API_KEY")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(60, env="RATE_LIMIT_RPM")
    rate_limit_burst: int = Field(10, env="RATE_LIMIT_BURST")
    
    # Data Protection
    encrypt_sensitive_data: bool = Field(True, env="ENCRYPT_SENSITIVE_DATA")
    data_retention_days: int = Field(90, env="DATA_RETENTION_DAYS")
    
    # Proxy Configuration
    proxy_rotation_enabled: bool = Field(False, env="PROXY_ROTATION_ENABLED")
    proxy_list: List[str] = Field(default_factory=list, env="PROXY_LIST")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class EnhancedSystemConfig:
    """Main configuration class that combines all settings."""
    
    def __init__(self):
        """Initialize all configuration sections."""
        self.langchain = LangChainConfig()
        self.redis = RedisConfig()
        self.database = DatabaseConfig()
        self.monitoring = MonitoringConfig()
        self.agent_system = AgentSystemConfig()
        self.security = SecurityConfig()
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for a specific agent type."""
        base_config = {
            "timeout": self.agent_system.default_timeout,
            "max_retry_attempts": self.agent_system.max_retry_attempts,
            "enable_memory": self.langchain.enable_memory,
            "memory_window_size": self.langchain.memory_window_size,
        }
        
        # Agent-specific configurations
        agent_configs = {
            "nlu_agent": {
                "model": self.langchain.openai_model,
                "temperature": self.langchain.openai_temperature,
                "max_tokens": self.langchain.openai_max_tokens,
            },
            "planning_agent": {
                "model": self.langchain.openai_model,
                "temperature": 0.2,  # Slightly higher for creative planning
                "max_tokens": self.langchain.openai_max_tokens,
            },
            "scraper_agent": {
                "use_proxy": self.security.proxy_rotation_enabled,
                "proxy_list": self.security.proxy_list,
                "rate_limit": self.security.rate_limit_requests_per_minute,
            }
        }
        
        base_config.update(agent_configs.get(agent_type, {}))
        return base_config
    
    def get_llm_config(self, provider: str = "openai") -> Dict[str, Any]:
        """Get LLM configuration for the specified provider."""
        if provider == "openai":
            return {
                "api_key": self.langchain.openai_api_key,
                "model": self.langchain.openai_model,
                "temperature": self.langchain.openai_temperature,
                "max_tokens": self.langchain.openai_max_tokens,
            }
        elif provider == "anthropic":
            return {
                "api_key": self.langchain.anthropic_api_key,
                "model": self.langchain.anthropic_model,
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def validate_configuration(self) -> List[str]:
        """Validate the configuration and return any issues."""
        issues = []
        
        # Check required API keys
        if not self.langchain.openai_api_key and not self.langchain.anthropic_api_key:
            issues.append("No LLM API key configured (OpenAI or Anthropic required)")
        
        # Check Redis configuration
        if not self.redis.redis_host:
            issues.append("Redis host not configured")
        
        # Check database configuration
        if not self.database.postgres_host:
            issues.append("PostgreSQL host not configured")
        
        # Check security settings
        if self.security.api_key_required and not self.security.api_key:
            issues.append("API key required but not configured")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "langchain": self.langchain.dict(),
            "redis": self.redis.dict(),
            "database": self.database.dict(),
            "monitoring": self.monitoring.dict(),
            "agent_system": self.agent_system.dict(),
            "security": self.security.dict(),
        }


# Global configuration instance
config = EnhancedSystemConfig()


def get_config() -> EnhancedSystemConfig:
    """Get the global configuration instance."""
    return config


def validate_system_config() -> bool:
    """Validate the system configuration."""
    issues = config.validate_configuration()
    
    if issues:
        print("⚠️  Configuration Issues Found:")
        for issue in issues:
            print(f"   • {issue}")
        print("\nPlease check your environment variables or .env file.")
        return False
    
    print("✅ Configuration validation passed!")
    return True
