"""
Configuration Manager for Enhanced CLI
Handles configuration loading, validation, and profile management.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class CLIProfile(BaseModel):
    """CLI profile configuration."""
    
    name: str = Field(..., description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    default_output_format: str = Field("json", description="Default output format")
    default_agents: List[str] = Field(default_factory=list, description="Default agents to use")
    auto_confirm: bool = Field(False, description="Auto-confirm operations")
    verbose_logging: bool = Field(False, description="Enable verbose logging")
    theme: str = Field("default", description="CLI theme")
    max_concurrent_tasks: int = Field(5, description="Maximum concurrent tasks")
    timeout: int = Field(300, description="Default timeout in seconds")
    retry_attempts: int = Field(3, description="Default retry attempts")
    
    @validator('default_output_format')
    def validate_output_format(cls, v):
        """Validate output format."""
        valid_formats = ['json', 'csv', 'excel', 'xml', 'pdf']
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid output format. Must be one of: {valid_formats}")
        return v.lower()
    
    @validator('max_concurrent_tasks')
    def validate_max_concurrent_tasks(cls, v):
        """Validate max concurrent tasks."""
        if v < 1 or v > 50:
            raise ValueError("Max concurrent tasks must be between 1 and 50")
        return v


class AgentConfiguration(BaseModel):
    """Agent-specific configuration."""
    
    agent_type: str = Field(..., description="Agent type")
    enabled: bool = Field(True, description="Whether agent is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")
    priority: int = Field(1, description="Agent priority (1-10)")
    timeout: int = Field(300, description="Agent timeout in seconds")
    retry_policy: Dict[str, Any] = Field(default_factory=dict, description="Retry policy")
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority."""
        if v < 1 or v > 10:
            raise ValueError("Priority must be between 1 and 10")
        return v


class CLIConfiguration(BaseModel):
    """Main CLI configuration."""
    
    version: str = Field("1.0.0", description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # General settings
    default_profile: str = Field("default", description="Default profile name")
    profiles: Dict[str, CLIProfile] = Field(default_factory=dict, description="Available profiles")
    
    # Agent configurations
    agents: Dict[str, AgentConfiguration] = Field(default_factory=dict, description="Agent configurations")
    
    # Logging configuration
    logging: Dict[str, Any] = Field(default_factory=dict, description="Logging configuration")
    
    # Security settings
    security: Dict[str, Any] = Field(default_factory=dict, description="Security settings")
    
    # Performance settings
    performance: Dict[str, Any] = Field(default_factory=dict, description="Performance settings")


class ConfigManager:
    """Configuration manager for the enhanced CLI."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the configuration manager."""
        self.logger = logging.getLogger("config_manager")
        
        # Set configuration directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path.home() / ".webscraper_cli"
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration file paths
        self.config_file = self.config_dir / "config.yaml"
        self.profiles_file = self.config_dir / "profiles.yaml"
        self.agents_file = self.config_dir / "agents.yaml"
        
        # Current configuration
        self.config: Optional[CLIConfiguration] = None
        self.current_profile: Optional[str] = None
        
        self.logger.info(f"Configuration manager initialized with directory: {self.config_dir}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from files."""
        try:
            # Load main configuration
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}
            
            # Load profiles
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r') as f:
                    profiles_data = yaml.safe_load(f) or {}
                config_data['profiles'] = profiles_data
            
            # Load agent configurations
            if self.agents_file.exists():
                with open(self.agents_file, 'r') as f:
                    agents_data = yaml.safe_load(f) or {}
                config_data['agents'] = agents_data
            
            # Create configuration object with defaults
            self.config = CLIConfiguration(**self._merge_with_defaults(config_data))
            
            # Set current profile
            self.current_profile = self.config.default_profile
            
            self.logger.info("Configuration loaded successfully")
            return self.config.dict()
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            self.config = CLIConfiguration(**self._get_default_config())
            return self.config.dict()
    
    def save_config(self) -> bool:
        """Save current configuration to files."""
        try:
            if not self.config:
                self.logger.warning("No configuration to save")
                return False
            
            # Update timestamp
            self.config.updated_at = datetime.now()
            
            # Save main configuration
            main_config = {
                'version': self.config.version,
                'created_at': self.config.created_at.isoformat(),
                'updated_at': self.config.updated_at.isoformat(),
                'default_profile': self.config.default_profile,
                'logging': self.config.logging,
                'security': self.config.security,
                'performance': self.config.performance
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(main_config, f, default_flow_style=False)
            
            # Save profiles
            profiles_data = {name: profile.dict() for name, profile in self.config.profiles.items()}
            with open(self.profiles_file, 'w') as f:
                yaml.dump(profiles_data, f, default_flow_style=False)
            
            # Save agent configurations
            agents_data = {name: agent.dict() for name, agent in self.config.agents.items()}
            with open(self.agents_file, 'w') as f:
                yaml.dump(agents_data, f, default_flow_style=False)
            
            self.logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def load_config_file(self, config_path: str) -> bool:
        """Load configuration from a specific file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                self.logger.error(f"Configuration file not found: {config_path}")
                return False
            
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f)
            
            # Merge with existing configuration
            if self.config:
                current_data = self.config.dict()
                current_data.update(config_data)
                self.config = CLIConfiguration(**current_data)
            else:
                self.config = CLIConfiguration(**self._merge_with_defaults(config_data))
            
            self.logger.info(f"Configuration loaded from: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration file: {e}")
            return False
    
    def get_profile(self, profile_name: Optional[str] = None) -> Optional[CLIProfile]:
        """Get a specific profile or the current profile."""
        if not self.config:
            return None
        
        profile_name = profile_name or self.current_profile
        return self.config.profiles.get(profile_name)
    
    def set_profile(self, profile_name: str) -> bool:
        """Set the current profile."""
        if not self.config or profile_name not in self.config.profiles:
            self.logger.error(f"Profile not found: {profile_name}")
            return False
        
        self.current_profile = profile_name
        self.config.default_profile = profile_name
        self.logger.info(f"Profile set to: {profile_name}")
        return True
    
    def create_profile(self, profile_name: str, profile_data: Dict[str, Any]) -> bool:
        """Create a new profile."""
        try:
            if not self.config:
                self.config = CLIConfiguration(**self._get_default_config())
            
            profile = CLIProfile(name=profile_name, **profile_data)
            self.config.profiles[profile_name] = profile
            
            self.logger.info(f"Profile created: {profile_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating profile: {e}")
            return False
    
    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile."""
        if not self.config or profile_name not in self.config.profiles:
            self.logger.error(f"Profile not found: {profile_name}")
            return False
        
        if profile_name == "default":
            self.logger.error("Cannot delete default profile")
            return False
        
        del self.config.profiles[profile_name]
        
        # If this was the current profile, switch to default
        if self.current_profile == profile_name:
            self.current_profile = "default"
            self.config.default_profile = "default"
        
        self.logger.info(f"Profile deleted: {profile_name}")
        return True
    
    def get_agent_config(self, agent_type: str) -> Optional[AgentConfiguration]:
        """Get configuration for a specific agent."""
        if not self.config:
            return None
        
        return self.config.agents.get(agent_type)
    
    def set_agent_config(self, agent_type: str, config_data: Dict[str, Any]) -> bool:
        """Set configuration for a specific agent."""
        try:
            if not self.config:
                self.config = CLIConfiguration(**self._get_default_config())
            
            agent_config = AgentConfiguration(agent_type=agent_type, **config_data)
            self.config.agents[agent_type] = agent_config
            
            self.logger.info(f"Agent configuration updated: {agent_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting agent configuration: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'version': '1.0.0',
            'default_profile': 'default',
            'profiles': {
                'default': CLIProfile(
                    name='default',
                    description='Default CLI profile',
                    default_output_format='json',
                    default_agents=['Scraper Agent', 'Parser Agent', 'Storage Agent'],
                    theme='default'
                ).dict()
            },
            'agents': {
                'scraper': AgentConfiguration(
                    agent_type='scraper',
                    config={'timeout': 30, 'retries': 3}
                ).dict(),
                'parser': AgentConfiguration(
                    agent_type='parser',
                    config={'normalize_data': True}
                ).dict(),
                'storage': AgentConfiguration(
                    agent_type='storage',
                    config={'default_format': 'json'}
                ).dict()
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/cli.log'
            },
            'security': {
                'enable_auth': False,
                'session_timeout': 3600
            },
            'performance': {
                'max_workers': 5,
                'chunk_size': 1000,
                'cache_enabled': True
            }
        }
    
    def _merge_with_defaults(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration data with defaults."""
        defaults = self._get_default_config()
        
        # Deep merge
        def deep_merge(default: Dict, override: Dict) -> Dict:
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(defaults, config_data)
