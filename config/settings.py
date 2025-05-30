"""
Settings module for the web scraping system.
"""
import os
import logging
from typing import Dict, Any, Optional, List
import yaml
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class Settings:
    """
    Settings class for the web scraping system.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings.
        
        Args:
            config_path: Path to the configuration file. If None, default settings are used.
        """
        # Default settings
        self.defaults = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None
            },
            "scraper": {
                "user_agents": [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
                ],
                "default_timeout": 30,
                "max_retries": 3,
                "retry_delay": 2.0,
                "respect_robots_txt": True,
                "default_headers": {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Cache-Control": "max-age=0"
                }
            },
            "parser": {
                "default_parser": "html.parser",
                "normalize_whitespace": True,
                "extract_metadata": True
            },
            "storage": {
                "output_dir": "output",
                "default_format": "json",
                "pretty_json": True,
                "csv_delimiter": ",",
                "excel_engine": "openpyxl"
            },
            "proxy": {
                "enabled": False,
                "rotation_enabled": True,
                "proxy_list_path": None,
                "check_interval": 600
            },
            "rate_limiting": {
                "enabled": True,
                "default_rate": 1,
                "default_period": 2.0,
                "adaptive": True
            }
        }
        
        # Load settings from config file if provided
        self.config = self.defaults.copy()
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Override with environment variables
        self._load_from_env()
        
        # Set up logging
        self._setup_logging()
    
    def _load_from_file(self, config_path: str) -> None:
        """
        Load settings from a configuration file.
        
        Args:
            config_path: Path to the configuration file.
        """
        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
            
            # Update config with file settings
            self._update_nested_dict(self.config, file_config)
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {str(e)}")
    
    def _load_from_env(self) -> None:
        """Load settings from environment variables."""
        # Mapping of environment variables to config keys
        env_mapping = {
            "SCRAPER_TIMEOUT": ("scraper", "default_timeout"),
            "SCRAPER_MAX_RETRIES": ("scraper", "max_retries"),
            "SCRAPER_RETRY_DELAY": ("scraper", "retry_delay"),
            "SCRAPER_RESPECT_ROBOTS": ("scraper", "respect_robots_txt"),
            "PARSER_DEFAULT_PARSER": ("parser", "default_parser"),
            "STORAGE_OUTPUT_DIR": ("storage", "output_dir"),
            "STORAGE_DEFAULT_FORMAT": ("storage", "default_format"),
            "PROXY_ENABLED": ("proxy", "enabled"),
            "PROXY_LIST_PATH": ("proxy", "proxy_list_path"),
            "RATE_LIMITING_ENABLED": ("rate_limiting", "enabled"),
            "RATE_LIMITING_DEFAULT_RATE": ("rate_limiting", "default_rate"),
            "LOGGING_LEVEL": ("logging", "level"),
            "LOGGING_FILE": ("logging", "file")
        }
        
        for env_var, config_path in env_mapping.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert value to appropriate type
                if env_value.lower() in ["true", "false"]:
                    # Boolean
                    env_value = env_value.lower() == "true"
                elif env_value.isdigit():
                    # Integer
                    env_value = int(env_value)
                elif env_value.replace(".", "", 1).isdigit():
                    # Float
                    env_value = float(env_value)
                
                # Update config
                self._set_nested_value(self.config, config_path, env_value)
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """
        Update a nested dictionary with values from another dictionary.
        
        Args:
            d: Dictionary to update.
            u: Dictionary with new values.
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def _set_nested_value(self, d: Dict[str, Any], path: tuple, value: Any) -> None:
        """
        Set a value in a nested dictionary.
        
        Args:
            d: Dictionary to update.
            path: Tuple of keys forming the path to the value.
            value: Value to set.
        """
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value
    
    def _setup_logging(self) -> None:
        """Set up logging based on configuration."""
        log_config = self.config["logging"]
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config["level"].upper(), logging.INFO),
            format=log_config["format"],
            filename=log_config["file"]
        )
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a value from the configuration.
        
        Args:
            *keys: Keys forming the path to the value.
            default: Default value to return if the key is not found.
            
        Returns:
            The value at the specified path, or the default value if not found.
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys_and_value: Any) -> None:
        """
        Set a value in the configuration.
        
        Args:
            *keys_and_value: Keys forming the path to the value, followed by the value.
        """
        if len(keys_and_value) < 2:
            raise ValueError("At least one key and a value are required")
        
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]
        
        self._set_nested_value(self.config, keys, value)
    
    def save(self, path: str) -> None:
        """
        Save the configuration to a file.
        
        Args:
            path: Path to save the configuration to.
        """
        try:
            with open(path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            logging.error(f"Error saving configuration to {path}: {str(e)}")
    
    def reset(self) -> None:
        """Reset the configuration to default values."""
        self.config = self.defaults.copy()
        self._setup_logging()


# Create a global settings instance
settings = Settings()
