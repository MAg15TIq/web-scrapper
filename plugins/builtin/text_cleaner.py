"""
Text Cleaner Plugin
Advanced text cleaning and normalization for scraped content.
"""

import re
import html
import unicodedata
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from plugins.base_plugin import BasePlugin, PluginMetadata, PluginType, PluginResult


class TextCleanerPlugin(BasePlugin):
    """
    Advanced text cleaning plugin with multiple cleaning operations.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="text-cleaner",
            version="1.0.0",
            description="Advanced text cleaning and normalization",
            author="WebScraper Team",
            plugin_type=PluginType.PROCESSOR,
            tags=["text", "cleaning", "normalization", "preprocessing"],
            config_schema={
                "remove_html": {"type": "boolean", "default": True},
                "decode_entities": {"type": "boolean", "default": True},
                "normalize_whitespace": {"type": "boolean", "default": True},
                "remove_extra_spaces": {"type": "boolean", "default": True},
                "strip_text": {"type": "boolean", "default": True},
                "normalize_unicode": {"type": "boolean", "default": True},
                "remove_urls": {"type": "boolean", "default": False},
                "remove_emails": {"type": "boolean", "default": False},
                "remove_phone_numbers": {"type": "boolean", "default": False},
                "convert_to_lowercase": {"type": "boolean", "default": False},
                "remove_punctuation": {"type": "boolean", "default": False},
                "remove_numbers": {"type": "boolean", "default": False},
                "custom_patterns": {"type": "array", "default": []},
                "min_length": {"type": "number", "default": 0},
                "max_length": {"type": "number", "default": 0}
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize the text cleaner plugin."""
        try:
            self.logger.info("Initializing Text Cleaner Plugin")
            
            # Load configuration
            self.remove_html = self.get_config_value("remove_html", True)
            self.decode_entities = self.get_config_value("decode_entities", True)
            self.normalize_whitespace = self.get_config_value("normalize_whitespace", True)
            self.remove_extra_spaces = self.get_config_value("remove_extra_spaces", True)
            self.strip_text = self.get_config_value("strip_text", True)
            self.normalize_unicode = self.get_config_value("normalize_unicode", True)
            self.remove_urls = self.get_config_value("remove_urls", False)
            self.remove_emails = self.get_config_value("remove_emails", False)
            self.remove_phone_numbers = self.get_config_value("remove_phone_numbers", False)
            self.convert_to_lowercase = self.get_config_value("convert_to_lowercase", False)
            self.remove_punctuation = self.get_config_value("remove_punctuation", False)
            self.remove_numbers = self.get_config_value("remove_numbers", False)
            self.custom_patterns = self.get_config_value("custom_patterns", [])
            self.min_length = self.get_config_value("min_length", 0)
            self.max_length = self.get_config_value("max_length", 0)
            
            # Compile regex patterns
            self._compile_patterns()
            
            self._initialized = True
            self._running = True
            
            self.logger.info("Text Cleaner Plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Text Cleaner Plugin: {e}")
            return False
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.phone_pattern = re.compile(
            r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        )
        self.html_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.number_pattern = re.compile(r'\d+')
        
        # Compile custom patterns
        self.custom_compiled_patterns = []
        for pattern in self.custom_patterns:
            try:
                self.custom_compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                self.logger.warning(f"Invalid custom pattern '{pattern}': {e}")
    
    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> PluginResult:
        """
        Clean and normalize text data.
        
        Args:
            input_data: Text data to clean (string, list of strings, or dict with text fields)
            context: Optional execution context
            
        Returns:
            PluginResult with cleaned text
        """
        try:
            self.logger.info("Executing text cleaning")
            
            # Process different input types
            if isinstance(input_data, str):
                cleaned_text = self._clean_text(input_data)
                result_data = cleaned_text
                
            elif isinstance(input_data, list):
                cleaned_texts = []
                for text in input_data:
                    if isinstance(text, str):
                        cleaned_texts.append(self._clean_text(text))
                    else:
                        cleaned_texts.append(text)  # Keep non-string items as-is
                result_data = cleaned_texts
                
            elif isinstance(input_data, dict):
                cleaned_dict = {}
                for key, value in input_data.items():
                    if isinstance(value, str):
                        cleaned_dict[key] = self._clean_text(value)
                    elif isinstance(value, list):
                        cleaned_list = []
                        for item in value:
                            if isinstance(item, str):
                                cleaned_list.append(self._clean_text(item))
                            else:
                                cleaned_list.append(item)
                        cleaned_dict[key] = cleaned_list
                    else:
                        cleaned_dict[key] = value
                result_data = cleaned_dict
                
            else:
                return PluginResult(
                    success=False,
                    error="Input data must be string, list, or dictionary"
                )
            
            # Generate metadata
            metadata = {
                "operations_applied": self._get_applied_operations(),
                "processing_timestamp": datetime.now().isoformat(),
                "input_type": type(input_data).__name__
            }
            
            self.logger.info("Text cleaning completed successfully")
            
            return PluginResult(
                success=True,
                data=result_data,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Text cleaning failed: {e}")
            return PluginResult(
                success=False,
                error=str(e)
            )
    
    def _clean_text(self, text: str) -> str:
        """Apply all cleaning operations to a single text string."""
        if not isinstance(text, str):
            return text
        
        original_text = text
        
        # HTML entity decoding
        if self.decode_entities:
            text = html.unescape(text)
        
        # HTML tag removal
        if self.remove_html:
            text = self.html_pattern.sub('', text)
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
        
        # URL removal
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Email removal
        if self.remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Phone number removal
        if self.remove_phone_numbers:
            text = self.phone_pattern.sub('', text)
        
        # Custom pattern removal
        for pattern in self.custom_compiled_patterns:
            text = pattern.sub('', text)
        
        # Whitespace normalization
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
        
        # Extra space removal
        if self.remove_extra_spaces:
            text = ' '.join(text.split())
        
        # Strip whitespace
        if self.strip_text:
            text = text.strip()
        
        # Case conversion
        if self.convert_to_lowercase:
            text = text.lower()
        
        # Punctuation removal
        if self.remove_punctuation:
            text = self.punctuation_pattern.sub('', text)
        
        # Number removal
        if self.remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # Length filtering
        if self.min_length > 0 and len(text) < self.min_length:
            return ""
        
        if self.max_length > 0 and len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def _get_applied_operations(self) -> List[str]:
        """Get list of operations that are currently enabled."""
        operations = []
        
        if self.decode_entities:
            operations.append("decode_entities")
        if self.remove_html:
            operations.append("remove_html")
        if self.normalize_unicode:
            operations.append("normalize_unicode")
        if self.remove_urls:
            operations.append("remove_urls")
        if self.remove_emails:
            operations.append("remove_emails")
        if self.remove_phone_numbers:
            operations.append("remove_phone_numbers")
        if self.custom_patterns:
            operations.append("custom_patterns")
        if self.normalize_whitespace:
            operations.append("normalize_whitespace")
        if self.remove_extra_spaces:
            operations.append("remove_extra_spaces")
        if self.strip_text:
            operations.append("strip_text")
        if self.convert_to_lowercase:
            operations.append("convert_to_lowercase")
        if self.remove_punctuation:
            operations.append("remove_punctuation")
        if self.remove_numbers:
            operations.append("remove_numbers")
        if self.min_length > 0 or self.max_length > 0:
            operations.append("length_filtering")
        
        return operations
    
    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """Get statistics about the cleaning operation."""
        return {
            "original_length": len(original_text),
            "cleaned_length": len(cleaned_text),
            "characters_removed": len(original_text) - len(cleaned_text),
            "reduction_percentage": ((len(original_text) - len(cleaned_text)) / len(original_text) * 100) if original_text else 0,
            "original_word_count": len(original_text.split()),
            "cleaned_word_count": len(cleaned_text.split())
        }
