"""
Advanced Command Parser with Pydantic validation and intelligent parsing.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from urllib.parse import urlparse
from pydantic import BaseModel, Field, validator, ValidationError

from models.langchain_models import ScrapingRequest, Priority, OutputFormat, ActionType


class ParsedCommand(BaseModel):
    """Parsed command structure with validation."""
    
    action: ActionType = Field(..., description="The action to perform")
    target: Optional[str] = Field(None, description="Target URL or resource")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command parameters")
    agents: List[str] = Field(default_factory=list, description="Required agents")
    priority: Priority = Field(default=Priority.NORMAL, description="Command priority")
    output_format: OutputFormat = Field(default=OutputFormat.JSON, description="Output format")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Data filters")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")
    
    @validator('target')
    def validate_target(cls, v):
        """Validate target URL or resource."""
        if v and v.startswith('http'):
            parsed = urlparse(v)
            if not parsed.netloc:
                raise ValueError("Invalid URL format")
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v):
        """Validate parameters dictionary."""
        if not isinstance(v, dict):
            raise ValueError("Parameters must be a dictionary")
        return v


class CommandPattern(BaseModel):
    """Command pattern for matching natural language inputs."""
    
    pattern: str = Field(..., description="Regex pattern to match")
    action: ActionType = Field(..., description="Action type for this pattern")
    parameter_extractors: Dict[str, str] = Field(default_factory=dict, description="Parameter extraction patterns")
    required_agents: List[str] = Field(default_factory=list, description="Required agents for this command")
    priority: Priority = Field(default=Priority.NORMAL, description="Default priority")


class CommandParser:
    """Advanced command parser with natural language understanding."""
    
    def __init__(self):
        """Initialize the command parser."""
        self.logger = logging.getLogger("command_parser")
        self.patterns = self._initialize_patterns()
        
        # Common parameter patterns
        self.param_patterns = {
            'url': r'(?:from\s+|at\s+)?(?:https?://)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)',
            'pages': r'(?:first\s+)?(\d+)\s+pages?',
            'format': r'(?:as|in|to)\s+(json|csv|excel|xml|pdf)',
            'selectors': r'(?:extract|get|find)\s+([^,]+(?:,\s*[^,]+)*)',
            'schedule': r'(?:every|each)\s+(\d+)\s+(minutes?|hours?|days?)',
            'limit': r'(?:limit|max|maximum)\s+(\d+)',
            'timeout': r'(?:timeout|wait)\s+(\d+)\s*(?:seconds?|mins?|minutes?)?'
        }
        
        self.logger.info("Command parser initialized with patterns")
    
    def _initialize_patterns(self) -> List[CommandPattern]:
        """Initialize command patterns for natural language parsing."""
        patterns = [
            # Scraping patterns
            CommandPattern(
                pattern=r'scrape|extract|get|fetch|collect|harvest',
                action=ActionType.SCRAPE,
                parameter_extractors={
                    'data_type': r'(?:scrape|extract|get|fetch)\s+([^from]+?)(?:\s+from|\s+at|$)',
                    'source': r'(?:from|at)\s+([^\s]+)',
                },
                required_agents=['Scraper Agent', 'Parser Agent', 'Storage Agent']
            ),
            
            # Analysis patterns
            CommandPattern(
                pattern=r'analyze|analyse|process|examine|study',
                action=ActionType.ANALYZE,
                parameter_extractors={
                    'analysis_type': r'(?:analyze|analyse|process|examine)\s+([^from]+?)(?:\s+from|\s+in|$)',
                    'source': r'(?:from|in)\s+([^\s]+)',
                },
                required_agents=['NLP Processing Agent', 'Data Transformation Agent']
            ),
            
            # Monitoring patterns
            CommandPattern(
                pattern=r'monitor|watch|track|observe|check',
                action=ActionType.MONITOR,
                parameter_extractors={
                    'target': r'(?:monitor|watch|track|observe|check)\s+([^for]+?)(?:\s+for|\s+every|$)',
                    'interval': r'(?:every|each)\s+(\d+)\s+(seconds?|minutes?|hours?)',
                },
                required_agents=['Monitoring Agent', 'Alert Agent']
            ),
            
            # Configuration patterns
            CommandPattern(
                pattern=r'configure|config|setup|set|change',
                action=ActionType.CONFIGURE,
                parameter_extractors={
                    'component': r'(?:configure|config|setup|set|change)\s+([^to]+?)(?:\s+to|\s+as|$)',
                    'value': r'(?:to|as)\s+([^\s]+)',
                },
                required_agents=['Configuration Agent']
            ),
            
            # Export patterns
            CommandPattern(
                pattern=r'export|save|download|output',
                action=ActionType.EXPORT,
                parameter_extractors={
                    'format': r'(?:export|save|download|output)\s+(?:as\s+|to\s+)?([a-zA-Z]+)',
                    'destination': r'(?:to|in)\s+([^\s]+)',
                },
                required_agents=['Storage Agent', 'Export Agent']
            ),
            
            # Schedule patterns
            CommandPattern(
                pattern=r'schedule|automate|repeat|recurring',
                action=ActionType.SCHEDULE,
                parameter_extractors={
                    'task': r'(?:schedule|automate|repeat)\s+([^every]+?)(?:\s+every|\s+at|$)',
                    'frequency': r'(?:every|each)\s+([^at]+?)(?:\s+at|$)',
                    'time': r'(?:at)\s+([^\s]+)',
                },
                required_agents=['Scheduler Agent', 'Task Manager']
            )
        ]
        
        return patterns
    
    def parse_command(self, command_text: str) -> Optional[ParsedCommand]:
        """Parse a natural language command into a structured format."""
        try:
            command_text = command_text.strip().lower()
            
            if not command_text:
                return None
            
            self.logger.debug(f"Parsing command: {command_text}")
            
            # Find matching pattern
            matched_pattern = self._find_matching_pattern(command_text)
            
            if not matched_pattern:
                self.logger.warning(f"No pattern matched for command: {command_text}")
                return None
            
            # Extract parameters
            parameters = self._extract_parameters(command_text, matched_pattern)
            
            # Extract common parameters
            common_params = self._extract_common_parameters(command_text)
            parameters.update(common_params)
            
            # Determine target
            target = self._extract_target(command_text, parameters)
            
            # Create parsed command
            parsed_command = ParsedCommand(
                action=matched_pattern.action,
                target=target,
                parameters=parameters,
                agents=matched_pattern.required_agents,
                priority=matched_pattern.priority,
                output_format=self._determine_output_format(command_text),
                filters=self._extract_filters(command_text),
                options=self._extract_options(command_text)
            )
            
            self.logger.info(f"Successfully parsed command: {matched_pattern.action}")
            return parsed_command
            
        except ValidationError as e:
            self.logger.error(f"Validation error parsing command: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing command: {e}")
            return None
    
    def _find_matching_pattern(self, command_text: str) -> Optional[CommandPattern]:
        """Find the best matching pattern for the command."""
        for pattern in self.patterns:
            if re.search(pattern.pattern, command_text, re.IGNORECASE):
                return pattern
        return None
    
    def _extract_parameters(self, command_text: str, pattern: CommandPattern) -> Dict[str, Any]:
        """Extract parameters using the pattern's extractors."""
        parameters = {}
        
        for param_name, extractor_pattern in pattern.parameter_extractors.items():
            match = re.search(extractor_pattern, command_text, re.IGNORECASE)
            if match:
                parameters[param_name] = match.group(1).strip()
        
        return parameters
    
    def _extract_common_parameters(self, command_text: str) -> Dict[str, Any]:
        """Extract common parameters from the command text."""
        parameters = {}
        
        for param_name, pattern in self.param_patterns.items():
            match = re.search(pattern, command_text, re.IGNORECASE)
            if match:
                if param_name == 'pages':
                    parameters['max_pages'] = int(match.group(1))
                elif param_name == 'format':
                    parameters['output_format'] = match.group(1).upper()
                elif param_name == 'selectors':
                    selectors = [s.strip() for s in match.group(1).split(',')]
                    parameters['selectors'] = selectors
                elif param_name == 'schedule':
                    parameters['schedule_interval'] = int(match.group(1))
                    parameters['schedule_unit'] = match.group(2)
                elif param_name == 'limit':
                    parameters['limit'] = int(match.group(1))
                elif param_name == 'timeout':
                    parameters['timeout'] = int(match.group(1))
                else:
                    parameters[param_name] = match.group(1)
        
        return parameters
    
    def _extract_target(self, command_text: str, parameters: Dict[str, Any]) -> Optional[str]:
        """Extract the target URL or resource."""
        # Check if URL is in parameters
        if 'url' in parameters:
            return parameters['url']
        
        if 'source' in parameters:
            return parameters['source']
        
        # Look for URL patterns
        url_pattern = r'(?:https?://)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)'
        match = re.search(url_pattern, command_text)
        if match:
            url = match.group(0)
            if not url.startswith('http'):
                url = 'https://' + url
            return url
        
        return None
    
    def _determine_output_format(self, command_text: str) -> OutputFormat:
        """Determine the desired output format."""
        format_patterns = {
            'json': r'\bjson\b',
            'csv': r'\bcsv\b',
            'excel': r'\bexcel\b|\bxlsx?\b',
            'xml': r'\bxml\b',
            'pdf': r'\bpdf\b'
        }
        
        for format_name, pattern in format_patterns.items():
            if re.search(pattern, command_text, re.IGNORECASE):
                return OutputFormat(format_name.upper())
        
        return OutputFormat.JSON
    
    def _extract_filters(self, command_text: str) -> Dict[str, Any]:
        """Extract data filters from the command."""
        filters = {}
        
        # Price filters
        price_pattern = r'(?:price|cost)\s+(?:between|from)\s+(\d+)\s+(?:to|and)\s+(\d+)'
        price_match = re.search(price_pattern, command_text, re.IGNORECASE)
        if price_match:
            filters['price_min'] = int(price_match.group(1))
            filters['price_max'] = int(price_match.group(2))
        
        # Date filters
        date_pattern = r'(?:from|since)\s+(\d{4}-\d{2}-\d{2})'
        date_match = re.search(date_pattern, command_text)
        if date_match:
            filters['date_from'] = date_match.group(1)
        
        # Category filters
        category_pattern = r'(?:category|type)\s+([a-zA-Z\s]+)'
        category_match = re.search(category_pattern, command_text, re.IGNORECASE)
        if category_match:
            filters['category'] = category_match.group(1).strip()
        
        return filters
    
    def _extract_options(self, command_text: str) -> Dict[str, Any]:
        """Extract additional options from the command."""
        options = {}
        
        # JavaScript rendering
        if re.search(r'\bjavascript\b|\bjs\b|\brender\b', command_text, re.IGNORECASE):
            options['render_js'] = True
        
        # Anti-detection
        if re.search(r'stealth|anti.?detect|avoid.?block', command_text, re.IGNORECASE):
            options['anti_detection'] = True
        
        # Parallel processing
        if re.search(r'parallel|concurrent|simultaneous', command_text, re.IGNORECASE):
            options['parallel'] = True
        
        # Clean data
        if re.search(r'clean|normalize|sanitize', command_text, re.IGNORECASE):
            options['clean_data'] = True
        
        return options
    
    def validate_command(self, parsed_command: ParsedCommand) -> bool:
        """Validate a parsed command for completeness and correctness."""
        try:
            # Check if action requires a target
            if parsed_command.action in [ActionType.SCRAPE, ActionType.ANALYZE] and not parsed_command.target:
                self.logger.warning("Scrape/Analyze action requires a target")
                return False
            
            # Check if required parameters are present
            if parsed_command.action == ActionType.SCRAPE:
                if not parsed_command.target and not parsed_command.parameters.get('source'):
                    self.logger.warning("Scrape action requires a target or source")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating command: {e}")
            return False
    
    def get_command_suggestions(self, partial_command: str) -> List[str]:
        """Get command suggestions based on partial input."""
        suggestions = []
        
        # Basic action suggestions
        actions = ['scrape', 'extract', 'analyze', 'monitor', 'configure', 'export', 'schedule']
        
        for action in actions:
            if action.startswith(partial_command.lower()):
                suggestions.append(action)
        
        # Add common command templates
        templates = [
            "scrape products from amazon.com",
            "extract news articles from techcrunch.com",
            "analyze sentiment from reviews",
            "monitor price changes every hour",
            "configure anti-detection settings",
            "export data as CSV",
            "schedule daily scraping"
        ]
        
        for template in templates:
            if partial_command.lower() in template:
                suggestions.append(template)
        
        return suggestions[:5]  # Return top 5 suggestions
