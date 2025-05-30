"""
Natural Language Understanding (NLU) Agent for parsing user requests.
This agent converts natural language requests into structured scraping tasks.
"""
import asyncio
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.langchain_base import EnhancedAgent
from models.langchain_models import (
    AgentConfig, AgentType, TaskRequest, TaskResponse,
    ScrapingRequest, ActionType, OutputFormat, Priority
)


class EntityExtractionResult(BaseModel):
    """Result of entity extraction from natural language."""
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent: str = Field(default="scrape")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    parsed_request: Optional[ScrapingRequest] = None


@tool
def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from natural language text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)

    # Also look for domain names without protocol
    domain_pattern = r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'
    domains = re.findall(domain_pattern, text)

    # Add protocol to domains if not present
    for domain in domains:
        if not any(domain in url for url in urls):
            urls.append(f"https://{domain}")

    return urls


@tool
def identify_data_points(text: str) -> List[str]:
    """Identify data points to extract from natural language description."""
    # Common data point patterns
    data_patterns = {
        'price': r'\b(?:price|cost|amount|fee|charge)\b',
        'title': r'\b(?:title|name|heading|product name)\b',
        'description': r'\b(?:description|details|summary|info)\b',
        'rating': r'\b(?:rating|score|stars|review)\b',
        'availability': r'\b(?:availability|stock|in stock|available)\b',
        'image': r'\b(?:image|photo|picture|img)\b',
        'link': r'\b(?:link|url|href)\b',
        'date': r'\b(?:date|time|timestamp|published)\b',
        'author': r'\b(?:author|writer|creator|by)\b',
        'category': r'\b(?:category|type|genre|section)\b'
    }

    identified_points = []
    text_lower = text.lower()

    for data_type, pattern in data_patterns.items():
        if re.search(pattern, text_lower):
            identified_points.append(data_type)

    # If no specific data points found, suggest common ones
    if not identified_points:
        identified_points = ['title', 'description', 'link']

    return identified_points


@tool
def determine_action_type(text: str) -> str:
    """Determine the type of action requested from natural language."""
    text_lower = text.lower()

    action_patterns = {
        ActionType.SCRAPE: [r'\b(?:scrape|extract|get|fetch|collect|gather)\b'],
        ActionType.MONITOR: [r'\b(?:monitor|watch|track|observe|follow)\b'],
        ActionType.ANALYZE: [r'\b(?:analyze|analyse|study|examine|investigate)\b'],
        ActionType.TRANSFORM: [r'\b(?:transform|convert|change|modify|format)\b'],
        ActionType.VALIDATE: [r'\b(?:validate|verify|check|confirm)\b']
    }

    for action, patterns in action_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return action.value

    return ActionType.SCRAPE.value  # Default action


class NLUAgent(EnhancedAgent):
    """
    Natural Language Understanding Agent that parses user requests
    and converts them into structured scraping tasks.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the NLU Agent.

        Args:
            config: Agent configuration
            llm: Language model for NLU tasks
        """
        if config is None:
            config = AgentConfig(
                agent_id="nlu-agent",
                agent_type=AgentType.NLU_AGENT,
                capabilities=["natural_language_processing", "entity_extraction", "intent_classification"]
            )

        if llm is None:
            # Use OpenAI GPT-4 as default (you can configure this)
            # Note: In production, set your OpenAI API key in environment variables
            try:
                llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.1,  # Low temperature for consistent parsing
                    max_tokens=1000
                )
            except Exception as e:
                # Fallback to None for demonstration (will use basic parsing)
                llm = None
                logging.warning(f"Could not initialize OpenAI LLM: {e}. Using fallback mode.")

        # Define tools for NLU tasks
        tools = [
            extract_urls_from_text,
            identify_data_points,
            determine_action_type
        ]

        # Create prompt template for NLU tasks
        prompt_template = PromptTemplate(
            input_variables=["input", "task_parameters"],
            template="""
You are a Natural Language Understanding agent for a web scraping system.
Your task is to parse natural language requests and extract structured information.

User Request: {input}
Task Parameters: {task_parameters}

Please analyze the request and extract:
1. Target websites or URLs
2. Data points to extract
3. Action type (scrape, monitor, analyze, etc.)
4. Output format preference
5. Priority level
6. Any time constraints

Use the available tools to help with extraction. Provide a clear, structured response.

Available tools:
- extract_urls_from_text: Extract URLs from text
- identify_data_points: Identify what data to extract
- determine_action_type: Determine the type of action requested

Think step by step and use the tools as needed.
"""
        )

        super().__init__(
            config=config,
            llm=llm,
            tools=tools,
            prompt_template=prompt_template
        )

        self.logger.info("NLU Agent initialized with language understanding capabilities")

    async def parse_natural_language_request(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ScrapingRequest:
        """
        Parse a natural language request into a structured ScrapingRequest.

        Args:
            user_input: Natural language input from user
            context: Additional context information

        Returns:
            Structured scraping request
        """
        self.logger.info(f"Parsing natural language request: {user_input[:100]}...")

        try:
            # Create a task request for parsing
            parse_task = TaskRequest(
                task_type="parse_nl_request",
                parameters={
                    "user_input": user_input,
                    "context": context or {}
                }
            )

            # Use LangChain reasoning to parse the request
            response = await self.execute_with_reasoning(parse_task, context)

            if response.status == "completed" and response.result:
                # Extract structured information from the response
                return self._build_scraping_request(user_input, response.result, context)
            else:
                # Fallback to basic parsing
                return await self._parse_request_basic(user_input, context)

        except Exception as e:
            self.logger.error(f"Error parsing natural language request: {e}")
            # Fallback to basic parsing
            return await self._parse_request_basic(user_input, context)

    def _build_scraping_request(
        self,
        user_input: str,
        parsed_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ScrapingRequest:
        """
        Build a ScrapingRequest from parsed results.

        Args:
            user_input: Original user input
            parsed_result: Results from LangChain parsing
            context: Additional context

        Returns:
            Structured scraping request
        """
        # Extract information from parsed result
        data = parsed_result.get("data", {})

        # Extract URLs using the tool
        urls = extract_urls_from_text(user_input)
        if not urls:
            # Try to extract from context or use common sites
            urls = context.get("default_sites", ["https://example.com"]) if context else ["https://example.com"]

        # Extract data points
        data_points = identify_data_points(user_input)

        # Determine action type
        action = determine_action_type(user_input)

        # Determine priority from keywords
        priority = self._determine_priority(user_input)

        # Determine output format
        output_format = self._determine_output_format(user_input)

        return ScrapingRequest(
            product_query=user_input,
            target_sites=urls,
            data_points=data_points,
            action=ActionType(action),
            output_format=output_format,
            priority=priority,
            filters=context.get("filters", {}) if context else {}
        )

    async def _parse_request_basic(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ScrapingRequest:
        """
        Basic parsing fallback when LangChain reasoning fails.

        Args:
            user_input: Natural language input
            context: Additional context

        Returns:
            Basic scraping request
        """
        self.logger.info("Using basic parsing fallback")

        # Use the tools directly for basic parsing
        urls = extract_urls_from_text(user_input)
        data_points = identify_data_points(user_input)
        action = determine_action_type(user_input)

        if not urls:
            urls = ["https://example.com"]  # Default fallback

        return ScrapingRequest(
            product_query=user_input,
            target_sites=urls,
            data_points=data_points,
            action=ActionType(action),
            priority=self._determine_priority(user_input),
            output_format=self._determine_output_format(user_input)
        )

    def _determine_priority(self, text: str) -> Priority:
        """Determine priority from text keywords."""
        text_lower = text.lower()

        if any(word in text_lower for word in ['urgent', 'asap', 'immediately', 'critical']):
            return Priority.CRITICAL
        elif any(word in text_lower for word in ['high', 'important', 'priority']):
            return Priority.HIGH
        elif any(word in text_lower for word in ['low', 'later', 'when possible']):
            return Priority.LOW
        else:
            return Priority.NORMAL

    def _determine_output_format(self, text: str) -> OutputFormat:
        """Determine output format from text."""
        text_lower = text.lower()

        format_keywords = {
            'json': OutputFormat.JSON,
            'csv': OutputFormat.CSV,
            'excel': OutputFormat.EXCEL,
            'pdf': OutputFormat.PDF,
            'html': OutputFormat.HTML,
            'xml': OutputFormat.XML
        }

        for keyword, format_type in format_keywords.items():
            if keyword in text_lower:
                return format_type

        return OutputFormat.JSON  # Default format

    async def execute_task_basic(self, task: TaskRequest) -> TaskResponse:
        """
        Basic task execution for NLU operations.

        Args:
            task: The task to execute

        Returns:
            Task response
        """
        if task.task_type == "parse_nl_request":
            user_input = task.parameters.get("user_input", "")
            context = task.parameters.get("context", {})

            # Perform basic NLU parsing
            result = {
                "urls": extract_urls_from_text(user_input),
                "data_points": identify_data_points(user_input),
                "action": determine_action_type(user_input),
                "priority": self._determine_priority(user_input).value,
                "output_format": self._determine_output_format(user_input).value
            }

            return TaskResponse(
                task_id=task.id,
                status="completed",
                result={"data": result},
                agent_id=self.agent_id
            )
        else:
            return TaskResponse(
                task_id=task.id,
                status="failed",
                error={"message": f"Unknown task type: {task.task_type}"},
                agent_id=self.agent_id
            )
