"""
Master Intelligence Agent (MIA) for the self-aware web scraping system.

This agent serves as the central decision-making unit that orchestrates all operations
with self-awareness capabilities.
"""
import asyncio
import logging
import time
import re
import json
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import uuid
from urllib.parse import urlparse

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.intelligence import (
    ContentType, InputType, WebsiteType, AgentCapability,
    AgentProfile, ContentAnalysisResult, AgentAssignment,
    IntelligenceDecision, LearningEvent, AgentSelectionCriteria
)


class MasterIntelligenceAgent(Agent):
    """
    Master Intelligence Agent (MIA) that serves as the central decision-making unit
    for the self-aware web scraping system.
    """
    def __init__(self, agent_id: str = "master_intelligence"):
        """
        Initialize a new Master Intelligence Agent.

        Args:
            agent_id: Unique identifier for the agent.
        """
        super().__init__(agent_id=agent_id, agent_type="master_intelligence", coordinator_id=None)

        # Agent profiles - stores information about all agents in the system
        self.agent_profiles: Dict[str, AgentProfile] = {}

        # Decision history - stores past decisions for learning
        self.decision_history: List[IntelligenceDecision] = []

        # Learning events - stores events for self-learning
        self.learning_events: List[LearningEvent] = []

        # Content type detection patterns
        self.content_patterns: Dict[ContentType, List[str]] = self._initialize_content_patterns()

        # Website type detection patterns
        self.website_patterns: Dict[WebsiteType, List[str]] = self._initialize_website_patterns()

        # Register message handlers
        self.register_handler("analyze_input", self._handle_analyze_input)
        self.register_handler("select_agents", self._handle_select_agents)
        self.register_handler("update_agent_profile", self._handle_update_agent_profile)
        self.register_handler("learning_event", self._handle_learning_event)

        # Start periodic tasks
        self._start_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the Master Intelligence Agent."""
        asyncio.create_task(self._periodic_agent_profile_update())
        asyncio.create_task(self._periodic_learning_analysis())

    async def _periodic_agent_profile_update(self) -> None:
        """Periodically update agent profiles based on performance."""
        while self.running:
            self.logger.debug("Running periodic agent profile update")
            # Update agent profiles based on recent performance
            for agent_id, profile in self.agent_profiles.items():
                # This is a placeholder for actual profile updating logic
                # In a real implementation, you would analyze recent performance
                # and update the profile accordingly
                pass

            # Sleep for 5 minutes
            await asyncio.sleep(300)

    async def _periodic_learning_analysis(self) -> None:
        """Periodically analyze learning events to improve decision making."""
        while self.running:
            self.logger.debug("Running periodic learning analysis")
            # Analyze learning events to improve decision making
            # This is a placeholder for actual learning analysis logic

            # Sleep for 15 minutes
            await asyncio.sleep(900)

    def _initialize_content_patterns(self) -> Dict[ContentType, List[str]]:
        """Initialize patterns for content type detection."""
        return {
            ContentType.HTML: ["<!DOCTYPE html>", "<html", "<body"],
            ContentType.JSON: ["{", "["],
            ContentType.XML: ["<?xml", "<"],
            ContentType.PDF: ["%PDF-"],
            ContentType.DOC: ["\xD0\xCF\x11\xE0"],
            ContentType.DOCX: ["PK"],
            ContentType.XLS: ["\xD0\xCF\x11\xE0"],
            ContentType.XLSX: ["PK"],
            ContentType.CSV: [",", ";"],
            ContentType.TXT: [],  # No specific pattern, fallback
        }

    def _initialize_website_patterns(self) -> Dict[WebsiteType, List[str]]:
        """Initialize patterns for website type detection."""
        return {
            WebsiteType.ECOMMERCE: ["cart", "checkout", "product", "shop", "store", "price"],
            WebsiteType.NEWS: ["news", "article", "headline", "reporter", "editor"],
            WebsiteType.BLOG: ["blog", "post", "author", "comment"],
            WebsiteType.SOCIAL_MEDIA: ["profile", "friend", "follow", "like", "share"],
            WebsiteType.FORUM: ["forum", "thread", "post", "reply", "topic"],
            WebsiteType.GOVERNMENT: ["government", "official", "agency", "department"],
            WebsiteType.ACADEMIC: ["university", "college", "academic", "research", "study"],
            WebsiteType.CORPORATE: ["company", "corporate", "business", "enterprise"],
        }

    async def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the Master Intelligence Agent.

        Args:
            agent: The agent to register.
        """
        self.logger.info(f"Registering agent: {agent.agent_id} (type: {agent.agent_type})")

        # Create a profile for the agent
        profile = AgentProfile(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            capabilities=self._infer_agent_capabilities(agent.agent_type),
            last_updated=time.time()
        )

        # Store the profile
        self.agent_profiles[agent.agent_id] = profile

    def _infer_agent_capabilities(self, agent_type: str) -> Set[AgentCapability]:
        """
        Infer agent capabilities based on agent type.

        Args:
            agent_type: The type of the agent.

        Returns:
            A set of agent capabilities.
        """
        capabilities = set()

        # Map agent types to capabilities
        type_to_capabilities = {
            "scraper": {AgentCapability.HTML_PARSING},
            "parser": {AgentCapability.HTML_PARSING, AgentCapability.JSON_PARSING, AgentCapability.XML_PARSING},
            "storage": {AgentCapability.DATA_STORAGE},
            "javascript": {AgentCapability.JAVASCRIPT_RENDERING},
            "authentication": {AgentCapability.AUTHENTICATION},
            "anti_detection": {AgentCapability.ANTI_DETECTION, AgentCapability.PROXY_MANAGEMENT},
            "data_transformation": {AgentCapability.DATA_TRANSFORMATION},
            "nlp_processing": {AgentCapability.NLP_PROCESSING},
            "image_processing": {AgentCapability.IMAGE_PROCESSING},
            "api_integration": {AgentCapability.API_INTEGRATION},
            "error_recovery": {AgentCapability.ERROR_RECOVERY},
            "data_extractor": {
                AgentCapability.HTML_PARSING,
                AgentCapability.JSON_PARSING,
                AgentCapability.XML_PARSING,
                AgentCapability.PDF_PROCESSING
            },
            "url_intelligence": {AgentCapability.URL_INTELLIGENCE},
            "content_recognition": {AgentCapability.CONTENT_RECOGNITION},
            "document_intelligence": {
                AgentCapability.PDF_PROCESSING,
                AgentCapability.DOCUMENT_PROCESSING
            },
            "quality_assurance": {AgentCapability.QUALITY_ASSURANCE, AgentCapability.DATA_VALIDATION},
            "performance_optimization": {AgentCapability.PERFORMANCE_OPTIMIZATION},
        }

        # Add capabilities based on agent type
        if agent_type in type_to_capabilities:
            capabilities.update(type_to_capabilities[agent_type])

        return capabilities

    async def analyze_input(self, input_data: Union[str, Dict[str, Any]]) -> ContentAnalysisResult:
        """
        Analyze input data to determine its type and characteristics.

        Args:
            input_data: The input data to analyze. Can be a URL, file path, or raw content.

        Returns:
            A ContentAnalysisResult object containing the analysis results.
        """
        self.logger.info(f"Analyzing input: {input_data[:100] if isinstance(input_data, str) else 'dict'}")

        # Determine input type
        input_type = self._determine_input_type(input_data)

        # Determine content type
        content_type = self._determine_content_type(input_data)

        # Determine website type if applicable
        website_type = None
        if input_type == InputType.URL:
            website_type = self._determine_website_type(input_data)

        # Create analysis result
        result = ContentAnalysisResult(
            content_type=content_type,
            input_type=input_type,
            website_type=website_type,
            complexity=self._calculate_complexity(input_data, content_type),
            requires_javascript=self._requires_javascript(input_data, content_type),
            requires_authentication=self._requires_authentication(input_data, input_type),
            has_anti_bot=self._has_anti_bot(input_data, input_type),
            has_pagination=self._has_pagination(input_data, content_type),
            estimated_pages=self._estimate_pages(input_data, content_type),
            language=self._detect_language(input_data, content_type),
            confidence=0.8,  # Placeholder confidence value
            metadata=self._extract_metadata(input_data, content_type, input_type)
        )

        self.logger.info(f"Input analysis result: {result}")
        return result

    def _determine_input_type(self, input_data: Union[str, Dict[str, Any]]) -> InputType:
        """
        Determine the type of input data.

        Args:
            input_data: The input data to analyze.

        Returns:
            The input type.
        """
        if isinstance(input_data, dict):
            # Check if it's an API configuration
            if "api_url" in input_data or "endpoint" in input_data:
                return InputType.API
            # Check if it's a database configuration
            elif "connection_string" in input_data or "database" in input_data:
                return InputType.DATABASE
            else:
                return InputType.UNKNOWN

        # Check if it's a URL
        if isinstance(input_data, str):
            if input_data.startswith(("http://", "https://")):
                return InputType.URL

            # Check if it's a file path
            if "/" in input_data or "\\" in input_data:
                return InputType.FILE

            # Otherwise, assume it's raw text
            return InputType.TEXT

        return InputType.UNKNOWN

    def _determine_content_type(self, input_data: Union[str, Dict[str, Any]]) -> ContentType:
        """
        Determine the content type of the input data.

        Args:
            input_data: The input data to analyze.

        Returns:
            The content type.
        """
        if isinstance(input_data, dict):
            return ContentType.JSON

        if not isinstance(input_data, str):
            return ContentType.UNKNOWN

        # Check for URL file extensions
        if input_data.startswith(("http://", "https://")):
            url_path = urlparse(input_data).path.lower()
            if url_path.endswith(".html") or url_path.endswith(".htm"):
                return ContentType.HTML
            elif url_path.endswith(".json"):
                return ContentType.JSON
            elif url_path.endswith(".xml"):
                return ContentType.XML
            elif url_path.endswith(".pdf"):
                return ContentType.PDF
            elif url_path.endswith((".doc", ".docx")):
                return ContentType.DOC
            elif url_path.endswith((".xls", ".xlsx")):
                return ContentType.XLS
            elif url_path.endswith(".csv"):
                return ContentType.CSV
            elif url_path.endswith(".txt"):
                return ContentType.TXT
            elif url_path.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                return ContentType.IMAGE
            elif url_path.endswith((".mp4", ".avi", ".mov", ".wmv")):
                return ContentType.VIDEO
            elif url_path.endswith((".mp3", ".wav", ".ogg")):
                return ContentType.AUDIO

        # Check content patterns
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if pattern in input_data[:1000]:  # Check first 1000 chars
                    return content_type

        # Default to HTML for URLs without extension
        if input_data.startswith(("http://", "https://")):
            return ContentType.HTML

        return ContentType.UNKNOWN

    def _determine_website_type(self, url: str) -> Optional[WebsiteType]:
        """
        Determine the type of website from a URL.

        Args:
            url: The URL to analyze.

        Returns:
            The website type, or None if not determinable.
        """
        if not url.startswith(("http://", "https://")):
            return None

        # Extract domain and path
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        # Check domain keywords
        domain_str = domain + path

        # Check for known website types
        for website_type, patterns in self.website_patterns.items():
            for pattern in patterns:
                if pattern in domain_str:
                    return website_type

        return WebsiteType.UNKNOWN

    def _calculate_complexity(self, input_data: Union[str, Dict[str, Any]], content_type: ContentType) -> float:
        """
        Calculate the complexity of the input data.

        Args:
            input_data: The input data to analyze.
            content_type: The content type of the input data.

        Returns:
            A complexity score between 0.0 and 1.0.
        """
        # This is a placeholder implementation
        # In a real implementation, you would analyze the content structure,
        # number of elements, depth of nesting, etc.

        if content_type == ContentType.HTML:
            if isinstance(input_data, str):
                # Count tags as a simple complexity measure
                tag_count = len(re.findall(r"<[^>]+>", input_data[:10000]))
                return min(1.0, tag_count / 1000)

        elif content_type == ContentType.JSON:
            if isinstance(input_data, dict):
                # Count keys as a simple complexity measure
                return min(1.0, len(json.dumps(input_data)) / 10000)
            elif isinstance(input_data, str):
                try:
                    # Try to parse as JSON
                    data = json.loads(input_data)
                    return min(1.0, len(input_data) / 10000)
                except:
                    pass

        # Default complexity based on length
        if isinstance(input_data, str):
            return min(1.0, len(input_data) / 50000)

        return 0.5  # Default medium complexity

    def _requires_javascript(self, input_data: Union[str, Dict[str, Any]], content_type: ContentType) -> bool:
        """
        Determine if the input data requires JavaScript rendering.

        Args:
            input_data: The input data to analyze.
            content_type: The content type of the input data.

        Returns:
            True if JavaScript rendering is required, False otherwise.
        """
        if content_type != ContentType.HTML or not isinstance(input_data, str):
            return False

        # Check for common JavaScript frameworks and patterns
        js_indicators = [
            "vue", "react", "angular", "ember", "backbone", "knockout",
            "document.getElementById", "document.querySelector",
            "window.onload", "$(document).ready", "addEventListener",
            "fetch(", "axios.", ".ajax", "XMLHttpRequest",
            "data-bind", "ng-", "v-", "x-data", "x-bind"
        ]

        for indicator in js_indicators:
            if indicator in input_data.lower():
                return True

        return False

    def _requires_authentication(self, input_data: Union[str, Dict[str, Any]], input_type: InputType) -> bool:
        """
        Determine if the input data requires authentication.

        Args:
            input_data: The input data to analyze.
            input_type: The input type.

        Returns:
            True if authentication is required, False otherwise.
        """
        if input_type != InputType.URL or not isinstance(input_data, str):
            return False

        # Check for common authentication indicators in URLs
        auth_indicators = [
            "/login", "/signin", "/account", "/profile", "/dashboard",
            "/admin", "/member", "/secure", "/auth", "/private"
        ]

        for indicator in auth_indicators:
            if indicator in input_data.lower():
                return True

        return False

    def _has_anti_bot(self, input_data: Union[str, Dict[str, Any]], input_type: InputType) -> bool:
        """
        Determine if the input data has anti-bot measures.

        Args:
            input_data: The input data to analyze.
            input_type: The input type.

        Returns:
            True if anti-bot measures are detected, False otherwise.
        """
        if input_type != InputType.URL or not isinstance(input_data, str):
            return False

        # Check for common anti-bot domains
        anti_bot_domains = [
            "cloudflare.com", "distil.io", "akamai.com", "imperva.com",
            "datadome.co", "perimeterx.com", "shapesecurity.com"
        ]

        parsed_url = urlparse(input_data)
        domain = parsed_url.netloc.lower()

        for anti_bot_domain in anti_bot_domains:
            if anti_bot_domain in domain:
                return True

        return False

    def _has_pagination(self, input_data: Union[str, Dict[str, Any]], content_type: ContentType) -> bool:
        """
        Determine if the input data has pagination.

        Args:
            input_data: The input data to analyze.
            content_type: The content type of the input data.

        Returns:
            True if pagination is detected, False otherwise.
        """
        if content_type != ContentType.HTML or not isinstance(input_data, str):
            return False

        # Check for common pagination indicators
        pagination_indicators = [
            "page=", "p=", "pg=", "?page", "&page",
            "pagination", "pager", "next", "prev", "previous",
            "load more", "show more", "infinite scroll"
        ]

        for indicator in pagination_indicators:
            if indicator in input_data.lower():
                return True

        return False

    def _estimate_pages(self, input_data: Union[str, Dict[str, Any]], content_type: ContentType) -> Optional[int]:
        """
        Estimate the number of pages in paginated content.

        Args:
            input_data: The input data to analyze.
            content_type: The content type of the input data.

        Returns:
            Estimated number of pages, or None if not applicable.
        """
        if not self._has_pagination(input_data, content_type):
            return None

        if content_type == ContentType.HTML and isinstance(input_data, str):
            # Look for pagination numbers
            page_numbers = re.findall(r'page=(\d+)', input_data)
            if page_numbers:
                try:
                    return max(int(p) for p in page_numbers)
                except ValueError:
                    pass

            # Look for page indicators like "Page 1 of 10"
            page_of_pattern = re.search(r'[Pp]age\s+\d+\s+of\s+(\d+)', input_data)
            if page_of_pattern:
                try:
                    return int(page_of_pattern.group(1))
                except ValueError:
                    pass

        # Default estimate
        return 5  # Assume 5 pages as a default

    def _detect_language(self, input_data: Union[str, Dict[str, Any]], content_type: ContentType) -> Optional[str]:
        """
        Detect the language of the input data.

        Args:
            input_data: The input data to analyze.
            content_type: The content type of the input data.

        Returns:
            ISO language code, or None if not applicable.
        """
        # This is a placeholder implementation
        # In a real implementation, you would use a language detection library

        # Default to English
        return "en"

    def _extract_metadata(self, input_data: Union[str, Dict[str, Any]], content_type: ContentType, input_type: InputType) -> Dict[str, Any]:
        """
        Extract metadata from the input data.

        Args:
            input_data: The input data to analyze.
            content_type: The content type of the input data.
            input_type: The input type.

        Returns:
            A dictionary containing metadata.
        """
        metadata = {}

        if input_type == InputType.URL and isinstance(input_data, str):
            # Extract URL components
            parsed_url = urlparse(input_data)
            metadata["domain"] = parsed_url.netloc
            metadata["path"] = parsed_url.path
            metadata["query"] = parsed_url.query
            metadata["scheme"] = parsed_url.scheme

        if content_type == ContentType.HTML and isinstance(input_data, str):
            # Extract title
            title_match = re.search(r'<title>(.*?)</title>', input_data, re.IGNORECASE | re.DOTALL)
            if title_match:
                metadata["title"] = title_match.group(1).strip()

            # Extract meta description
            desc_match = re.search(r'<meta\s+name=["\'](description|Description)["\'][^>]*content=["\'](.*?)["\']', input_data)
            if desc_match:
                metadata["description"] = desc_match.group(2).strip()

        return metadata

    async def select_agents(self, analysis_result: ContentAnalysisResult) -> List[AgentAssignment]:
        """
        Select appropriate agents based on content analysis.

        Args:
            analysis_result: The content analysis result.

        Returns:
            A list of agent assignments.
        """
        self.logger.info(f"Selecting agents for content type: {analysis_result.content_type}")

        # Create selection criteria based on analysis
        criteria = self._create_selection_criteria(analysis_result)

        # Find agents matching the criteria
        matching_agents = self._find_matching_agents(criteria)

        # Create agent assignments
        assignments = []
        for agent_id, agent_type, capabilities, confidence in matching_agents:
            # Create a task ID
            task_id = str(uuid.uuid4())

            # Determine task type based on agent type and capabilities
            task_type = self._determine_task_type(agent_type, capabilities, analysis_result)

            # Create assignment
            assignment = AgentAssignment(
                agent_id=agent_id,
                agent_type=agent_type,
                task_id=task_id,
                task_type=task_type,
                priority=self._calculate_priority(agent_type, analysis_result),
                capabilities_used=list(capabilities),
                confidence=confidence
            )

            assignments.append(assignment)

        self.logger.info(f"Selected {len(assignments)} agents")
        return assignments

    def _create_selection_criteria(self, analysis_result: ContentAnalysisResult) -> AgentSelectionCriteria:
        """
        Create agent selection criteria based on content analysis.

        Args:
            analysis_result: The content analysis result.

        Returns:
            Agent selection criteria.
        """
        # Required capabilities based on content type
        required_capabilities = set()

        # Add content type specific capabilities
        if analysis_result.content_type == ContentType.HTML:
            required_capabilities.add(AgentCapability.HTML_PARSING)
        elif analysis_result.content_type == ContentType.JSON:
            required_capabilities.add(AgentCapability.JSON_PARSING)
        elif analysis_result.content_type == ContentType.XML:
            required_capabilities.add(AgentCapability.XML_PARSING)
        elif analysis_result.content_type == ContentType.PDF:
            required_capabilities.add(AgentCapability.PDF_PROCESSING)
        elif analysis_result.content_type in [ContentType.DOC, ContentType.DOCX, ContentType.XLS, ContentType.XLSX]:
            required_capabilities.add(AgentCapability.DOCUMENT_PROCESSING)
        elif analysis_result.content_type == ContentType.IMAGE:
            required_capabilities.add(AgentCapability.IMAGE_PROCESSING)

        # Add feature specific capabilities
        if analysis_result.requires_javascript:
            required_capabilities.add(AgentCapability.JAVASCRIPT_RENDERING)

        if analysis_result.requires_authentication:
            required_capabilities.add(AgentCapability.AUTHENTICATION)

        if analysis_result.has_anti_bot:
            required_capabilities.add(AgentCapability.ANTI_DETECTION)

        if analysis_result.has_pagination:
            required_capabilities.add(AgentCapability.PAGINATION_HANDLING)

        # Preferred capabilities
        preferred_capabilities = {
            AgentCapability.DATA_VALIDATION,
            AgentCapability.ERROR_RECOVERY,
            AgentCapability.PERFORMANCE_OPTIMIZATION,
            AgentCapability.QUALITY_ASSURANCE
        }

        # Create criteria
        criteria = AgentSelectionCriteria(
            required_capabilities=required_capabilities,
            preferred_capabilities=preferred_capabilities,
            min_success_rate=0.7,
            max_error_rate=0.3,
            content_type=analysis_result.content_type,
            website_type=analysis_result.website_type,
            task_complexity=analysis_result.complexity
        )

        return criteria

    def _find_matching_agents(self, criteria: AgentSelectionCriteria) -> List[Tuple[str, str, Set[AgentCapability], float]]:
        """
        Find agents matching the selection criteria.

        Args:
            criteria: The agent selection criteria.

        Returns:
            A list of tuples containing (agent_id, agent_type, capabilities, confidence).
        """
        matching_agents = []

        for agent_id, profile in self.agent_profiles.items():
            # Check if agent has all required capabilities
            if not criteria.required_capabilities.issubset(profile.capabilities):
                continue

            # Check performance metrics
            if profile.performance.success_rate < criteria.min_success_rate:
                continue

            if profile.performance.error_rate > criteria.max_error_rate:
                continue

            # Calculate match confidence
            confidence = self._calculate_match_confidence(profile, criteria)

            # Add to matching agents
            matching_agents.append((
                agent_id,
                profile.agent_type,
                profile.capabilities,
                confidence
            ))

        # Sort by confidence (descending)
        matching_agents.sort(key=lambda x: x[3], reverse=True)

        return matching_agents

    def _calculate_match_confidence(self, profile: AgentProfile, criteria: AgentSelectionCriteria) -> float:
        """
        Calculate the confidence of an agent match.

        Args:
            profile: The agent profile.
            criteria: The selection criteria.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        # Base confidence from success rate
        confidence = profile.performance.success_rate

        # Adjust based on preferred capabilities
        preferred_count = len(criteria.preferred_capabilities.intersection(profile.capabilities))
        preferred_bonus = preferred_count * 0.05
        confidence += preferred_bonus

        # Adjust based on specializations
        if criteria.content_type and criteria.content_type.value in profile.specializations:
            confidence += profile.specializations[criteria.content_type.value] * 0.1

        if criteria.website_type and criteria.website_type.value in profile.specializations:
            confidence += profile.specializations[criteria.website_type.value] * 0.1

        # Cap at 1.0
        return min(1.0, confidence)

    def _determine_task_type(self, agent_type: str, capabilities: Set[AgentCapability], analysis_result: ContentAnalysisResult) -> str:
        """
        Determine the task type for an agent.

        Args:
            agent_type: The agent type.
            capabilities: The agent capabilities.
            analysis_result: The content analysis result.

        Returns:
            A task type string.
        """
        # Map agent types to default task types
        agent_to_task = {
            "scraper": TaskType.FETCH_URL,
            "parser": TaskType.PARSE_CONTENT,
            "storage": TaskType.STORE_DATA,
            "javascript": TaskType.RENDER_JS,
            "authentication": TaskType.AUTHENTICATE,
            "anti_detection": TaskType.GENERATE_FINGERPRINT,
            "data_transformation": TaskType.TRANSFORM_DATA,
            "nlp_processing": TaskType.NLP_ENTITY_EXTRACTION,
            "image_processing": TaskType.IMAGE_EXTRACTION,
            "api_integration": TaskType.API_REQUEST,
            "error_recovery": TaskType.MONITOR_SYSTEM,
            "data_extractor": TaskType.EXTRACT_DATA,
            "url_intelligence": TaskType.ANALYZE_URL,
            "content_recognition": TaskType.RECOGNIZE_CONTENT,
            "document_intelligence": TaskType.PROCESS_DOCUMENT,
            "quality_assurance": TaskType.VALIDATE_DATA,
            "performance_optimization": TaskType.OPTIMIZE_PERFORMANCE
        }

        # Return the default task type for the agent type
        if agent_type in agent_to_task:
            return agent_to_task[agent_type]

        # Fallback to a generic task type
        return TaskType.FETCH_URL

    def _calculate_priority(self, agent_type: str, analysis_result: ContentAnalysisResult) -> int:
        """
        Calculate the priority for an agent assignment.

        Args:
            agent_type: The agent type.
            analysis_result: The content analysis result.

        Returns:
            A priority value (lower is higher priority).
        """
        # Base priorities for agent types
        base_priorities = {
            "anti_detection": 1,  # Highest priority
            "authentication": 2,
            "scraper": 3,
            "javascript": 4,
            "parser": 5,
            "data_extractor": 6,
            "data_transformation": 7,
            "storage": 8,
            "error_recovery": 9,
            "quality_assurance": 10  # Lowest priority
        }

        # Get base priority or default to middle priority
        priority = base_priorities.get(agent_type, 5)

        # Adjust based on content complexity
        if analysis_result.complexity > 0.7:
            # Higher complexity tasks get higher priority
            priority -= 1

        # Ensure priority is within bounds
        return max(1, min(10, priority))

    async def _handle_analyze_input(self, message: Message) -> None:
        """
        Handle an analyze input message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "input_data"):
            self.logger.warning("Received analyze_input message without input_data")
            return

        try:
            # Analyze the input
            result = await self.analyze_input(message.input_data)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result.dict()
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error analyzing input: {str(e)}")

            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)

    async def _handle_select_agents(self, message: Message) -> None:
        """
        Handle a select agents message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "analysis_result"):
            self.logger.warning("Received select_agents message without analysis_result")
            return

        try:
            # Convert dict to ContentAnalysisResult if needed
            analysis_result = message.analysis_result
            if isinstance(analysis_result, dict):
                analysis_result = ContentAnalysisResult(**analysis_result)

            # Select agents
            assignments = await self.select_agents(analysis_result)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"assignments": [a.dict() for a in assignments]}
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error selecting agents: {str(e)}")

            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)

    async def _handle_update_agent_profile(self, message: Message) -> None:
        """
        Handle an update agent profile message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "agent_id") or not hasattr(message, "profile_data"):
            self.logger.warning("Received update_agent_profile message without required fields")
            return

        try:
            agent_id = message.agent_id
            profile_data = message.profile_data

            # Check if agent exists
            if agent_id not in self.agent_profiles:
                self.logger.warning(f"Agent {agent_id} not found")
                return

            # Update profile
            profile = self.agent_profiles[agent_id]

            # Update performance metrics if provided
            if "performance" in profile_data:
                perf_data = profile_data["performance"]
                if "success_rate" in perf_data:
                    profile.performance.success_rate = perf_data["success_rate"]
                if "average_execution_time" in perf_data:
                    profile.performance.average_execution_time = perf_data["average_execution_time"]
                if "error_rate" in perf_data:
                    profile.performance.error_rate = perf_data["error_rate"]
                if "task_count" in perf_data:
                    profile.performance.task_count = perf_data["task_count"]

            # Update specializations if provided
            if "specializations" in profile_data:
                profile.specializations.update(profile_data["specializations"])

            # Update capabilities if provided
            if "capabilities" in profile_data:
                new_capabilities = set(profile_data["capabilities"])
                profile.capabilities.update(new_capabilities)

            # Update last updated timestamp
            profile.last_updated = time.time()

            # Send confirmation
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"success": True, "agent_id": agent_id}
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error updating agent profile: {str(e)}")

            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)

    async def _handle_learning_event(self, message: Message) -> None:
        """
        Handle a learning event message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "event_type") or not hasattr(message, "data"):
            self.logger.warning("Received learning_event message without required fields")
            return

        try:
            # Create learning event
            event = LearningEvent(
                event_type=message.event_type,
                agent_id=message.agent_id if hasattr(message, "agent_id") else None,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                data=message.data
            )

            # Store event
            self.learning_events.append(event)

            # Process event
            await self._process_learning_event(event)

            # Send confirmation
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"success": True, "event_id": event.event_id}
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error handling learning event: {str(e)}")

            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)

    async def _process_learning_event(self, event: LearningEvent) -> None:
        """
        Process a learning event.

        Args:
            event: The learning event to process.
        """
        # This is a placeholder implementation
        # In a real implementation, you would update agent profiles, decision-making
        # algorithms, etc. based on the learning event

        self.logger.info(f"Processing learning event: {event.event_type}")

        # Update agent profile if applicable
        if event.agent_id and event.agent_id in self.agent_profiles:
            profile = self.agent_profiles[event.agent_id]

            # Update based on event type
            if event.event_type == "task_success":
                # Update success metrics
                profile.performance.success_rate = (
                    (profile.performance.success_rate * profile.performance.task_count + 1) /
                    (profile.performance.task_count + 1)
                )
                profile.performance.task_count += 1
                profile.performance.last_success_time = event.timestamp

                # Update specializations if applicable
                if "content_type" in event.data:
                    content_type = event.data["content_type"]
                    if content_type not in profile.specializations:
                        profile.specializations[content_type] = 0.5
                    else:
                        profile.specializations[content_type] = min(
                            1.0, profile.specializations[content_type] + 0.05
                        )

            elif event.event_type == "task_error":
                # Update error metrics
                profile.performance.error_rate = (
                    (profile.performance.error_rate * profile.performance.task_count + 1) /
                    (profile.performance.task_count + 1)
                )
                profile.performance.task_count += 1
                profile.performance.last_error_time = event.timestamp

            # Update last execution time
            profile.performance.last_execution_time = event.timestamp

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        self.logger.info(f"Executing task: {task.type}")

        if task.type == TaskType.ANALYZE_INPUT:
            # Analyze input
            input_data = task.parameters.get("input_data")
            if not input_data:
                raise ValueError("Missing input_data parameter")

            result = await self.analyze_input(input_data)
            return result.dict()

        elif task.type == TaskType.SELECT_AGENTS:
            # Select agents
            analysis_result = task.parameters.get("analysis_result")
            if not analysis_result:
                raise ValueError("Missing analysis_result parameter")

            # Convert dict to ContentAnalysisResult if needed
            if isinstance(analysis_result, dict):
                analysis_result = ContentAnalysisResult(**analysis_result)

            assignments = await self.select_agents(analysis_result)
            return {"assignments": [a.dict() for a in assignments]}

        else:
            raise ValueError(f"Unsupported task type: {task.type}")
