"""
Enhanced Content Recognition Agent with Advanced AI & Intelligence Features.

This agent identifies and categorizes any input content using machine learning,
pattern recognition, and intelligent content classification.
"""
import asyncio
import logging
import time
import re
import json
import os
import io
import mimetypes
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import uuid
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
import numpy as np
from collections import defaultdict, Counter

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.intelligence import (
    ContentType, InputType, WebsiteType, AgentCapability,
    AgentProfile, ContentAnalysisResult
)

# Enhanced AI features
try:
    import spacy
    from textblob import TextBlob
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    logging.warning("Advanced NLP libraries not available. Install spacy and textblob for enhanced features.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Machine learning libraries not available. Install scikit-learn for ML features.")


class EnhancedContentRecognitionAgent(Agent):
    """
    Enhanced Content Recognition Agent with Advanced AI & Intelligence Features.

    Features:
    - ML-based content type detection
    - Dynamic extraction strategy selection
    - Smart pagination detection
    - Content quality assessment
    - Pattern learning and adaptation
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new Enhanced Content Recognition Agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        super().__init__(agent_id=agent_id, agent_type="enhanced_content_recognition", coordinator_id=coordinator_id)

        # HTTP client for making requests
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        )

        # Enhanced content patterns with ML support
        self.content_patterns = self._initialize_enhanced_content_patterns()

        # ML models for content classification
        self.ml_models = self._initialize_ml_models()

        # Pattern learning system
        self.pattern_learner = self._initialize_pattern_learner()

        # Content quality assessor
        self.quality_assessor = self._initialize_quality_assessor()

        # Cache for content analysis results with TTL
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour TTL

        # Site-specific extraction strategies
        self.extraction_strategies: Dict[str, Dict[str, Any]] = {}

        # Performance metrics
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_classifications": 0,
            "cache_hits": 0,
            "ml_predictions": 0,
            "pattern_matches": 0
        }

        # Register enhanced message handlers
        self.register_handler("recognize_content", self._handle_recognize_content)
        self.register_handler("detect_content_type", self._handle_detect_content_type)
        self.register_handler("analyze_structure", self._handle_analyze_structure)
        self.register_handler("classify_website", self._handle_classify_website)
        self.register_handler("suggest_extraction_strategy", self._handle_suggest_extraction_strategy)
        self.register_handler("assess_content_quality", self._handle_assess_content_quality)
        self.register_handler("learn_from_feedback", self._handle_learn_from_feedback)

        # Start enhanced periodic tasks
        self._start_enhanced_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the Content Recognition Agent."""
        asyncio.create_task(self._periodic_cache_cleanup())

    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up the analysis cache."""
        while self.running:
            self.logger.debug("Running periodic cache cleanup")

            # Remove cache entries older than 1 hour
            current_time = time.time()
            expired_keys = []

            for key, cache_entry in self.analysis_cache.items():
                if current_time - cache_entry.get("timestamp", 0) > 3600:
                    expired_keys.append(key)

            # Remove expired entries
            for key in expired_keys:
                del self.analysis_cache[key]

            self.logger.debug(f"Removed {len(expired_keys)} expired cache entries")

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

    async def recognize_content(self, input_data: Union[str, Dict[str, Any], bytes]) -> Dict[str, Any]:
        """
        Recognize and analyze input content.

        Args:
            input_data: The input data to analyze. Can be a URL, file path, raw content, or bytes.

        Returns:
            A dictionary containing the analysis results.
        """
        self.logger.info(f"Recognizing content: {type(input_data)}")

        # Generate a cache key
        if isinstance(input_data, str):
            cache_key = input_data[:100]  # Use first 100 chars as key
        elif isinstance(input_data, dict):
            cache_key = json.dumps(input_data, sort_keys=True)[:100]
        elif isinstance(input_data, bytes):
            cache_key = str(hash(input_data))
        else:
            cache_key = str(uuid.uuid4())

        # Check cache first
        if cache_key in self.analysis_cache:
            self.logger.info(f"Using cached analysis for content")
            return self.analysis_cache[cache_key]["result"]

        # Determine input type
        input_type = self._determine_input_type(input_data)

        # Fetch content if it's a URL
        content = input_data
        if input_type == InputType.URL and isinstance(input_data, str):
            try:
                response = await self.client.get(input_data)
                content = response.content  # Use bytes for better detection
                headers = dict(response.headers)
                content_type_header = headers.get("content-type", "")
            except Exception as e:
                self.logger.error(f"Error fetching URL: {str(e)}")
                content = b""
                headers = {}
                content_type_header = ""
        else:
            headers = {}
            content_type_header = ""

        # Determine content type
        content_type, confidence = self._determine_content_type(content, content_type_header)

        # Convert bytes to string for text-based content types
        if isinstance(content, bytes) and content_type in [ContentType.HTML, ContentType.JSON, ContentType.XML, ContentType.CSV, ContentType.TXT]:
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content = content.decode("latin-1")
                except Exception:
                    content = str(content)

        # Analyze content structure
        structure_info = self._analyze_structure(content, content_type)

        # Create result
        result = {
            "input_type": input_type.value,
            "content_type": content_type.value,
            "confidence": confidence,
            "structure": structure_info,
            "metadata": self._extract_metadata(content, content_type, headers),
            "size": len(content) if isinstance(content, (str, bytes)) else len(json.dumps(content)) if isinstance(content, dict) else 0,
            "headers": headers
        }

        # Cache the result
        self.analysis_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        self.logger.info(f"Content recognition complete: {content_type.value}")
        return result

    def _determine_input_type(self, input_data: Union[str, Dict[str, Any], bytes]) -> InputType:
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

        # If it's bytes, it could be file content
        if isinstance(input_data, bytes):
            return InputType.FILE

        return InputType.UNKNOWN

    def _determine_content_type(self, content: Union[str, Dict[str, Any], bytes], content_type_header: str = "") -> Tuple[ContentType, float]:
        """
        Determine the content type of the input data.

        Args:
            content: The content to analyze.
            content_type_header: The Content-Type header if available.

        Returns:
            A tuple containing (content_type, confidence).
        """
        # Check content type header first
        if content_type_header:
            if "text/html" in content_type_header:
                return ContentType.HTML, 0.9
            elif "application/json" in content_type_header:
                return ContentType.JSON, 0.9
            elif "text/xml" in content_type_header or "application/xml" in content_type_header:
                return ContentType.XML, 0.9
            elif "application/pdf" in content_type_header:
                return ContentType.PDF, 0.9
            elif "application/msword" in content_type_header:
                return ContentType.DOC, 0.9
            elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type_header:
                return ContentType.DOCX, 0.9
            elif "application/vnd.ms-excel" in content_type_header:
                return ContentType.XLS, 0.9
            elif "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type_header:
                return ContentType.XLSX, 0.9
            elif "text/csv" in content_type_header:
                return ContentType.CSV, 0.9
            elif "text/plain" in content_type_header:
                return ContentType.TXT, 0.9
            elif "image/" in content_type_header:
                return ContentType.IMAGE, 0.9
            elif "video/" in content_type_header:
                return ContentType.VIDEO, 0.9
            elif "audio/" in content_type_header:
                return ContentType.AUDIO, 0.9

        # If it's a dictionary, it's likely JSON
        if isinstance(content, dict):
            return ContentType.JSON, 1.0

        # If it's bytes, check for magic numbers
        if isinstance(content, bytes):
            # Check for PDF
            if content.startswith(b"%PDF-"):
                return ContentType.PDF, 1.0

            # Check for DOC
            if content.startswith(b"\xD0\xCF\x11\xE0"):
                return ContentType.DOC, 0.8

            # Check for DOCX/XLSX (both are ZIP-based)
            if content.startswith(b"PK"):
                # Need more sophisticated checks to distinguish between DOCX and XLSX
                return ContentType.DOCX, 0.6

            # Try to decode as text
            try:
                text_content = content.decode("utf-8", errors="ignore")
                # Now check text patterns
                return self._check_text_patterns(text_content)
            except:
                return ContentType.UNKNOWN, 0.1

        # If it's a string, check for patterns
        if isinstance(content, str):
            return self._check_text_patterns(content)

        return ContentType.UNKNOWN, 0.1

    def _check_text_patterns(self, content: str) -> Tuple[ContentType, float]:
        """
        Check text content for patterns to determine content type.

        Args:
            content: The text content to analyze.

        Returns:
            A tuple containing (content_type, confidence).
        """
        # Check for HTML
        if content.strip().startswith(("<!DOCTYPE html>", "<html")) or "<body" in content:
            return ContentType.HTML, 0.9

        # Check for JSON
        if content.strip().startswith(("{", "[")):
            try:
                json.loads(content)
                return ContentType.JSON, 1.0
            except:
                pass

        # Check for XML
        if content.strip().startswith(("<?xml", "<")) and ">" in content:
            return ContentType.XML, 0.8

        # Check for CSV
        if "," in content:
            lines = content.split("\n")
            if len(lines) > 1:
                # Check if all lines have the same number of commas
                comma_counts = [line.count(",") for line in lines[:10] if line.strip()]
                if comma_counts and all(count == comma_counts[0] for count in comma_counts) and comma_counts[0] > 0:
                    return ContentType.CSV, 0.8

        # Default to TXT
        return ContentType.TXT, 0.5

    def _analyze_structure(self, content: Union[str, Dict[str, Any], bytes], content_type: ContentType) -> Dict[str, Any]:
        """
        Analyze the structure of the content.

        Args:
            content: The content to analyze.
            content_type: The content type.

        Returns:
            A dictionary containing structure information.
        """
        structure = {}

        if content_type == ContentType.HTML and isinstance(content, str):
            # Parse HTML
            soup = BeautifulSoup(content, "html.parser")

            # Count elements
            elements = soup.find_all()
            structure["element_count"] = len(elements)

            # Count element types
            element_types = {}
            for element in elements:
                element_name = element.name
                if element_name in element_types:
                    element_types[element_name] += 1
                else:
                    element_types[element_name] = 1
            structure["element_types"] = element_types

            # Extract main sections
            structure["has_header"] = bool(soup.find("header") or soup.find(id="header") or soup.find(class_="header"))
            structure["has_footer"] = bool(soup.find("footer") or soup.find(id="footer") or soup.find(class_="footer"))
            structure["has_nav"] = bool(soup.find("nav") or soup.find(id="nav") or soup.find(class_="nav"))
            structure["has_sidebar"] = bool(soup.find(id="sidebar") or soup.find(class_="sidebar"))
            structure["has_main"] = bool(soup.find("main") or soup.find(id="main") or soup.find(class_="main"))

            # Count forms
            forms = soup.find_all("form")
            structure["form_count"] = len(forms)

            # Count tables
            tables = soup.find_all("table")
            structure["table_count"] = len(tables)

            # Count links
            links = soup.find_all("a")
            structure["link_count"] = len(links)

            # Count images
            images = soup.find_all("img")
            structure["image_count"] = len(images)

            # Count scripts
            scripts = soup.find_all("script")
            structure["script_count"] = len(scripts)

            # Count styles
            styles = soup.find_all("style")
            structure["style_count"] = len(styles)

            # Check for iframes
            iframes = soup.find_all("iframe")
            structure["iframe_count"] = len(iframes)

        elif content_type == ContentType.JSON:
            # Parse JSON
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                except:
                    data = {}
            else:
                data = content

            # Analyze structure
            structure["type"] = "object" if isinstance(data, dict) else "array" if isinstance(data, list) else "value"

            if isinstance(data, dict):
                structure["key_count"] = len(data)
                structure["keys"] = list(data.keys())

                # Analyze value types
                value_types = {}
                for value in data.values():
                    value_type = type(value).__name__
                    if value_type in value_types:
                        value_types[value_type] += 1
                    else:
                        value_types[value_type] = 1
                structure["value_types"] = value_types

                # Check for nested objects/arrays
                structure["has_nested_objects"] = any(isinstance(v, dict) for v in data.values())
                structure["has_nested_arrays"] = any(isinstance(v, list) for v in data.values())

            elif isinstance(data, list):
                structure["item_count"] = len(data)

                # Analyze item types
                item_types = {}
                for item in data:
                    item_type = type(item).__name__
                    if item_type in item_types:
                        item_types[item_type] += 1
                    else:
                        item_types[item_type] = 1
                structure["item_types"] = item_types

                # Check for nested objects/arrays
                structure["has_nested_objects"] = any(isinstance(item, dict) for item in data)
                structure["has_nested_arrays"] = any(isinstance(item, list) for item in data)

        elif content_type == ContentType.XML and isinstance(content, str):
            # Parse XML
            soup = BeautifulSoup(content, "xml")

            # Count elements
            elements = soup.find_all()
            structure["element_count"] = len(elements)

            # Count element types
            element_types = {}
            for element in elements:
                element_name = element.name
                if element_name in element_types:
                    element_types[element_name] += 1
                else:
                    element_types[element_name] = 1
            structure["element_types"] = element_types

            # Get root element
            if soup.contents and soup.contents[0].name:
                structure["root_element"] = soup.contents[0].name

        elif content_type == ContentType.CSV and isinstance(content, str):
            # Parse CSV
            lines = content.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]

            structure["line_count"] = len(non_empty_lines)

            if non_empty_lines:
                # Detect delimiter
                first_line = non_empty_lines[0]
                comma_count = first_line.count(",")
                semicolon_count = first_line.count(";")
                tab_count = first_line.count("\t")

                if semicolon_count > comma_count and semicolon_count > tab_count:
                    delimiter = ";"
                elif tab_count > comma_count and tab_count > semicolon_count:
                    delimiter = "\t"
                else:
                    delimiter = ","

                structure["delimiter"] = delimiter

                # Count columns
                structure["column_count"] = first_line.count(delimiter) + 1

                # Check if has header
                if len(non_empty_lines) > 1:
                    # Heuristic: if first row has different format than others, it's likely a header
                    first_row_format = all(not cell.replace(".", "", 1).isdigit() for cell in first_line.split(delimiter))
                    second_row_format = all(not cell.replace(".", "", 1).isdigit() for cell in non_empty_lines[1].split(delimiter))
                    structure["has_header"] = first_row_format != second_row_format

        elif content_type in [ContentType.PDF, ContentType.DOC, ContentType.DOCX, ContentType.XLS, ContentType.XLSX] and isinstance(content, bytes):
            # For binary formats, just store the size
            structure["size_bytes"] = len(content)

        return structure

    def _extract_metadata(self, content: Union[str, Dict[str, Any], bytes], content_type: ContentType, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract metadata from the content.

        Args:
            content: The content to analyze.
            content_type: The content type.
            headers: HTTP headers if available.

        Returns:
            A dictionary containing metadata.
        """
        metadata = {}

        # Add content-type from headers
        if "content-type" in headers:
            metadata["content_type_header"] = headers["content-type"]

        # Add last-modified from headers
        if "last-modified" in headers:
            metadata["last_modified"] = headers["last-modified"]

        if content_type == ContentType.HTML and isinstance(content, str):
            # Parse HTML
            soup = BeautifulSoup(content, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text()

            # Extract meta tags
            meta_tags = soup.find_all("meta")
            meta_data = {}

            for tag in meta_tags:
                name = tag.get("name", tag.get("property", ""))
                content = tag.get("content", "")

                if name and content:
                    meta_data[name] = content

            if meta_data:
                metadata["meta_tags"] = meta_data

            # Extract charset
            charset_meta = soup.find("meta", charset=True)
            if charset_meta:
                metadata["charset"] = charset_meta["charset"]

            # Extract language
            html_tag = soup.find("html")
            if html_tag and html_tag.get("lang"):
                metadata["language"] = html_tag["lang"]

        elif content_type == ContentType.JSON and isinstance(content, (str, dict)):
            # For JSON, extract top-level metadata fields
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                except:
                    data = {}
            else:
                data = content

            if isinstance(data, dict):
                # Common metadata fields
                metadata_fields = ["title", "description", "author", "version", "date", "created", "modified", "language"]

                for field in metadata_fields:
                    if field in data:
                        metadata[field] = data[field]

        return metadata

    async def _handle_recognize_content(self, message: Message) -> None:
        """
        Handle a recognize content message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "content"):
            self.logger.warning("Received recognize_content message without content")
            return

        try:
            # Recognize the content
            result = await self.recognize_content(message.content)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error recognizing content: {str(e)}")

            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)

    async def _handle_detect_content_type(self, message: Message) -> None:
        """
        Handle a detect content type message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "content"):
            self.logger.warning("Received detect_content_type message without content")
            return

        try:
            # Detect content type
            content_type_header = message.content_type_header if hasattr(message, "content_type_header") else ""
            content_type, confidence = self._determine_content_type(message.content, content_type_header)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={
                    "content_type": content_type.value,
                    "confidence": confidence
                }
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error detecting content type: {str(e)}")

            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)

    async def _handle_analyze_structure(self, message: Message) -> None:
        """
        Handle an analyze structure message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "content") or not hasattr(message, "content_type"):
            self.logger.warning("Received analyze_structure message without required fields")
            return

        try:
            # Convert string content type to enum
            if isinstance(message.content_type, str):
                try:
                    content_type = ContentType(message.content_type)
                except ValueError:
                    content_type = ContentType.UNKNOWN
            else:
                content_type = message.content_type

            # Analyze structure
            structure = self._analyze_structure(message.content, content_type)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"structure": structure}
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error analyzing structure: {str(e)}")

            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        self.logger.info(f"Executing task: {task.type}")

        if task.type == TaskType.RECOGNIZE_CONTENT:
            # Recognize content
            content = task.parameters.get("content")
            if not content:
                raise ValueError("Missing content parameter")

            return await self.recognize_content(content)

        elif task.type == TaskType.DETECT_CONTENT_TYPE:
            # Detect content type
            content = task.parameters.get("content")
            if not content:
                raise ValueError("Missing content parameter")

            content_type_header = task.parameters.get("content_type_header", "")
            content_type, confidence = self._determine_content_type(content, content_type_header)

            return {
                "content_type": content_type.value,
                "confidence": confidence
            }

        elif task.type == TaskType.ANALYZE_STRUCTURE:
            # Analyze structure
            content = task.parameters.get("content")
            if not content:
                raise ValueError("Missing content parameter")

            content_type_str = task.parameters.get("content_type")
            if not content_type_str:
                raise ValueError("Missing content_type parameter")

            # Convert string content type to enum
            try:
                content_type = ContentType(content_type_str)
            except ValueError:
                content_type = ContentType.UNKNOWN

            structure = self._analyze_structure(content, content_type)
            return {"structure": structure}

        else:
            raise ValueError(f"Unsupported task type: {task.type}")

    # ===== ENHANCED AI FEATURES =====

    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for content classification."""
        models = {}

        if ML_AVAILABLE:
            # TF-IDF vectorizer for text analysis
            models['tfidf'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            # K-means clustering for content grouping
            models['kmeans'] = KMeans(n_clusters=5, random_state=42)

            self.logger.info("ML models initialized successfully")
        else:
            self.logger.warning("ML models not available - install scikit-learn")

        return models

    def _initialize_pattern_learner(self) -> Dict[str, Any]:
        """Initialize pattern learning system."""
        return {
            'site_patterns': defaultdict(list),
            'extraction_success_rates': defaultdict(float),
            'content_type_patterns': defaultdict(list),
            'structure_patterns': defaultdict(list),
            'learning_enabled': True
        }

    def _initialize_quality_assessor(self) -> Dict[str, Any]:
        """Initialize content quality assessment system."""
        return {
            'quality_metrics': {
                'completeness': 0.0,
                'accuracy': 0.0,
                'consistency': 0.0,
                'freshness': 0.0
            },
            'quality_thresholds': {
                'completeness': 0.8,
                'accuracy': 0.9,
                'consistency': 0.85,
                'freshness': 0.7
            },
            'assessment_history': []
        }

    def _start_enhanced_periodic_tasks(self) -> None:
        """Start enhanced periodic tasks."""
        asyncio.create_task(self._periodic_cache_cleanup())
        asyncio.create_task(self._periodic_pattern_analysis())
        asyncio.create_task(self._periodic_model_training())
        asyncio.create_task(self._periodic_performance_analysis())

    async def _periodic_pattern_analysis(self) -> None:
        """Periodically analyze patterns and update learning models."""
        while self.running:
            try:
                self.logger.debug("Running periodic pattern analysis")

                # Analyze site patterns
                await self._analyze_site_patterns()

                # Update extraction strategies
                await self._update_extraction_strategies()

                # Clean old patterns
                await self._cleanup_old_patterns()

                self.logger.debug("Pattern analysis completed")

            except Exception as e:
                self.logger.error(f"Error in pattern analysis: {e}")

            # Sleep for 30 minutes
            await asyncio.sleep(1800)

    async def _periodic_model_training(self) -> None:
        """Periodically retrain ML models with new data."""
        while self.running:
            try:
                if ML_AVAILABLE and len(self.analysis_cache) > 50:
                    self.logger.debug("Running periodic model training")

                    # Extract training data from cache
                    training_data = self._extract_training_data()

                    if training_data:
                        # Retrain models
                        await self._retrain_models(training_data)

                        self.logger.debug("Model training completed")

            except Exception as e:
                self.logger.error(f"Error in model training: {e}")

            # Sleep for 2 hours
            await asyncio.sleep(7200)

    async def _periodic_performance_analysis(self) -> None:
        """Periodically analyze performance and optimize."""
        while self.running:
            try:
                self.logger.debug("Running periodic performance analysis")

                # Calculate performance metrics
                total_analyses = self.performance_metrics['total_analyses']
                if total_analyses > 0:
                    cache_hit_rate = self.performance_metrics['cache_hits'] / total_analyses
                    success_rate = self.performance_metrics['successful_classifications'] / total_analyses

                    self.logger.info(f"Performance metrics - Cache hit rate: {cache_hit_rate:.2%}, Success rate: {success_rate:.2%}")

                # Reset metrics for next period
                self.performance_metrics = {
                    "total_analyses": 0,
                    "successful_classifications": 0,
                    "cache_hits": 0,
                    "ml_predictions": 0,
                    "pattern_matches": 0
                }

            except Exception as e:
                self.logger.error(f"Error in performance analysis: {e}")

            # Sleep for 1 hour
            await asyncio.sleep(3600)

    async def classify_website_type(self, url: str, content: str) -> Dict[str, Any]:
        """
        Classify website type using AI and pattern recognition.

        Args:
            url: Website URL
            content: HTML content

        Returns:
            Classification results with confidence scores
        """
        self.performance_metrics['total_analyses'] += 1

        try:
            # Parse content
            soup = BeautifulSoup(content, 'html.parser')

            # Extract features for classification
            features = self._extract_website_features(url, soup)

            # Use ML classification if available
            if ML_AVAILABLE and 'tfidf' in self.ml_models:
                ml_classification = await self._ml_classify_website(features, content)
            else:
                ml_classification = None

            # Use rule-based classification
            rule_classification = self._rule_based_classify_website(features)

            # Combine results
            final_classification = self._combine_classifications(ml_classification, rule_classification)

            # Learn from this classification
            await self._learn_from_classification(url, features, final_classification)

            self.performance_metrics['successful_classifications'] += 1

            return final_classification

        except Exception as e:
            self.logger.error(f"Error classifying website: {e}")
            return {
                'website_type': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }

    def _extract_website_features(self, url: str, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract features from website for classification."""
        features = {}

        # URL features
        parsed_url = urlparse(url)
        features['domain'] = parsed_url.netloc
        features['path_depth'] = len([p for p in parsed_url.path.split('/') if p])
        features['has_subdomain'] = len(parsed_url.netloc.split('.')) > 2

        # Content features
        text_content = soup.get_text()
        features['text_length'] = len(text_content)
        features['word_count'] = len(text_content.split())

        # Structure features
        features['form_count'] = len(soup.find_all('form'))
        features['table_count'] = len(soup.find_all('table'))
        features['list_count'] = len(soup.find_all(['ul', 'ol']))
        features['image_count'] = len(soup.find_all('img'))
        features['link_count'] = len(soup.find_all('a'))
        features['script_count'] = len(soup.find_all('script'))

        # E-commerce indicators
        features['has_cart'] = bool(soup.find(text=re.compile(r'cart|basket|checkout', re.I)))
        features['has_price'] = bool(soup.find(text=re.compile(r'\$\d+|\d+\.\d+\s*(USD|EUR|GBP)', re.I)))
        features['has_product'] = bool(soup.find(text=re.compile(r'product|item|buy now|add to cart', re.I)))

        # News indicators
        features['has_article'] = bool(soup.find(['article', 'main']) or soup.find(class_=re.compile(r'article|post|news', re.I)))
        features['has_date'] = bool(soup.find(text=re.compile(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', re.I)))
        features['has_author'] = bool(soup.find(text=re.compile(r'by\s+\w+|author:', re.I)))

        # Social media indicators
        features['has_social_links'] = bool(soup.find('a', href=re.compile(r'facebook|twitter|instagram|linkedin', re.I)))
        features['has_share_buttons'] = bool(soup.find(text=re.compile(r'share|tweet|like', re.I)))

        # Forum indicators
        features['has_thread'] = bool(soup.find(text=re.compile(r'thread|topic|reply|post #', re.I)))
        features['has_user_profile'] = bool(soup.find(text=re.compile(r'profile|member|user', re.I)))

        return features

    async def _ml_classify_website(self, features: Dict[str, Any], content: str) -> Optional[Dict[str, Any]]:
        """Use ML models to classify website type."""
        try:
            if not ML_AVAILABLE:
                return None

            # Convert features to vector
            feature_vector = self._features_to_vector(features)

            # Use TF-IDF on content
            if hasattr(self.ml_models['tfidf'], 'vocabulary_'):
                content_vector = self.ml_models['tfidf'].transform([content])

                # Simple classification based on content similarity
                # In a real implementation, you'd train a proper classifier
                website_types = ['ecommerce', 'news', 'blog', 'forum', 'corporate', 'social']

                # Placeholder ML classification
                # You would replace this with actual trained model prediction
                confidence = 0.7
                predicted_type = 'unknown'

                return {
                    'type': predicted_type,
                    'confidence': confidence,
                    'method': 'ml'
                }

        except Exception as e:
            self.logger.error(f"ML classification error: {e}")

        return None

    def _rule_based_classify_website(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Use rule-based classification for website type."""
        scores = defaultdict(float)

        # E-commerce scoring
        if features.get('has_cart', False):
            scores['ecommerce'] += 0.4
        if features.get('has_price', False):
            scores['ecommerce'] += 0.3
        if features.get('has_product', False):
            scores['ecommerce'] += 0.3

        # News scoring
        if features.get('has_article', False):
            scores['news'] += 0.4
        if features.get('has_date', False):
            scores['news'] += 0.3
        if features.get('has_author', False):
            scores['news'] += 0.3

        # Forum scoring
        if features.get('has_thread', False):
            scores['forum'] += 0.5
        if features.get('has_user_profile', False):
            scores['forum'] += 0.3

        # Social media scoring
        if features.get('has_social_links', False):
            scores['social'] += 0.3
        if features.get('has_share_buttons', False):
            scores['social'] += 0.2

        # Blog scoring (similar to news but different patterns)
        if features.get('has_article', False) and features.get('has_author', False):
            scores['blog'] += 0.4

        # Corporate scoring (default for business sites)
        if not any(scores.values()):
            scores['corporate'] = 0.5

        # Find best match
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            return {
                'type': best_type[0],
                'confidence': min(best_type[1], 1.0),
                'method': 'rule_based',
                'all_scores': dict(scores)
            }

        return {
            'type': 'unknown',
            'confidence': 0.0,
            'method': 'rule_based'
        }

    async def suggest_extraction_strategy(self, url: str, content: str, target_data: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Suggest optimal extraction strategy based on content analysis.

        Args:
            url: Website URL
            content: HTML content
            target_data: List of data types to extract (e.g., ['title', 'price', 'description'])

        Returns:
            Suggested extraction strategy with selectors and confidence
        """
        try:
            # Classify website type first
            website_classification = await self.classify_website_type(url, content)
            website_type = website_classification.get('type', 'unknown')

            # Parse content
            soup = BeautifulSoup(content, 'html.parser')

            # Get site-specific patterns if available
            domain = urlparse(url).netloc
            site_patterns = self.pattern_learner['site_patterns'].get(domain, [])

            # Generate strategy based on website type and patterns
            strategy = self._generate_extraction_strategy(website_type, soup, target_data, site_patterns)

            # Add pagination detection
            pagination_info = await self._detect_smart_pagination(url, content)
            strategy['pagination'] = pagination_info

            # Add quality assessment
            quality_score = await self._assess_extraction_quality(strategy, soup)
            strategy['quality_score'] = quality_score

            return strategy

        except Exception as e:
            self.logger.error(f"Error suggesting extraction strategy: {e}")
            return {
                'error': str(e),
                'strategy': 'fallback',
                'selectors': {},
                'confidence': 0.0
            }

    def _generate_extraction_strategy(self, website_type: str, soup: BeautifulSoup,
                                    target_data: Optional[List[str]], site_patterns: List[Dict]) -> Dict[str, Any]:
        """Generate extraction strategy based on website type and patterns."""
        strategy = {
            'website_type': website_type,
            'selectors': {},
            'confidence': 0.0,
            'method': 'adaptive'
        }

        # Use site-specific patterns if available
        if site_patterns:
            strategy['selectors'] = site_patterns[-1].get('selectors', {})
            strategy['confidence'] = 0.8
            strategy['method'] = 'learned_patterns'
            return strategy

        # Generate selectors based on website type
        if website_type == 'ecommerce':
            strategy['selectors'] = self._get_ecommerce_selectors(soup)
        elif website_type == 'news':
            strategy['selectors'] = self._get_news_selectors(soup)
        elif website_type == 'blog':
            strategy['selectors'] = self._get_blog_selectors(soup)
        elif website_type == 'forum':
            strategy['selectors'] = self._get_forum_selectors(soup)
        else:
            strategy['selectors'] = self._get_generic_selectors(soup)

        # Filter selectors based on target data if specified
        if target_data:
            filtered_selectors = {}
            for target in target_data:
                if target in strategy['selectors']:
                    filtered_selectors[target] = strategy['selectors'][target]
            strategy['selectors'] = filtered_selectors

        # Calculate confidence based on selector quality
        strategy['confidence'] = self._calculate_selector_confidence(strategy['selectors'], soup)

        return strategy

    def _get_ecommerce_selectors(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Get selectors optimized for e-commerce sites."""
        selectors = {}

        # Common e-commerce selectors
        title_candidates = [
            'h1.product-title', 'h1.product-name', '.product-title', '.product-name',
            'h1[data-testid*="product"]', '.pdp-product-name', '.product-info h1'
        ]

        price_candidates = [
            '.price', '.product-price', '.current-price', '.sale-price',
            '[data-testid*="price"]', '.price-current', '.price-now'
        ]

        description_candidates = [
            '.product-description', '.product-details', '.description',
            '.product-info', '.pdp-description'
        ]

        image_candidates = [
            '.product-image img', '.product-photo img', '.main-image img',
            '.hero-image img', '[data-testid*="image"] img'
        ]

        # Test selectors and pick the best ones
        selectors['title'] = self._find_best_selector(soup, title_candidates)
        selectors['price'] = self._find_best_selector(soup, price_candidates)
        selectors['description'] = self._find_best_selector(soup, description_candidates)
        selectors['image'] = self._find_best_selector(soup, image_candidates)

        return {k: v for k, v in selectors.items() if v}

    def _get_news_selectors(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Get selectors optimized for news sites."""
        selectors = {}

        title_candidates = [
            'h1.headline', 'h1.article-title', '.article-headline',
            'h1[data-testid*="headline"]', '.entry-title', '.post-title'
        ]

        content_candidates = [
            '.article-content', '.article-body', '.entry-content',
            '.post-content', '.story-body', '.article-text'
        ]

        author_candidates = [
            '.author', '.byline', '.article-author', '.post-author',
            '[data-testid*="author"]', '.author-name'
        ]

        date_candidates = [
            '.publish-date', '.article-date', '.post-date',
            'time[datetime]', '.timestamp', '.date'
        ]

        selectors['title'] = self._find_best_selector(soup, title_candidates)
        selectors['content'] = self._find_best_selector(soup, content_candidates)
        selectors['author'] = self._find_best_selector(soup, author_candidates)
        selectors['date'] = self._find_best_selector(soup, date_candidates)

        return {k: v for k, v in selectors.items() if v}

    def _get_blog_selectors(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Get selectors optimized for blog sites."""
        # Similar to news but with different patterns
        return self._get_news_selectors(soup)

    def _get_forum_selectors(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Get selectors optimized for forum sites."""
        selectors = {}

        thread_candidates = [
            '.thread-title', '.topic-title', '.post-title',
            'h1.thread', '.discussion-title'
        ]

        post_candidates = [
            '.post-content', '.message-content', '.post-body',
            '.forum-post', '.thread-post'
        ]

        user_candidates = [
            '.username', '.author', '.poster', '.user-name',
            '.member-name', '.post-author'
        ]

        selectors['thread_title'] = self._find_best_selector(soup, thread_candidates)
        selectors['post_content'] = self._find_best_selector(soup, post_candidates)
        selectors['username'] = self._find_best_selector(soup, user_candidates)

        return {k: v for k, v in selectors.items() if v}

    def _get_generic_selectors(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Get generic selectors for unknown site types."""
        selectors = {}

        # Generic patterns
        title_candidates = ['h1', 'title', '.title', '.heading']
        content_candidates = ['.content', '.main', 'main', 'article', '.article']

        selectors['title'] = self._find_best_selector(soup, title_candidates)
        selectors['content'] = self._find_best_selector(soup, content_candidates)

        return {k: v for k, v in selectors.items() if v}

    def _find_best_selector(self, soup: BeautifulSoup, candidates: List[str]) -> Optional[str]:
        """Find the best selector from candidates based on element presence and content quality."""
        for selector in candidates:
            try:
                elements = soup.select(selector)
                if elements:
                    # Check if elements have meaningful content
                    for element in elements[:3]:  # Check first 3 elements
                        text = element.get_text(strip=True)
                        if text and len(text) > 3:  # Has meaningful content
                            return selector
            except Exception:
                continue
        return None

    def _calculate_selector_confidence(self, selectors: Dict[str, str], soup: BeautifulSoup) -> float:
        """Calculate confidence score for selectors."""
        if not selectors:
            return 0.0

        total_score = 0.0
        for selector in selectors.values():
            try:
                elements = soup.select(selector)
                if elements:
                    # Score based on element count and content quality
                    element_score = min(len(elements) / 10.0, 1.0)  # Normalize to 0-1

                    # Check content quality
                    content_quality = 0.0
                    for element in elements[:3]:
                        text = element.get_text(strip=True)
                        if text:
                            content_quality += min(len(text) / 100.0, 1.0)

                    content_quality = content_quality / min(len(elements), 3)
                    total_score += (element_score + content_quality) / 2
            except Exception:
                continue

        return min(total_score / len(selectors), 1.0)

    async def _detect_smart_pagination(self, url: str, content: str) -> Dict[str, Any]:
        """Detect pagination using enhanced AI techniques."""
        try:
            # Use existing pagination engine
            from utils.pagination_engine import PaginationEngine

            engine = PaginationEngine()
            pagination_info = await engine.detect_pagination(url, content)

            # Enhance with AI analysis
            if pagination_info.get('has_pagination', False):
                # Analyze pagination patterns for learning
                domain = urlparse(url).netloc
                pattern = {
                    'domain': domain,
                    'method': pagination_info['best_method']['type'],
                    'confidence': pagination_info['best_method']['confidence'],
                    'timestamp': time.time()
                }

                # Store pattern for learning
                self.pattern_learner['site_patterns'][domain].append(pattern)

                # Limit stored patterns per domain
                if len(self.pattern_learner['site_patterns'][domain]) > 10:
                    self.pattern_learner['site_patterns'][domain] = \
                        self.pattern_learner['site_patterns'][domain][-10:]

            return pagination_info

        except Exception as e:
            self.logger.error(f"Error in smart pagination detection: {e}")
            return {'has_pagination': False, 'error': str(e)}

    async def _assess_extraction_quality(self, strategy: Dict[str, Any], soup: BeautifulSoup) -> float:
        """Assess the quality of an extraction strategy."""
        try:
            selectors = strategy.get('selectors', {})
            if not selectors:
                return 0.0

            quality_scores = []

            for field, selector in selectors.items():
                try:
                    elements = soup.select(selector)
                    if elements:
                        # Assess element quality
                        element_quality = self._assess_element_quality(elements[0])
                        quality_scores.append(element_quality)
                    else:
                        quality_scores.append(0.0)
                except Exception:
                    quality_scores.append(0.0)

            return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        except Exception as e:
            self.logger.error(f"Error assessing extraction quality: {e}")
            return 0.0

    def _assess_element_quality(self, element) -> float:
        """Assess the quality of a single element."""
        score = 0.0

        # Check if element has text content
        text = element.get_text(strip=True)
        if text:
            score += 0.4

            # Bonus for meaningful length
            if len(text) > 10:
                score += 0.2

            # Bonus for structured content
            if any(char in text for char in ['.', ',', ':', ';']):
                score += 0.1

        # Check for attributes that indicate importance
        if element.get('id'):
            score += 0.1
        if element.get('class'):
            score += 0.1

        # Check for semantic HTML
        if element.name in ['h1', 'h2', 'h3', 'title', 'strong', 'em']:
            score += 0.1

        return min(score, 1.0)

    # ===== ENHANCED MESSAGE HANDLERS =====

    async def _handle_classify_website(self, message) -> None:
        """Handle website classification requests."""
        try:
            url = message.url
            content = message.content

            result = await self.classify_website_type(url, content)

            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, 'task_id', None),
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error handling website classification: {e}")
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, 'task_id', None),
                error=str(e)
            )
            self.outbox.put(error_message)

    async def _handle_suggest_extraction_strategy(self, message) -> None:
        """Handle extraction strategy suggestion requests."""
        try:
            url = message.url
            content = message.content
            target_data = getattr(message, 'target_data', None)

            result = await self.suggest_extraction_strategy(url, content, target_data)

            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, 'task_id', None),
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error handling extraction strategy suggestion: {e}")
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, 'task_id', None),
                error=str(e)
            )
            self.outbox.put(error_message)

    async def _handle_assess_content_quality(self, message) -> None:
        """Handle content quality assessment requests."""
        try:
            content = message.content
            content_type = getattr(message, 'content_type', 'html')

            quality_assessment = await self._assess_content_quality(content, content_type)

            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, 'task_id', None),
                result=quality_assessment
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error handling content quality assessment: {e}")
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, 'task_id', None),
                error=str(e)
            )
            self.outbox.put(error_message)

    async def _handle_learn_from_feedback(self, message) -> None:
        """Handle learning from feedback requests."""
        try:
            feedback_data = message.feedback_data

            await self._learn_from_feedback(feedback_data)

            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, 'task_id', None),
                result={'status': 'learned', 'message': 'Feedback processed successfully'}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error handling learning from feedback: {e}")
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, 'task_id', None),
                error=str(e)
            )
            self.outbox.put(error_message)

    # ===== LEARNING AND ADAPTATION METHODS =====

    async def _learn_from_classification(self, url: str, features: Dict[str, Any], classification: Dict[str, Any]) -> None:
        """Learn from classification results to improve future predictions."""
        try:
            domain = urlparse(url).netloc

            # Store classification pattern
            pattern = {
                'domain': domain,
                'features': features,
                'classification': classification,
                'timestamp': time.time()
            }

            self.pattern_learner['content_type_patterns'][domain].append(pattern)

            # Limit stored patterns per domain
            if len(self.pattern_learner['content_type_patterns'][domain]) > 20:
                self.pattern_learner['content_type_patterns'][domain] = \
                    self.pattern_learner['content_type_patterns'][domain][-20:]

        except Exception as e:
            self.logger.error(f"Error learning from classification: {e}")

    async def _learn_from_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Learn from user feedback to improve extraction strategies."""
        try:
            url = feedback_data.get('url')
            strategy = feedback_data.get('strategy')
            success_rate = feedback_data.get('success_rate', 0.0)
            extracted_data = feedback_data.get('extracted_data')

            if url and strategy:
                domain = urlparse(url).netloc

                # Update success rates
                current_rate = self.pattern_learner['extraction_success_rates'][domain]
                # Exponential moving average
                alpha = 0.3
                new_rate = alpha * success_rate + (1 - alpha) * current_rate
                self.pattern_learner['extraction_success_rates'][domain] = new_rate

                # Store successful patterns
                if success_rate > 0.7:  # Only learn from successful extractions
                    pattern = {
                        'domain': domain,
                        'selectors': strategy.get('selectors', {}),
                        'success_rate': success_rate,
                        'timestamp': time.time()
                    }

                    self.pattern_learner['site_patterns'][domain].append(pattern)

                    # Limit stored patterns
                    if len(self.pattern_learner['site_patterns'][domain]) > 10:
                        self.pattern_learner['site_patterns'][domain] = \
                            self.pattern_learner['site_patterns'][domain][-10:]

        except Exception as e:
            self.logger.error(f"Error learning from feedback: {e}")

    async def _assess_content_quality(self, content: str, content_type: str = 'html') -> Dict[str, Any]:
        """Assess the quality of content using multiple metrics."""
        try:
            quality_metrics = {
                'completeness': 0.0,
                'accuracy': 0.0,
                'consistency': 0.0,
                'freshness': 0.0,
                'overall': 0.0
            }

            if content_type.lower() == 'html':
                soup = BeautifulSoup(content, 'html.parser')

                # Completeness: Check for essential elements
                completeness_score = 0.0
                essential_elements = ['title', 'body']
                for element in essential_elements:
                    if soup.find(element):
                        completeness_score += 1.0
                completeness_score = completeness_score / len(essential_elements)

                # Check for content density
                text_content = soup.get_text(strip=True)
                if text_content:
                    completeness_score += min(len(text_content) / 1000.0, 1.0)
                    completeness_score = min(completeness_score / 2, 1.0)

                quality_metrics['completeness'] = completeness_score

                # Accuracy: Check for broken elements
                accuracy_score = 1.0

                # Check for broken images
                images = soup.find_all('img')
                broken_images = sum(1 for img in images if not img.get('src'))
                if images:
                    accuracy_score -= (broken_images / len(images)) * 0.3

                # Check for broken links (basic check)
                links = soup.find_all('a')
                broken_links = sum(1 for link in links if not link.get('href'))
                if links:
                    accuracy_score -= (broken_links / len(links)) * 0.3

                quality_metrics['accuracy'] = max(accuracy_score, 0.0)

                # Consistency: Check for consistent structure
                consistency_score = 0.0

                # Check for consistent heading structure
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if headings:
                    h1_count = len(soup.find_all('h1'))
                    if h1_count == 1:  # Good practice
                        consistency_score += 0.5
                    elif h1_count == 0:
                        consistency_score += 0.2

                # Check for semantic HTML usage
                semantic_elements = soup.find_all(['article', 'section', 'nav', 'header', 'footer', 'main'])
                if semantic_elements:
                    consistency_score += 0.5

                quality_metrics['consistency'] = min(consistency_score, 1.0)

                # Freshness: Check for date indicators
                freshness_score = 0.0

                # Look for date patterns in text
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
                    r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
                ]

                for pattern in date_patterns:
                    if re.search(pattern, text_content):
                        freshness_score = 0.8
                        break

                # Check for time elements
                time_elements = soup.find_all('time')
                if time_elements:
                    freshness_score = max(freshness_score, 0.9)

                quality_metrics['freshness'] = freshness_score

            # Calculate overall quality
            weights = {
                'completeness': 0.3,
                'accuracy': 0.3,
                'consistency': 0.2,
                'freshness': 0.2
            }

            overall_score = sum(
                quality_metrics[metric] * weight
                for metric, weight in weights.items()
            )

            quality_metrics['overall'] = overall_score

            # Add quality assessment to history
            assessment = {
                'metrics': quality_metrics,
                'timestamp': time.time(),
                'content_length': len(content)
            }

            self.quality_assessor['assessment_history'].append(assessment)

            # Limit history size
            if len(self.quality_assessor['assessment_history']) > 100:
                self.quality_assessor['assessment_history'] = \
                    self.quality_assessor['assessment_history'][-100:]

            return {
                'quality_metrics': quality_metrics,
                'assessment': 'high' if overall_score > 0.8 else 'medium' if overall_score > 0.5 else 'low',
                'recommendations': self._generate_quality_recommendations(quality_metrics)
            }

        except Exception as e:
            self.logger.error(f"Error assessing content quality: {e}")
            return {
                'error': str(e),
                'quality_metrics': {},
                'assessment': 'unknown'
            }

    def _generate_quality_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []

        if metrics.get('completeness', 0) < 0.7:
            recommendations.append("Content appears incomplete. Consider checking for missing elements or data.")

        if metrics.get('accuracy', 0) < 0.8:
            recommendations.append("Found potential accuracy issues. Check for broken links or images.")

        if metrics.get('consistency', 0) < 0.6:
            recommendations.append("Content structure could be more consistent. Consider using semantic HTML.")

        if metrics.get('freshness', 0) < 0.5:
            recommendations.append("Content may be outdated. Look for recent timestamps or dates.")

        if not recommendations:
            recommendations.append("Content quality looks good!")

        return recommendations

    # ===== UTILITY METHODS =====

    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features dictionary to numerical vector for ML."""
        if not ML_AVAILABLE:
            return np.array([])

        # Simple feature vectorization
        vector = []

        # Numerical features
        numerical_features = [
            'text_length', 'word_count', 'form_count', 'table_count',
            'list_count', 'image_count', 'link_count', 'script_count', 'path_depth'
        ]

        for feature in numerical_features:
            vector.append(features.get(feature, 0))

        # Boolean features (convert to 0/1)
        boolean_features = [
            'has_subdomain', 'has_cart', 'has_price', 'has_product',
            'has_article', 'has_date', 'has_author', 'has_social_links',
            'has_share_buttons', 'has_thread', 'has_user_profile'
        ]

        for feature in boolean_features:
            vector.append(1 if features.get(feature, False) else 0)

        return np.array(vector)

    def _combine_classifications(self, ml_result: Optional[Dict], rule_result: Dict) -> Dict[str, Any]:
        """Combine ML and rule-based classification results."""
        if not ml_result:
            return rule_result

        # Simple combination: use ML if confidence is high, otherwise use rules
        ml_confidence = ml_result.get('confidence', 0.0)
        rule_confidence = rule_result.get('confidence', 0.0)

        if ml_confidence > 0.8:
            return ml_result
        elif rule_confidence > ml_confidence:
            return rule_result
        else:
            # Weighted combination
            combined_confidence = (ml_confidence + rule_confidence) / 2
            return {
                'type': rule_result.get('type', 'unknown'),
                'confidence': combined_confidence,
                'method': 'combined',
                'ml_result': ml_result,
                'rule_result': rule_result
            }

    async def _analyze_site_patterns(self) -> None:
        """Analyze stored site patterns for insights."""
        try:
            # Analyze success rates by domain
            for domain, patterns in self.pattern_learner['site_patterns'].items():
                if patterns:
                    avg_success = sum(p.get('success_rate', 0) for p in patterns) / len(patterns)
                    self.pattern_learner['extraction_success_rates'][domain] = avg_success

        except Exception as e:
            self.logger.error(f"Error analyzing site patterns: {e}")

    async def _update_extraction_strategies(self) -> None:
        """Update extraction strategies based on learned patterns."""
        try:
            # Update strategies for domains with good success rates
            for domain, success_rate in self.pattern_learner['extraction_success_rates'].items():
                if success_rate > 0.8:
                    patterns = self.pattern_learner['site_patterns'].get(domain, [])
                    if patterns:
                        # Use the most recent successful pattern
                        best_pattern = max(patterns, key=lambda p: p.get('success_rate', 0))
                        self.extraction_strategies[domain] = best_pattern

        except Exception as e:
            self.logger.error(f"Error updating extraction strategies: {e}")

    async def _cleanup_old_patterns(self) -> None:
        """Clean up old patterns to prevent memory bloat."""
        try:
            current_time = time.time()
            max_age = 7 * 24 * 3600  # 7 days

            # Clean site patterns
            for domain in list(self.pattern_learner['site_patterns'].keys()):
                patterns = self.pattern_learner['site_patterns'][domain]
                fresh_patterns = [
                    p for p in patterns
                    if current_time - p.get('timestamp', 0) < max_age
                ]

                if fresh_patterns:
                    self.pattern_learner['site_patterns'][domain] = fresh_patterns
                else:
                    del self.pattern_learner['site_patterns'][domain]

            # Clean content type patterns
            for domain in list(self.pattern_learner['content_type_patterns'].keys()):
                patterns = self.pattern_learner['content_type_patterns'][domain]
                fresh_patterns = [
                    p for p in patterns
                    if current_time - p.get('timestamp', 0) < max_age
                ]

                if fresh_patterns:
                    self.pattern_learner['content_type_patterns'][domain] = fresh_patterns
                else:
                    del self.pattern_learner['content_type_patterns'][domain]

        except Exception as e:
            self.logger.error(f"Error cleaning up old patterns: {e}")

    def _extract_training_data(self) -> Optional[List[Dict[str, Any]]]:
        """Extract training data from cache for ML model training."""
        try:
            training_data = []

            for cache_entry in self.analysis_cache.values():
                result = cache_entry.get('result', {})
                if result:
                    training_data.append(result)

            return training_data if len(training_data) > 10 else None

        except Exception as e:
            self.logger.error(f"Error extracting training data: {e}")
            return None

    async def _retrain_models(self, training_data: List[Dict[str, Any]]) -> None:
        """Retrain ML models with new training data."""
        try:
            if not ML_AVAILABLE:
                return

            # Extract text content for TF-IDF training
            text_content = []
            for data in training_data:
                # Extract text from structure information
                structure = data.get('structure', {})
                if isinstance(structure, dict):
                    text_content.append(str(structure))

            if len(text_content) > 10:
                # Retrain TF-IDF vectorizer
                self.ml_models['tfidf'].fit(text_content)
                self.logger.info(f"Retrained TF-IDF model with {len(text_content)} samples")

        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")


# Maintain backward compatibility
ContentRecognitionAgent = EnhancedContentRecognitionAgent
