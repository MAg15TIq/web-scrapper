"""
Content Recognition Agent for the self-aware web scraping system.

This agent identifies and categorizes any input content.
"""
import asyncio
import logging
import time
import re
import json
import os
import io
import mimetypes
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import uuid
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.intelligence import (
    ContentType, InputType, WebsiteType, AgentCapability,
    AgentProfile, ContentAnalysisResult
)


class ContentRecognitionAgent(Agent):
    """
    Content Recognition Agent that identifies and categorizes any input content.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new Content Recognition Agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        super().__init__(agent_id=agent_id, agent_type="content_recognition", coordinator_id=coordinator_id)

        # HTTP client for making requests
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        )

        # Content type detection patterns
        self.content_patterns = self._initialize_content_patterns()

        # Cache for content analysis results
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}

        # Register message handlers
        self.register_handler("recognize_content", self._handle_recognize_content)
        self.register_handler("detect_content_type", self._handle_detect_content_type)
        self.register_handler("analyze_structure", self._handle_analyze_structure)

        # Start periodic tasks
        self._start_periodic_tasks()

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
