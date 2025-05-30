"""
Data extraction agent for the web scraping system.
"""
import asyncio
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import re
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import PyPDF2
import xml.etree.ElementTree as ET
from playwright.async_api import async_playwright
import cv2
import pytesseract
from PIL import Image
import io
from concurrent.futures import ProcessPoolExecutor

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage
from models.task import Task, TaskStatus, TaskType


class DataExtractorAgent(Agent):
    """
    Agent responsible for extracting data from various sources and formats.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new data extractor agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="data_extractor")
        self.coordinator_id = coordinator_id

        # Enhanced cache configuration
        self.extraction_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 1000  # Maximum number of cache entries
        self.cache_memory_limit = 1024 * 1024 * 100  # 100MB
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size": 0
        }

        # Parallel processing configuration
        self.max_workers = min(32, (os.cpu_count() or 1) * 4)
        self.chunk_size = 1000  # Number of items to process in parallel
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

        # Data validation rules
        self.validation_rules: Dict[str, Dict[str, Any]] = {
            "required_fields": [],
            "field_types": {},
            "value_ranges": {},
            "patterns": {},
            "custom_validators": {}
        }

        # Register message handlers
        self.register_handler("extract_data", self._handle_extract_data)
        self.register_handler("validate_data", self._handle_validate_data)
        self.register_handler("clear_cache", self._handle_clear_cache)
        self.register_handler("get_cache_stats", self._handle_get_cache_stats)

        # Start periodic tasks
        self._start_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the extractor."""
        asyncio.create_task(self._periodic_cache_cleanup())
        asyncio.create_task(self._periodic_cache_stats_report())

    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up expired cache entries."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                if not self.running:
                    break

                await self._cleanup_cache()
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}", exc_info=True)

    async def _cleanup_cache(self) -> None:
        """Clean up cache based on TTL, size, and memory usage."""
        current_time = time.time()

        # Calculate adaptive TTL based on cache hit rate
        hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"]) if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
        adaptive_ttl = self.cache_ttl * (1 + hit_rate)  # Increase TTL for frequently accessed data

        # Remove expired entries
        expired_keys = [
            key for key, data in self.extraction_cache.items()
            if current_time - data["timestamp"] > adaptive_ttl
        ]
        for key in expired_keys:
            self._remove_cache_entry(key)

        # Check cache size
        if len(self.extraction_cache) > self.max_cache_size:
            # Remove least recently used entries
            sorted_entries = sorted(
                self.extraction_cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            for key, _ in sorted_entries[:len(sorted_entries) - self.max_cache_size]:
                self._remove_cache_entry(key)

        # Check memory usage
        if self.cache_stats["total_size"] > self.cache_memory_limit:
            # Remove largest entries until under limit
            sorted_by_size = sorted(
                self.extraction_cache.items(),
                key=lambda x: x[1]["size"]
            )
            while self.cache_stats["total_size"] > self.cache_memory_limit and sorted_by_size:
                key, _ = sorted_by_size.pop()
                self._remove_cache_entry(key)

    def _remove_cache_entry(self, key: str) -> None:
        """Remove a cache entry and update statistics."""
        if key in self.extraction_cache:
            self.cache_stats["total_size"] -= self.extraction_cache[key]["size"]
            self.cache_stats["evictions"] += 1
            del self.extraction_cache[key]

    async def _periodic_cache_stats_report(self) -> None:
        """Periodically report cache statistics."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                if not self.running:
                    break

                self.logger.info(f"Cache stats: {self.cache_stats}")

                # Reset stats
                self.cache_stats["hits"] = 0
                self.cache_stats["misses"] = 0
                self.cache_stats["evictions"] = 0
            except Exception as e:
                self.logger.error(f"Error in cache stats report: {str(e)}", exc_info=True)

    async def _handle_extract_data(self, message: Message) -> None:
        """Handle data extraction request."""
        try:
            url = message.data.get("url")
            extraction_type = message.data.get("extraction_type", "html")
            options = message.data.get("options", {})

            # Check cache first
            cache_key = f"{url}:{extraction_type}:{hash(str(options))}"
            if cache_key in self.extraction_cache:
                cache_entry = self.extraction_cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                    self.cache_stats["hits"] += 1
                    response = ResultMessage(
                        sender_id=self.agent_id,
                        recipient_id=message.sender_id,
                        task_id=message.task_id if hasattr(message, "task_id") else None,
                        result=cache_entry["data"]
                    )
                    self.outbox.put(response)
                    return

            self.cache_stats["misses"] += 1

            # Extract data
            if extraction_type == "html":
                result = await self._extract_html(url, options)
            elif extraction_type == "json":
                result = await self._extract_json(url, options)
            elif extraction_type == "xml":
                result = await self._extract_xml(url, options)
            elif extraction_type == "pdf":
                result = await self._extract_pdf(url, options)
            elif extraction_type == "image":
                result = await self._extract_image(url, options)
            elif extraction_type == "table":
                result = await self._extract_table(url, options)
            elif extraction_type == "dynamic":
                result = await self._extract_dynamic(url, options)
            else:
                raise ValueError(f"Unsupported extraction type: {extraction_type}")

            # Validate data if rules exist
            if self.validation_rules:
                validation_result = await self._validate_data(result)
                if not validation_result["valid"]:
                    raise ValueError(f"Data validation failed: {validation_result['errors']}")

            # Cache result
            self._cache_result(cache_key, result)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Data extraction error: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="extraction_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        self.running = False
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    async def _extract_html(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from HTML with parallel processing."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html = await response.text()

            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Get selectors
            selectors = options.get("selectors", {})

            # Process selectors in parallel
            loop = asyncio.get_event_loop()
            tasks = []

            for field, selector in selectors.items():
                tasks.append(loop.run_in_executor(
                    self.executor,
                    self._process_selector,
                    soup,
                    selector
                ))

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # Combine results
            data = {}
            for field, result in zip(selectors.keys(), results):
                data[field] = result

            return data

        except Exception as e:
            raise ValueError(f"HTML extraction failed: {str(e)}")

    def _process_selector(self, soup: BeautifulSoup, selector: Dict[str, Any]) -> Any:
        """Process a single selector in parallel."""
        try:
            selector_type = selector.get("type", "css")
            selector_value = selector.get("value")
            attribute = selector.get("attribute")
            multiple = selector.get("multiple", False)

            if selector_type == "css":
                elements = soup.select(selector_value)
            elif selector_type == "xpath":
                elements = soup.xpath(selector_value)
            else:
                raise ValueError(f"Unsupported selector type: {selector_type}")

            if not elements:
                return None

            if not multiple:
                element = elements[0]
                if attribute:
                    return element.get(attribute)
                return element.text.strip()

            results = []
            for element in elements:
                if attribute:
                    results.append(element.get(attribute))
                else:
                    results.append(element.text.strip())

            return results

        except Exception as e:
            raise ValueError(f"Selector processing failed: {str(e)}")

    async def _extract_json(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from JSON content.

        Args:
            url: The URL to extract from.
            options: Extraction parameters including paths.

        Returns:
            A dictionary containing the extracted data.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()

        result = {}

        # Extract based on paths
        for key, path in options.get("paths", {}).items():
            try:
                # Handle nested paths
                value = data
                for part in path.split("."):
                    if isinstance(value, dict):
                        value = value[part]
                    elif isinstance(value, list):
                        value = value[int(part)]
                    else:
                        raise ValueError(f"Invalid path: {path}")

                result[key] = value
            except (KeyError, IndexError, ValueError) as e:
                self.logger.warning(f"Error extracting path {path}: {str(e)}")
                result[key] = None

        return result

    async def _extract_xml(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from XML content.

        Args:
            url: The URL to extract from.
            options: Extraction parameters including XPath expressions.

        Returns:
            A dictionary containing the extracted data.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                xml = await response.text()

        root = ET.fromstring(xml)
        result = {}

        # Extract based on XPath expressions
        for key, xpath in options.get("xpaths", {}).items():
            try:
                elements = root.findall(xpath)
                if elements:
                    if len(elements) == 1:
                        result[key] = elements[0].text
                    else:
                        result[key] = [elem.text for elem in elements]
            except Exception as e:
                self.logger.warning(f"Error extracting XPath {xpath}: {str(e)}")
                result[key] = None

        return result

    async def _extract_pdf(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from PDF content.

        Args:
            url: The URL to extract from.
            options: Extraction parameters.

        Returns:
            A dictionary containing the extracted data.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                pdf_data = await response.read()

        result = {
            "text": "",
            "metadata": {},
            "pages": []
        }

        # Read PDF
        pdf_file = io.BytesIO(pdf_data)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Extract text from each page
        for page in pdf_reader.pages:
            result["pages"].append(page.extract_text())

        # Combine all pages
        result["text"] = "\n".join(result["pages"])

        # Extract metadata
        if pdf_reader.metadata:
            result["metadata"] = {
                "title": pdf_reader.metadata.get("/Title"),
                "author": pdf_reader.metadata.get("/Author"),
                "subject": pdf_reader.metadata.get("/Subject"),
                "creator": pdf_reader.metadata.get("/Creator"),
                "producer": pdf_reader.metadata.get("/Producer"),
                "creation_date": pdf_reader.metadata.get("/CreationDate")
            }

        return result

    async def _extract_image(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from image content.

        Args:
            url: The URL to extract from.
            options: Extraction parameters.

        Returns:
            A dictionary containing the extracted data.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                image_data = await response.read()

        result = {
            "text": "",
            "metadata": {},
            "size": None,
            "format": None
        }

        # Read image
        image = Image.open(io.BytesIO(image_data))

        # Extract text using OCR if requested
        if options.get("extract_text", False):
            result["text"] = pytesseract.image_to_string(image)

        # Extract metadata
        result["metadata"] = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size
        }

        # Extract features if requested
        if options.get("extract_features", False):
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Extract features
            result["features"] = {
                "edges": cv2.Canny(cv_image, 100, 200).tolist(),
                "corners": cv2.goodFeaturesToTrack(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY), 25, 0.01, 10).tolist()
            }

        return result

    async def _extract_table(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from HTML tables.

        Args:
            url: The URL to extract from.
            options: Extraction parameters including table selectors.

        Returns:
            A dictionary containing the extracted data.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')
        result = {}

        # Extract tables based on selectors
        for key, selector in options.get("table_selectors", {}).items():
            tables = soup.select(selector)
            if tables:
                # Convert tables to pandas DataFrames
                dfs = []
                for table in tables:
                    df = pd.read_html(str(table))[0]
                    dfs.append(df)

                # Combine tables if multiple found
                if len(dfs) == 1:
                    result[key] = dfs[0].to_dict(orient="records")
                else:
                    result[key] = [df.to_dict(orient="records") for df in dfs]

        return result

    async def _extract_dynamic(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from dynamic content requiring JavaScript execution.

        Args:
            url: The URL to extract from.
            options: Extraction parameters including wait selectors.

        Returns:
            A dictionary containing the extracted data.
        """
        if not self.context:
            raise RuntimeError("Browser context not initialized")

        page = await self.context.new_page()
        try:
            # Navigate to URL
            await page.goto(url, wait_until="networkidle")

            # Wait for specific elements if requested
            if "wait_for" in options:
                await page.wait_for_selector(options["wait_for"])

            # Execute custom JavaScript if provided
            if "script" in options:
                result = await page.evaluate(options["script"])
            else:
                # Extract data using selectors
                result = {}
                for key, selector in options.get("selectors", {}).items():
                    elements = await page.query_selector_all(selector)
                    if elements:
                        if len(elements) == 1:
                            result[key] = await elements[0].text_content()
                        else:
                            result[key] = [await elem.text_content() for elem in elements]

            return result

        finally:
            await page.close()

    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted data against rules."""
        errors = []

        # Check required fields
        for field in self.validation_rules["required_fields"]:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check field types
        for field, expected_type in self.validation_rules["field_types"].items():
            if field in data:
                if not isinstance(data[field], eval(expected_type)):
                    errors.append(f"Invalid type for field {field}: expected {expected_type}")

        # Check value ranges
        for field, (min_val, max_val) in self.validation_rules["value_ranges"].items():
            if field in data:
                if not (min_val <= data[field] <= max_val):
                    errors.append(f"Value out of range for field {field}")

        # Check patterns
        for field, pattern in self.validation_rules["patterns"].items():
            if field in data:
                if not re.match(pattern, str(data[field])):
                    errors.append(f"Value does not match pattern for field {field}")

        # Run custom validators
        for field, validator in self.validation_rules["custom_validators"].items():
            if field in data:
                try:
                    if not validator(data[field]):
                        errors.append(f"Custom validation failed for field {field}")
                except Exception as e:
                    errors.append(f"Custom validation error for field {field}: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _cache_result(self, key: str, data: Any) -> None:
        """Cache extraction result with size limits."""
        try:
            # Calculate data size
            data_size = len(str(data).encode())

            # Check if cache is full
            if len(self.extraction_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(
                    self.extraction_cache.keys(),
                    key=lambda k: self.extraction_cache[k]["timestamp"]
                )
                del self.extraction_cache[oldest_key]
                self.cache_stats["evictions"] += 1

            # Check memory limit
            while self.cache_stats["total_size"] + data_size > self.cache_memory_limit:
                if not self.extraction_cache:
                    break
                oldest_key = min(
                    self.extraction_cache.keys(),
                    key=lambda k: self.extraction_cache[k]["timestamp"]
                )
                self.cache_stats["total_size"] -= len(str(self.extraction_cache[oldest_key]["data"]).encode())
                del self.extraction_cache[oldest_key]
                self.cache_stats["evictions"] += 1

            # Add to cache
            self.extraction_cache[key] = {
                "data": data,
                "timestamp": time.time()
            }
            self.cache_stats["total_size"] += data_size

        except Exception as e:
            self.logger.error(f"Error caching result: {str(e)}")

    async def _handle_validate_data(self, message: Message) -> None:
        """Handle data validation request."""
        try:
            data = message.data.get("data", {})
            rules = message.data.get("rules", {})

            # Update validation rules
            if rules:
                self.validation_rules.update(rules)

            # Validate data
            validation_result = await self._validate_data(data)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=validation_result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="validation_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_clear_cache(self, message: Message) -> None:
        """Handle cache clear request."""
        try:
            self.extraction_cache.clear()
            self.cache_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_size": 0
            }

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"status": "Cache cleared"}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Cache clear error: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="cache_clear_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_get_cache_stats(self, message: Message) -> None:
        """Handle cache stats request."""
        try:
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={
                    "stats": self.cache_stats,
                    "cache_size": len(self.extraction_cache),
                    "cache_ttl": self.cache_ttl,
                    "max_cache_size": self.max_cache_size,
                    "cache_memory_limit": self.cache_memory_limit
                }
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Cache stats error: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="cache_stats_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _send_message(self, message: Message) -> None:
        """
        Send a message to another agent.

        Args:
            message: The message to send.
        """
        # Add message to outbox
        self.outbox.put(message)

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        if task.type == TaskType.EXTRACT_DATA:
            url = task.parameters.get("url")
            extraction_type = task.parameters.get("extraction_type", "html")
            options = task.parameters.get("options", {})

            # Check cache first
            cache_key = f"{url}:{extraction_type}:{hash(str(options))}"
            if cache_key in self.extraction_cache:
                cache_entry = self.extraction_cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                    self.cache_stats["hits"] += 1
                    return cache_entry["data"]

            self.cache_stats["misses"] += 1

            # Extract data
            if extraction_type == "html":
                result = await self._extract_html(url, options)
            elif extraction_type == "json":
                result = await self._extract_json(url, options)
            elif extraction_type == "xml":
                result = await self._extract_xml(url, options)
            elif extraction_type == "pdf":
                result = await self._extract_pdf(url, options)
            elif extraction_type == "image":
                result = await self._extract_image(url, options)
            elif extraction_type == "table":
                result = await self._extract_table(url, options)
            elif extraction_type == "dynamic":
                result = await self._extract_dynamic(url, options)
            else:
                raise ValueError(f"Unsupported extraction type: {extraction_type}")

            # Validate data if rules exist
            if self.validation_rules:
                validation_result = await self._validate_data(result)
                if not validation_result["valid"]:
                    raise ValueError(f"Data validation failed: {validation_result['errors']}")

            # Cache result
            self._cache_result(cache_key, result)

            return result

        elif task.type == TaskType.VALIDATE_DATA:
            data = task.parameters.get("data")
            rules = task.parameters.get("rules")

            if rules:
                # Use provided rules
                old_rules = self.validation_rules
                self.validation_rules = rules
                result = await self._validate_data(data)
                self.validation_rules = old_rules
            else:
                # Use default rules
                result = await self._validate_data(data)

            return result

        elif task.type == TaskType.CLEAR_CACHE:
            self.extraction_cache.clear()
            self.cache_stats["total_size"] = 0

            return {"status": "success", "message": "Cache cleared"}

        else:
            raise ValueError(f"Unsupported task type for data extractor agent: {task.type}")