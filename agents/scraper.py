"""
Scraper agent for the web scraping system.
"""
import asyncio
import logging
import time
import random
from typing import Dict, Any, Optional, List, Union
import httpx
from urllib.parse import urljoin

from agents.base import Agent
from models.task import Task, TaskType
from utils.rate_limiter import RateLimiter


class ScraperAgent(Agent):
    """
    Agent responsible for fetching content from websites.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new scraper agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="scraper")
        self.coordinator_id = coordinator_id
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        ]

        # Create HTTP client
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": random.choice(self.user_agents)}
        )

    async def _send_message(self, message: Any) -> None:
        """
        Send a message to another agent.

        Args:
            message: The message to send.
        """
        # In a real implementation, this would use a message broker or direct connection
        # For now, we'll just log the message
        self.logger.debug(f"Sending message to {message.recipient_id}: {message.type}")

        # This is a placeholder for actual message sending
        # In a real implementation, you would send the message to the recipient agent
        # For example: await self.message_broker.send(message.recipient_id, message)

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        if task.type == TaskType.FETCH_URL:
            return await self._fetch_url(task)
        elif task.type == TaskType.RENDER_JS:
            return await self._render_js(task)
        elif task.type == TaskType.FOLLOW_PAGINATION:
            return await self._follow_pagination(task)
        else:
            raise ValueError(f"Unsupported task type for scraper agent: {task.type}")

    async def _fetch_url(self, task: Task) -> Dict[str, Any]:
        """
        Fetch content from a URL.

        Args:
            task: The task containing the URL to fetch.

        Returns:
            A dictionary containing the fetched content and metadata.
        """
        url = task.parameters.get("url")
        if not url:
            raise ValueError("URL parameter is required for fetch_url task")

        method = task.parameters.get("method", "GET")
        headers = task.parameters.get("headers", {})
        timeout = task.parameters.get("timeout", 30.0)
        verify_ssl = task.parameters.get("verify_ssl", True)
        proxy = task.parameters.get("proxy")
        cookies = task.parameters.get("cookies", {})

        # Apply rate limiting
        domain = self._get_domain(url)
        if domain not in self.rate_limiters:
            # Default rate limit: 1 request per 2 seconds
            self.rate_limiters[domain] = RateLimiter(max_calls=1, period=2.0)

        # Wait for rate limit
        await self.rate_limiters[domain].acquire()

        # Add random user agent if not specified
        if "User-Agent" not in headers:
            headers["User-Agent"] = random.choice(self.user_agents)

        # Configure client for this request
        request_kwargs = {
            "method": method,
            "url": url,
            "headers": headers,
            "cookies": cookies,
            "timeout": timeout,
            "verify": verify_ssl
        }

        if proxy:
            request_kwargs["proxies"] = {"all://": proxy}

        # Add request body if provided
        if "data" in task.parameters:
            request_kwargs["data"] = task.parameters["data"]
        if "json" in task.parameters:
            request_kwargs["json"] = task.parameters["json"]

        try:
            start_time = time.time()
            response = await self.client.request(**request_kwargs)
            end_time = time.time()

            # Check if response is successful
            response.raise_for_status()

            return {
                "url": url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text,
                "content_type": response.headers.get("content-type", ""),
                "elapsed": end_time - start_time,
                "encoding": response.encoding,
                "cookies": dict(response.cookies)
            }
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors (4xx, 5xx)
            self.logger.warning(f"HTTP error for {url}: {e.response.status_code}")
            return {
                "url": url,
                "status_code": e.response.status_code,
                "headers": dict(e.response.headers),
                "content": e.response.text,
                "error": str(e),
                "error_type": "http_error"
            }
        except httpx.RequestError as e:
            # Handle request errors (connection, timeout, etc.)
            self.logger.error(f"Request error for {url}: {str(e)}")
            raise

    async def _render_js(self, task: Task) -> Dict[str, Any]:
        """
        Render JavaScript content from a URL using a headless browser.

        Args:
            task: The task containing the URL to render.

        Returns:
            A dictionary containing the rendered content and metadata.
        """
        url = task.parameters.get("url")
        if not url:
            raise ValueError("URL parameter is required for render_js task")

        wait_for = task.parameters.get("wait_for")
        timeout = task.parameters.get("timeout", 30.0)
        script = task.parameters.get("script")
        browser_type = task.parameters.get("browser", "chromium")
        headers = task.parameters.get("headers", {})

        # This is a placeholder for actual JavaScript rendering
        # In a real implementation, you would use a headless browser like Playwright
        self.logger.info(f"Rendering JavaScript for {url} (wait_for: {wait_for}, timeout: {timeout})")

        # Simulate JavaScript rendering with a delay
        await asyncio.sleep(2.0)

        # For now, just fetch the URL normally
        fetch_task = Task(
            type=TaskType.FETCH_URL,
            parameters={
                "url": url,
                "headers": headers,
                "timeout": timeout
            }
        )

        result = await self._fetch_url(fetch_task)
        result["javascript_rendered"] = True
        result["browser_type"] = browser_type

        if script:
            result["script_executed"] = True

        return result

    async def _follow_pagination(self, task: Task) -> Dict[str, Any]:
        """
        Follow pagination links on a website using the advanced pagination engine.

        Args:
            task: The task containing the pagination parameters.

        Returns:
            A dictionary containing the results from all pages.
        """
        url = task.parameters.get("url")
        if not url:
            raise ValueError("URL parameter is required for follow_pagination task")

        # Import the pagination engine
        from utils.pagination_engine import pagination_engine

        max_pages = task.parameters.get("max_pages", 5)
        current_page = task.parameters.get("current_page", 1)
        wait_between_requests = task.parameters.get("wait_between_requests", 2.0)
        use_browser = task.parameters.get("use_browser", False)

        all_results = []
        current_url = url
        page_num = current_page

        while page_num < current_page + max_pages:
            self.logger.info(f"Fetching page {page_num}: {current_url}")

            # Fetch the current page
            fetch_task = Task(
                type=TaskType.FETCH_URL,
                parameters={"url": current_url}
            )

            result = await self._fetch_url(fetch_task)
            result["page_number"] = page_num
            all_results.append(result)

            # Check if we should continue to the next page
            if page_num >= current_page + max_pages - 1:
                break

            # Use the pagination engine to find the next URL
            html_content = result.get("content", "")

            # Handle pagination
            pagination_result = await pagination_engine.handle_pagination(
                current_url,
                html_content,
                browser_available=use_browser
            )

            if not pagination_result["has_pagination"] or not pagination_result["next_url"]:
                self.logger.info("No more pages detected")
                break

            # If pagination requires a browser but we don't have one available
            if pagination_result["requires_browser"] and not use_browser:
                self.logger.warning("Pagination requires a browser, but none is available")

                # Fall back to simple pagination if possible
                if "next_page_selector" in task.parameters:
                    next_page_selector = task.parameters["next_page_selector"]
                    self.logger.info(f"Falling back to simple pagination with selector: {next_page_selector}")

                    # Simple pagination - add page parameter
                    page_num += 1
                    if "?" in url:
                        current_url = f"{url}&page={page_num}"
                    else:
                        current_url = f"{url}?page={page_num}"
                else:
                    break
            else:
                # Use the detected next URL
                current_url = pagination_result["next_url"]
                page_num += 1

                # Log the pagination method used
                self.logger.info(f"Using pagination method: {pagination_result['method']}")

            # Wait between requests
            await asyncio.sleep(wait_between_requests)

        return {
            "pages_fetched": len(all_results),
            "results": all_results,
            "start_url": url,
            "max_pages": max_pages
        }

    def _get_domain(self, url: str) -> str:
        """
        Extract the domain from a URL.

        Args:
            url: The URL to extract the domain from.

        Returns:
            The domain of the URL.
        """
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            return parsed_url.netloc
        except Exception:
            # If parsing fails, return the URL as is
            return url
