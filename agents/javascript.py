"""
JavaScript rendering agent for the web scraping system.
"""
import os
import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from agents.base import Agent
from models.task import Task, TaskType


class BrowserPool:
    """
    Manages a pool of browser instances for JavaScript rendering.
    """
    def __init__(self, max_browsers: int = 3, browser_type: str = "chromium"):
        """
        Initialize a new browser pool.
        
        Args:
            max_browsers: Maximum number of browser instances to keep in the pool.
            browser_type: Type of browser to use ("chromium", "firefox", or "webkit").
        """
        self.max_browsers = max_browsers
        self.browser_type = browser_type
        self.browsers = {}
        self.pages = {}
        self.contexts = {}
        self.last_used = {}
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger("browser_pool")
        self.playwright = None
    
    async def initialize(self) -> None:
        """Initialize the browser pool."""
        try:
            from playwright.async_api import async_playwright
            self.playwright = await async_playwright().start()
            self.logger.info(f"Initialized browser pool with {self.browser_type} browsers")
        except ImportError:
            self.logger.error("Playwright not installed. Please install with: pip install playwright")
            self.logger.error("Then run: playwright install")
            raise
    
    async def get_browser(self) -> Any:
        """
        Get a browser instance from the pool.
        
        Returns:
            A browser instance.
        """
        async with self.lock:
            # Initialize if not already done
            if not self.playwright:
                await self.initialize()
            
            # Check if we have available browsers
            if len(self.browsers) < self.max_browsers:
                # Create a new browser instance
                browser_id = f"browser-{len(self.browsers) + 1}"
                
                # Launch the browser
                if self.browser_type == "chromium":
                    browser = await self.playwright.chromium.launch(headless=True)
                elif self.browser_type == "firefox":
                    browser = await self.playwright.firefox.launch(headless=True)
                elif self.browser_type == "webkit":
                    browser = await self.playwright.webkit.launch(headless=True)
                else:
                    raise ValueError(f"Unsupported browser type: {self.browser_type}")
                
                self.browsers[browser_id] = browser
                self.last_used[browser_id] = time.time()
                
                self.logger.debug(f"Created new browser instance: {browser_id}")
                return browser_id, browser
            
            # Find the least recently used browser
            lru_browser_id = min(self.last_used.items(), key=lambda x: x[1])[0]
            self.last_used[lru_browser_id] = time.time()
            
            return lru_browser_id, self.browsers[lru_browser_id]
    
    async def create_context(self, browser_id: str, user_agent: Optional[str] = None, 
                           viewport: Optional[Dict[str, int]] = None, 
                           locale: Optional[str] = None,
                           timezone_id: Optional[str] = None,
                           permissions: Optional[List[str]] = None) -> str:
        """
        Create a new browser context.
        
        Args:
            browser_id: ID of the browser to create the context in.
            user_agent: User agent string to use.
            viewport: Viewport dimensions.
            locale: Locale to use.
            timezone_id: Timezone ID to use.
            permissions: List of permissions to grant.
            
        Returns:
            The ID of the created context.
        """
        if browser_id not in self.browsers:
            raise ValueError(f"Unknown browser ID: {browser_id}")
        
        browser = self.browsers[browser_id]
        
        # Set up context options
        context_options = {}
        if user_agent:
            context_options["user_agent"] = user_agent
        if viewport:
            context_options["viewport"] = viewport
        if locale:
            context_options["locale"] = locale
        if timezone_id:
            context_options["timezone_id"] = timezone_id
        
        # Create the context
        context = await browser.new_context(**context_options)
        
        # Set permissions if provided
        if permissions:
            await context.grant_permissions(permissions)
        
        # Generate context ID
        context_id = f"context-{browser_id}-{len(self.contexts) + 1}"
        self.contexts[context_id] = context
        
        self.logger.debug(f"Created new browser context: {context_id}")
        return context_id
    
    async def create_page(self, context_id: str) -> str:
        """
        Create a new page in a browser context.
        
        Args:
            context_id: ID of the context to create the page in.
            
        Returns:
            The ID of the created page.
        """
        if context_id not in self.contexts:
            raise ValueError(f"Unknown context ID: {context_id}")
        
        context = self.contexts[context_id]
        
        # Create the page
        page = await context.new_page()
        
        # Generate page ID
        page_id = f"page-{context_id}-{len(self.pages) + 1}"
        self.pages[page_id] = page
        
        self.logger.debug(f"Created new page: {page_id}")
        return page_id
    
    async def get_page(self, page_id: str) -> Any:
        """
        Get a page by ID.
        
        Args:
            page_id: ID of the page to get.
            
        Returns:
            The page object.
        """
        if page_id not in self.pages:
            raise ValueError(f"Unknown page ID: {page_id}")
        
        return self.pages[page_id]
    
    async def close_page(self, page_id: str) -> None:
        """
        Close a page.
        
        Args:
            page_id: ID of the page to close.
        """
        if page_id not in self.pages:
            return
        
        page = self.pages[page_id]
        await page.close()
        del self.pages[page_id]
        
        self.logger.debug(f"Closed page: {page_id}")
    
    async def close_context(self, context_id: str) -> None:
        """
        Close a browser context.
        
        Args:
            context_id: ID of the context to close.
        """
        if context_id not in self.contexts:
            return
        
        # Close all pages in this context
        pages_to_close = [page_id for page_id in self.pages if context_id in page_id]
        for page_id in pages_to_close:
            await self.close_page(page_id)
        
        # Close the context
        context = self.contexts[context_id]
        await context.close()
        del self.contexts[context_id]
        
        self.logger.debug(f"Closed context: {context_id}")
    
    async def close_browser(self, browser_id: str) -> None:
        """
        Close a browser.
        
        Args:
            browser_id: ID of the browser to close.
        """
        if browser_id not in self.browsers:
            return
        
        # Close all contexts in this browser
        contexts_to_close = [context_id for context_id in self.contexts if browser_id in context_id]
        for context_id in contexts_to_close:
            await self.close_context(context_id)
        
        # Close the browser
        browser = self.browsers[browser_id]
        await browser.close()
        del self.browsers[browser_id]
        if browser_id in self.last_used:
            del self.last_used[browser_id]
        
        self.logger.debug(f"Closed browser: {browser_id}")
    
    async def close_all(self) -> None:
        """Close all browsers, contexts, and pages."""
        # Close all browsers
        for browser_id in list(self.browsers.keys()):
            await self.close_browser(browser_id)
        
        # Stop Playwright
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        
        self.logger.info("Closed all browsers and stopped Playwright")


class JavaScriptAgent(Agent):
    """
    Agent responsible for rendering JavaScript and interacting with dynamic websites.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new JavaScript agent.
        
        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="javascript")
        self.coordinator_id = coordinator_id
        self.browser_pool = BrowserPool()
        self.screenshots_dir = Path("screenshots")
        
        # Create screenshots directory if it doesn't exist
        os.makedirs(self.screenshots_dir, exist_ok=True)
    
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
        if task.type == TaskType.RENDER_PAGE:
            return await self._render_page(task)
        elif task.type == TaskType.INTERACT_WITH_PAGE:
            return await self._interact_with_page(task)
        elif task.type == TaskType.SCROLL_PAGE:
            return await self._scroll_page(task)
        elif task.type == TaskType.TAKE_SCREENSHOT:
            return await self._take_screenshot(task)
        elif task.type == TaskType.EXECUTE_SCRIPT:
            return await self._execute_script(task)
        else:
            raise ValueError(f"Unsupported task type for JavaScript agent: {task.type}")
    
    async def _render_page(self, task: Task) -> Dict[str, Any]:
        """
        Render a page with a headless browser.
        
        Args:
            task: The task containing the page to render.
            
        Returns:
            A dictionary containing the rendered page content and metadata.
        """
        url = task.parameters.get("url")
        if not url:
            raise ValueError("URL parameter is required for render_page task")
        
        wait_for = task.parameters.get("wait_for")
        timeout = task.parameters.get("timeout", 30)
        viewport = task.parameters.get("viewport", {"width": 1920, "height": 1080})
        browser_type = task.parameters.get("browser", "chromium")
        device = task.parameters.get("device", "Desktop")
        headers = task.parameters.get("headers", {})
        cookies = task.parameters.get("cookies", {})
        wait_until = task.parameters.get("wait_until", "networkidle")
        
        # Set browser type
        self.browser_pool.browser_type = browser_type
        
        # Get a browser
        browser_id, _ = await self.browser_pool.get_browser()
        
        # Create a context with the specified user agent
        user_agent = headers.get("User-Agent")
        context_id = await self.browser_pool.create_context(
            browser_id=browser_id,
            user_agent=user_agent,
            viewport=viewport
        )
        
        # Create a page
        page_id = await self.browser_pool.create_page(context_id)
        page = await self.browser_pool.get_page(page_id)
        
        # Set extra HTTP headers
        if headers:
            await page.set_extra_http_headers(headers)
        
        # Set cookies
        if cookies:
            formatted_cookies = []
            for name, value in cookies.items():
                cookie = {"name": name, "value": value, "url": url}
                formatted_cookies.append(cookie)
            await page.context.add_cookies(formatted_cookies)
        
        try:
            # Navigate to the URL
            self.logger.info(f"Navigating to {url}")
            response = await page.goto(url, wait_until=wait_until, timeout=timeout * 1000)
            
            # Wait for selector if specified
            if wait_for:
                self.logger.info(f"Waiting for selector: {wait_for}")
                await page.wait_for_selector(wait_for, timeout=timeout * 1000)
            
            # Get page content
            content = await page.content()
            
            # Get page title
            title = await page.title()
            
            # Get response status and headers
            status = response.status if response else None
            response_headers = response.headers if response else {}
            
            # Take a screenshot
            screenshot_path = self.screenshots_dir / f"{page_id}.png"
            await page.screenshot(path=str(screenshot_path))
            
            return {
                "page_id": page_id,
                "context_id": context_id,
                "browser_id": browser_id,
                "url": url,
                "title": title,
                "status_code": status,
                "headers": response_headers,
                "content": content,
                "screenshot_path": str(screenshot_path),
                "viewport": viewport,
                "browser_type": browser_type
            }
        
        except Exception as e:
            # Close the page on error
            await self.browser_pool.close_page(page_id)
            self.logger.error(f"Error rendering page {url}: {str(e)}")
            raise
    
    async def _interact_with_page(self, task: Task) -> Dict[str, Any]:
        """
        Interact with elements on a page.
        
        Args:
            task: The task containing the interactions to perform.
            
        Returns:
            A dictionary containing the result of the interactions.
        """
        page_id = task.parameters.get("page_id")
        if not page_id:
            raise ValueError("page_id parameter is required for interact_with_page task")
        
        actions = task.parameters.get("actions", [])
        if not actions:
            raise ValueError("actions parameter is required for interact_with_page task")
        
        timeout = task.parameters.get("timeout", 30)
        
        # Get the page
        try:
            page = await self.browser_pool.get_page(page_id)
        except ValueError as e:
            self.logger.error(f"Error getting page {page_id}: {str(e)}")
            raise
        
        results = []
        
        try:
            # Perform each action
            for i, action in enumerate(actions):
                action_type = action.get("type")
                selector = action.get("selector")
                
                self.logger.info(f"Performing action {i+1}/{len(actions)}: {action_type}")
                
                if action_type == "click":
                    if not selector:
                        raise ValueError(f"selector is required for click action")
                    await page.click(selector, timeout=timeout * 1000)
                    results.append({"action": "click", "selector": selector, "success": True})
                
                elif action_type == "fill":
                    if not selector:
                        raise ValueError(f"selector is required for fill action")
                    value = action.get("value", "")
                    await page.fill(selector, value, timeout=timeout * 1000)
                    results.append({"action": "fill", "selector": selector, "value": value, "success": True})
                
                elif action_type == "select":
                    if not selector:
                        raise ValueError(f"selector is required for select action")
                    value = action.get("value")
                    if value:
                        await page.select_option(selector, value=value, timeout=timeout * 1000)
                    else:
                        label = action.get("label")
                        if label:
                            await page.select_option(selector, label=label, timeout=timeout * 1000)
                        else:
                            index = action.get("index")
                            if index is not None:
                                await page.select_option(selector, index=index, timeout=timeout * 1000)
                    results.append({"action": "select", "selector": selector, "success": True})
                
                elif action_type == "wait":
                    duration = action.get("duration", 1.0)
                    await asyncio.sleep(duration)
                    results.append({"action": "wait", "duration": duration, "success": True})
                
                elif action_type == "wait_for_selector":
                    if not selector:
                        raise ValueError(f"selector is required for wait_for_selector action")
                    state = action.get("state", "visible")
                    await page.wait_for_selector(selector, state=state, timeout=timeout * 1000)
                    results.append({"action": "wait_for_selector", "selector": selector, "success": True})
                
                elif action_type == "wait_for_navigation":
                    wait_until = action.get("wait_until", "networkidle")
                    async with page.expect_navigation(wait_until=wait_until, timeout=timeout * 1000):
                        if selector:
                            await page.click(selector)
                        else:
                            # If no selector, assume navigation is triggered by previous action
                            pass
                    results.append({"action": "wait_for_navigation", "success": True})
                
                elif action_type == "check":
                    if not selector:
                        raise ValueError(f"selector is required for check action")
                    await page.check(selector, timeout=timeout * 1000)
                    results.append({"action": "check", "selector": selector, "success": True})
                
                elif action_type == "uncheck":
                    if not selector:
                        raise ValueError(f"selector is required for uncheck action")
                    await page.uncheck(selector, timeout=timeout * 1000)
                    results.append({"action": "uncheck", "selector": selector, "success": True})
                
                elif action_type == "press":
                    if not selector:
                        raise ValueError(f"selector is required for press action")
                    key = action.get("key")
                    if not key:
                        raise ValueError(f"key is required for press action")
                    await page.press(selector, key, timeout=timeout * 1000)
                    results.append({"action": "press", "selector": selector, "key": key, "success": True})
                
                else:
                    self.logger.warning(f"Unknown action type: {action_type}")
                    results.append({"action": action_type, "success": False, "error": "Unknown action type"})
            
            # Get updated content
            content = await page.content()
            
            # Get page title
            title = await page.title()
            
            # Get current URL
            current_url = page.url
            
            # Take a screenshot
            screenshot_path = self.screenshots_dir / f"{page_id}-after-interaction.png"
            await page.screenshot(path=str(screenshot_path))
            
            return {
                "page_id": page_id,
                "actions_performed": results,
                "content": content,
                "title": title,
                "url": current_url,
                "screenshot_path": str(screenshot_path)
            }
        
        except Exception as e:
            self.logger.error(f"Error interacting with page {page_id}: {str(e)}")
            results.append({"action": action_type, "success": False, "error": str(e)})
            raise
    
    async def _scroll_page(self, task: Task) -> Dict[str, Any]:
        """
        Scroll a page to load lazy content.
        
        Args:
            task: The task containing the scrolling parameters.
            
        Returns:
            A dictionary containing the result of the scrolling.
        """
        page_id = task.parameters.get("page_id")
        if not page_id:
            raise ValueError("page_id parameter is required for scroll_page task")
        
        scroll_behavior = task.parameters.get("scroll_behavior", "smooth")
        max_scrolls = task.parameters.get("max_scrolls", 10)
        scroll_delay = task.parameters.get("scroll_delay", 1.0)
        scroll_amount = task.parameters.get("scroll_amount", 800)
        target_selector = task.parameters.get("target_selector")
        wait_for_selector = task.parameters.get("wait_for_selector")
        
        # Get the page
        try:
            page = await self.browser_pool.get_page(page_id)
        except ValueError as e:
            self.logger.error(f"Error getting page {page_id}: {str(e)}")
            raise
        
        try:
            # If target selector is provided, scroll to it
            if target_selector:
                self.logger.info(f"Scrolling to selector: {target_selector}")
                await page.evaluate(f"""
                    (selector) => {{
                        const element = document.querySelector(selector);
                        if (element) {{
                            element.scrollIntoView({{ behavior: '{scroll_behavior}', block: 'center' }});
                        }}
                    }}
                """, target_selector)
                await asyncio.sleep(scroll_delay)
            
            # Otherwise, perform incremental scrolling
            else:
                self.logger.info(f"Performing incremental scrolling ({max_scrolls} scrolls)")
                
                for i in range(max_scrolls):
                    self.logger.debug(f"Scroll {i+1}/{max_scrolls}")
                    
                    # Scroll down
                    await page.evaluate(f"""
                        window.scrollBy({{
                            top: {scroll_amount},
                            behavior: '{scroll_behavior}'
                        }});
                    """)
                    
                    # Wait for the specified delay
                    await asyncio.sleep(scroll_delay)
                    
                    # If wait_for_selector is provided, check if it's visible
                    if wait_for_selector:
                        is_visible = await page.is_visible(wait_for_selector)
                        if is_visible:
                            self.logger.info(f"Found selector after {i+1} scrolls: {wait_for_selector}")
                            break
            
            # Get updated content
            content = await page.content()
            
            # Take a screenshot
            screenshot_path = self.screenshots_dir / f"{page_id}-after-scrolling.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            
            # Get page height
            page_height = await page.evaluate("""
                Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight,
                    document.body.offsetHeight,
                    document.documentElement.offsetHeight,
                    document.body.clientHeight,
                    document.documentElement.clientHeight
                )
            """)
            
            return {
                "page_id": page_id,
                "scrolls_performed": min(max_scrolls, i + 1) if 'i' in locals() else 1,
                "content": content,
                "page_height": page_height,
                "screenshot_path": str(screenshot_path),
                "target_found": is_visible if wait_for_selector and 'is_visible' in locals() else None
            }
        
        except Exception as e:
            self.logger.error(f"Error scrolling page {page_id}: {str(e)}")
            raise
    
    async def _take_screenshot(self, task: Task) -> Dict[str, Any]:
        """
        Take a screenshot of a page.
        
        Args:
            task: The task containing the screenshot parameters.
            
        Returns:
            A dictionary containing the screenshot information.
        """
        page_id = task.parameters.get("page_id")
        if not page_id:
            raise ValueError("page_id parameter is required for take_screenshot task")
        
        output_path = task.parameters.get("output_path")
        if not output_path:
            output_path = str(self.screenshots_dir / f"{page_id}-{int(time.time())}.png")
        
        full_page = task.parameters.get("full_page", True)
        clip = task.parameters.get("clip")
        omit_background = task.parameters.get("omit_background", False)
        quality = task.parameters.get("quality")
        
        # Get the page
        try:
            page = await self.browser_pool.get_page(page_id)
        except ValueError as e:
            self.logger.error(f"Error getting page {page_id}: {str(e)}")
            raise
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Prepare screenshot options
            screenshot_options = {
                "path": output_path,
                "full_page": full_page,
                "omit_background": omit_background
            }
            
            # Add clip if provided
            if clip:
                screenshot_options["clip"] = clip
            
            # Add quality if provided (for JPEG)
            if quality and output_path.lower().endswith(".jpg") or output_path.lower().endswith(".jpeg"):
                screenshot_options["quality"] = quality
            
            # Take the screenshot
            self.logger.info(f"Taking screenshot: {output_path}")
            await page.screenshot(**screenshot_options)
            
            # Get page dimensions
            dimensions = await page.evaluate("""
                {
                    width: Math.max(
                        document.body.scrollWidth,
                        document.documentElement.scrollWidth,
                        document.body.offsetWidth,
                        document.documentElement.offsetWidth,
                        document.body.clientWidth,
                        document.documentElement.clientWidth
                    ),
                    height: Math.max(
                        document.body.scrollHeight,
                        document.documentElement.scrollHeight,
                        document.body.offsetHeight,
                        document.documentElement.offsetHeight,
                        document.body.clientHeight,
                        document.documentElement.clientHeight
                    )
                }
            """)
            
            return {
                "page_id": page_id,
                "screenshot_path": output_path,
                "full_page": full_page,
                "dimensions": dimensions,
                "clip": clip,
                "file_size": os.path.getsize(output_path)
            }
        
        except Exception as e:
            self.logger.error(f"Error taking screenshot of page {page_id}: {str(e)}")
            raise
    
    async def _execute_script(self, task: Task) -> Dict[str, Any]:
        """
        Execute JavaScript on a page.
        
        Args:
            task: The task containing the script to execute.
            
        Returns:
            A dictionary containing the result of the script execution.
        """
        page_id = task.parameters.get("page_id")
        if not page_id:
            raise ValueError("page_id parameter is required for execute_script task")
        
        script = task.parameters.get("script")
        if not script:
            raise ValueError("script parameter is required for execute_script task")
        
        args = task.parameters.get("args", [])
        timeout = task.parameters.get("timeout", 30)
        
        # Get the page
        try:
            page = await self.browser_pool.get_page(page_id)
        except ValueError as e:
            self.logger.error(f"Error getting page {page_id}: {str(e)}")
            raise
        
        try:
            # Execute the script
            self.logger.info(f"Executing script on page {page_id}")
            result = await page.evaluate(script, *args, timeout=timeout * 1000)
            
            # Convert result to JSON-serializable format
            if result is not None:
                try:
                    # Try to convert to JSON and back to ensure it's serializable
                    result = json.loads(json.dumps(result))
                except (TypeError, json.JSONDecodeError):
                    # If not serializable, convert to string
                    result = str(result)
            
            return {
                "page_id": page_id,
                "result": result,
                "script_length": len(script)
            }
        
        except Exception as e:
            self.logger.error(f"Error executing script on page {page_id}: {str(e)}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        await self.browser_pool.close_all()
