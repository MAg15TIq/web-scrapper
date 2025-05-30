"""
Advanced pagination handling engine for the web scraping system.
"""
import re
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from bs4 import BeautifulSoup

class PaginationStrategy:
    """Base class for pagination strategies."""
    
    def __init__(self, name: str):
        """
        Initialize a pagination strategy.
        
        Args:
            name: Name of the strategy.
        """
        self.name = name
        self.logger = logging.getLogger(f"pagination.{name}")
    
    async def detect(self, url: str, html: str) -> Optional[Dict[str, Any]]:
        """
        Detect if this strategy can handle pagination for the given URL and HTML.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            Dictionary with pagination information if detected, None otherwise.
        """
        raise NotImplementedError("Subclasses must implement detect()")
    
    async def get_next_url(self, url: str, html: str) -> Optional[str]:
        """
        Get the URL of the next page.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            URL of the next page, or None if there is no next page.
        """
        raise NotImplementedError("Subclasses must implement get_next_url()")


class URLPatternStrategy(PaginationStrategy):
    """Strategy for detecting pagination based on URL patterns."""
    
    def __init__(self):
        """Initialize the URL pattern strategy."""
        super().__init__("url_pattern")
        
        # Common URL parameter patterns for pagination
        self.param_patterns = [
            "page", "p", "pg", "pgnum", "pagenum", "pagina", "pageid", "offset", "start"
        ]
    
    async def detect(self, url: str, html: str) -> Optional[Dict[str, Any]]:
        """
        Detect if the URL contains pagination parameters.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            Dictionary with pagination information if detected, None otherwise.
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Check if any pagination parameters are present
        for param in self.param_patterns:
            if param in query_params:
                try:
                    current_page = int(query_params[param][0])
                    
                    self.logger.info(f"Detected URL pagination pattern: {param}={current_page}")
                    
                    return {
                        "type": "url_pattern",
                        "param": param,
                        "current_page": current_page,
                        "base_url": url.split(f"{param}=")[0].rstrip("&?"),
                        "confidence": 0.9
                    }
                except (ValueError, IndexError):
                    continue
        
        # Check for path-based pagination (e.g., /page/2/)
        path_match = re.search(r"/(page|p)/(\d+)/?$", parsed_url.path)
        if path_match:
            param = path_match.group(1)
            current_page = int(path_match.group(2))
            
            self.logger.info(f"Detected path-based pagination: /{param}/{current_page}/")
            
            # Construct base URL by removing the pagination part
            path_parts = parsed_url.path.split(f"/{param}/{current_page}")
            base_path = path_parts[0]
            
            base_url = urlunparse((
                parsed_url.scheme,
                parsed_url.netloc,
                base_path,
                parsed_url.params,
                parsed_url.query,
                parsed_url.fragment
            ))
            
            return {
                "type": "path_pattern",
                "param": param,
                "current_page": current_page,
                "base_url": base_url,
                "confidence": 0.8
            }
        
        return None
    
    async def get_next_url(self, url: str, html: str) -> Optional[str]:
        """
        Get the URL of the next page based on URL pattern.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            URL of the next page, or None if there is no next page.
        """
        # Detect pagination pattern
        pagination_info = await self.detect(url, html)
        if not pagination_info:
            return None
        
        # Construct next page URL
        next_page = pagination_info["current_page"] + 1
        
        if pagination_info["type"] == "url_pattern":
            # URL parameter-based pagination
            base_url = pagination_info["base_url"]
            param = pagination_info["param"]
            
            # Check if base_url already has query parameters
            if "?" in base_url:
                next_url = f"{base_url}&{param}={next_page}"
            else:
                next_url = f"{base_url}?{param}={next_page}"
        
        elif pagination_info["type"] == "path_pattern":
            # Path-based pagination
            base_url = pagination_info["base_url"]
            param = pagination_info["param"]
            next_url = f"{base_url}/{param}/{next_page}/"
        
        else:
            return None
        
        self.logger.info(f"Generated next URL: {next_url}")
        return next_url


class DOMButtonStrategy(PaginationStrategy):
    """Strategy for detecting pagination based on DOM buttons."""
    
    def __init__(self):
        """Initialize the DOM button strategy."""
        super().__init__("dom_button")
        
        # Common text patterns for next page buttons
        self.next_text_patterns = [
            r"next",
            r"next\s*page",
            r"next\s*»",
            r"»",
            r"›",
            r"forward",
            r"more",
            r"continue",
            r"siguiente",
            r"próximo",
            r"weiter",
            r"suivant"
        ]
        
        # Common CSS selectors for pagination elements
        self.pagination_selectors = [
            "a.next",
            "a.pagination-next",
            "a.next-page",
            "a[rel='next']",
            "a[aria-label*='next' i]",
            "a.page-link[aria-label*='next' i]",
            ".pagination a:last-child",
            ".pager-next a",
            ".pagination__next",
            "li.next a",
            "nav.pagination a:last-child"
        ]
    
    async def detect(self, url: str, html: str) -> Optional[Dict[str, Any]]:
        """
        Detect if the page contains next page buttons.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            Dictionary with pagination information if detected, None otherwise.
        """
        soup = BeautifulSoup(html, "lxml")
        
        # Try specific selectors first
        for selector in self.pagination_selectors:
            next_link = soup.select_one(selector)
            if next_link and next_link.get("href"):
                self.logger.info(f"Found next button with selector: {selector}")
                return {
                    "type": "dom_button",
                    "selector": selector,
                    "next_url": urljoin(url, next_link["href"]),
                    "confidence": 0.9
                }
        
        # Try text-based detection
        for pattern in self.next_text_patterns:
            # Look for links containing the pattern
            links = soup.find_all("a", href=True)
            for link in links:
                link_text = link.get_text().lower().strip()
                if re.search(pattern, link_text, re.IGNORECASE):
                    self.logger.info(f"Found next button with text pattern: {pattern}")
                    return {
                        "type": "dom_button",
                        "text_pattern": pattern,
                        "next_url": urljoin(url, link["href"]),
                        "confidence": 0.8
                    }
        
        # Look for pagination elements with numbered links
        pagination_elements = soup.select(".pagination, .pager, nav[aria-label*='pagination' i], ul.page-numbers")
        for pagination in pagination_elements:
            links = pagination.find_all("a", href=True)
            current_links = pagination.select("span.current, a.active, li.active a, a[aria-current='page']")
            
            if current_links:
                # Try to find the current page number
                current_link = current_links[0]
                try:
                    current_page = int(re.search(r"\d+", current_link.get_text()).group())
                    
                    # Look for a link with the next page number
                    next_page = current_page + 1
                    for link in links:
                        link_text = link.get_text().strip()
                        if link_text == str(next_page):
                            self.logger.info(f"Found numbered pagination link for page {next_page}")
                            return {
                                "type": "dom_button",
                                "current_page": current_page,
                                "next_page": next_page,
                                "next_url": urljoin(url, link["href"]),
                                "confidence": 0.7
                            }
                except (ValueError, AttributeError):
                    continue
        
        return None
    
    async def get_next_url(self, url: str, html: str) -> Optional[str]:
        """
        Get the URL of the next page based on DOM buttons.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            URL of the next page, or None if there is no next page.
        """
        # Detect pagination buttons
        pagination_info = await self.detect(url, html)
        if not pagination_info:
            return None
        
        return pagination_info["next_url"]


class JavaScriptEventStrategy(PaginationStrategy):
    """Strategy for detecting pagination based on JavaScript events."""
    
    def __init__(self):
        """Initialize the JavaScript event strategy."""
        super().__init__("javascript_event")
    
    async def detect(self, url: str, html: str) -> Optional[Dict[str, Any]]:
        """
        Detect if the page contains JavaScript-based pagination.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            Dictionary with pagination information if detected, None otherwise.
        """
        # Look for common JavaScript pagination patterns
        js_pagination_patterns = [
            r"loadPage\s*\(\s*(\d+)\s*\+\s*1\s*\)",
            r"pagination\.next\(\)",
            r"nextPage\(\)",
            r"gotoPage\s*\(\s*currentPage\s*\+\s*1\s*\)",
            r"data-page\s*=\s*['\"]\d+['\"]",
            r"data-next-page\s*=\s*['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in js_pagination_patterns:
            if re.search(pattern, html):
                self.logger.info(f"Detected JavaScript pagination pattern: {pattern}")
                return {
                    "type": "javascript_event",
                    "pattern": pattern,
                    "requires_browser": True,
                    "confidence": 0.6
                }
        
        # Look for infinite scroll indicators
        infinite_scroll_patterns = [
            r"infiniteScroll",
            r"infinite-scroll",
            r"loadMore",
            r"load-more",
            r"data-infinite-scroll",
            r"data-load-more"
        ]
        
        for pattern in infinite_scroll_patterns:
            if re.search(pattern, html):
                self.logger.info(f"Detected infinite scroll pattern: {pattern}")
                return {
                    "type": "javascript_event",
                    "pattern": pattern,
                    "infinite_scroll": True,
                    "requires_browser": True,
                    "confidence": 0.5
                }
        
        return None
    
    async def get_next_url(self, url: str, html: str) -> Optional[str]:
        """
        Get the URL of the next page based on JavaScript events.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            URL of the next page, or None if there is no next page.
        """
        # JavaScript-based pagination typically requires a browser
        # This method returns None to indicate that a browser is needed
        return None


class PaginationEngine:
    """
    Advanced pagination handling engine that combines multiple detection strategies.
    """
    def __init__(self):
        """Initialize the pagination engine."""
        self.logger = logging.getLogger("pagination_engine")
        
        # Initialize strategies
        self.strategies = [
            URLPatternStrategy(),
            DOMButtonStrategy(),
            JavaScriptEventStrategy()
        ]
    
    async def detect_pagination(self, url: str, html: str) -> Dict[str, Any]:
        """
        Detect pagination method for a given URL and HTML content.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            Dictionary with pagination information.
        """
        results = []
        
        # Try all strategies
        for strategy in self.strategies:
            result = await strategy.detect(url, html)
            if result:
                results.append(result)
        
        if not results:
            self.logger.info("No pagination detected")
            return {
                "has_pagination": False,
                "methods": []
            }
        
        # Sort results by confidence
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        self.logger.info(f"Detected {len(results)} pagination methods")
        return {
            "has_pagination": True,
            "methods": results,
            "best_method": results[0]
        }
    
    async def get_next_url(self, url: str, html: str) -> Optional[str]:
        """
        Get the URL of the next page.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            
        Returns:
            URL of the next page, or None if there is no next page.
        """
        # Detect pagination
        pagination_info = await self.detect_pagination(url, html)
        if not pagination_info["has_pagination"]:
            return None
        
        # Try strategies in order of confidence
        for method in pagination_info["methods"]:
            strategy_name = method["type"]
            for strategy in self.strategies:
                if strategy.name == strategy_name:
                    next_url = await strategy.get_next_url(url, html)
                    if next_url:
                        return next_url
        
        return None
    
    async def handle_pagination(self, url: str, html: str, browser_available: bool = False) -> Dict[str, Any]:
        """
        Handle pagination for a given URL and HTML content.
        
        Args:
            url: Current URL.
            html: HTML content of the current page.
            browser_available: Whether a browser is available for JavaScript rendering.
            
        Returns:
            Dictionary with pagination information.
        """
        # Detect pagination
        pagination_info = await self.detect_pagination(url, html)
        
        # If no pagination detected, return early
        if not pagination_info["has_pagination"]:
            return {
                "has_pagination": False,
                "next_url": None,
                "requires_browser": False
            }
        
        # Check if the best method requires a browser
        best_method = pagination_info["best_method"]
        requires_browser = best_method.get("requires_browser", False)
        
        # If a browser is required but not available, try other methods
        if requires_browser and not browser_available:
            self.logger.warning("Best pagination method requires a browser, but none is available")
            
            # Try to find a method that doesn't require a browser
            for method in pagination_info["methods"]:
                if not method.get("requires_browser", False):
                    best_method = method
                    requires_browser = False
                    break
        
        # Get the next URL
        next_url = None
        for method in pagination_info["methods"]:
            strategy_name = method["type"]
            for strategy in self.strategies:
                if strategy.name == strategy_name:
                    if method.get("requires_browser", False) and not browser_available:
                        continue
                    
                    next_url = await strategy.get_next_url(url, html)
                    if next_url:
                        break
            
            if next_url:
                break
        
        return {
            "has_pagination": True,
            "next_url": next_url,
            "requires_browser": requires_browser,
            "method": best_method["type"],
            "details": best_method
        }

# Create a singleton instance
pagination_engine = PaginationEngine()
