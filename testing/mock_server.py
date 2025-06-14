"""
Mock Server for Web Scraper Testing
Provides controllable HTTP responses for testing scenarios.
"""

import asyncio
import json
import logging
import random
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from aiohttp import web, ClientSession
import aiohttp
from urllib.parse import urlparse, parse_qs


@dataclass
class MockResponse:
    """Mock HTTP response definition."""
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""
    delay_seconds: float = 0
    content_type: str = "text/html"
    encoding: str = "utf-8"


@dataclass
class MockEndpoint:
    """Mock endpoint configuration."""
    path: str
    method: str = "GET"
    response: MockResponse = field(default_factory=MockResponse)
    response_function: Optional[Callable] = None
    hit_count: int = 0
    enabled: bool = True


class MockServer:
    """
    Mock HTTP server for testing web scraping scenarios.
    
    Provides controllable responses, delays, errors, and dynamic content
    for comprehensive testing of scraping configurations.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        """
        Initialize mock server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger("mock_server")
        self.app = web.Application()
        self.endpoints: Dict[str, MockEndpoint] = {}
        self.global_delay = 0.0
        self.error_rate = 0.0  # Percentage of requests that should fail
        self.running = False
        self.server = None
        
        # Setup default routes
        self._setup_default_routes()
    
    def _setup_default_routes(self):
        """Setup default mock endpoints."""
        # Catch-all route
        self.app.router.add_route("*", "/{path:.*}", self._handle_request)
    
    async def _handle_request(self, request: web.Request) -> web.Response:
        """Handle incoming HTTP requests."""
        path = request.path
        method = request.method.upper()
        
        # Find matching endpoint
        endpoint_key = f"{method}:{path}"
        endpoint = self.endpoints.get(endpoint_key)
        
        if not endpoint or not endpoint.enabled:
            # Return 404 for unregistered endpoints
            return web.Response(
                status=404,
                text=f"Mock endpoint not found: {method} {path}",
                content_type="text/plain"
            )
        
        # Increment hit count
        endpoint.hit_count += 1
        
        # Apply global delay
        if self.global_delay > 0:
            await asyncio.sleep(self.global_delay)
        
        # Apply endpoint-specific delay
        if endpoint.response.delay_seconds > 0:
            await asyncio.sleep(endpoint.response.delay_seconds)
        
        # Simulate random errors
        if self.error_rate > 0 and random.random() < (self.error_rate / 100):
            return web.Response(
                status=500,
                text="Simulated server error",
                content_type="text/plain"
            )
        
        # Generate response
        if endpoint.response_function:
            # Dynamic response
            try:
                response_data = endpoint.response_function(request)
                if isinstance(response_data, MockResponse):
                    mock_response = response_data
                else:
                    mock_response = MockResponse(body=str(response_data))
            except Exception as e:
                self.logger.error(f"Error in response function: {e}")
                return web.Response(
                    status=500,
                    text=f"Response function error: {str(e)}",
                    content_type="text/plain"
                )
        else:
            # Static response
            mock_response = endpoint.response
        
        # Build response headers
        headers = mock_response.headers.copy()
        headers["Content-Type"] = mock_response.content_type
        
        return web.Response(
            status=mock_response.status_code,
            text=mock_response.body,
            headers=headers
        )
    
    def add_endpoint(self, endpoint: MockEndpoint) -> None:
        """Add a mock endpoint."""
        endpoint_key = f"{endpoint.method.upper()}:{endpoint.path}"
        self.endpoints[endpoint_key] = endpoint
        self.logger.info(f"Added mock endpoint: {endpoint_key}")
    
    def remove_endpoint(self, path: str, method: str = "GET") -> bool:
        """Remove a mock endpoint."""
        endpoint_key = f"{method.upper()}:{path}"
        if endpoint_key in self.endpoints:
            del self.endpoints[endpoint_key]
            self.logger.info(f"Removed mock endpoint: {endpoint_key}")
            return True
        return False
    
    def get_endpoint_stats(self) -> Dict[str, Any]:
        """Get statistics for all endpoints."""
        stats = {}
        for key, endpoint in self.endpoints.items():
            stats[key] = {
                "hit_count": endpoint.hit_count,
                "enabled": endpoint.enabled,
                "path": endpoint.path,
                "method": endpoint.method
            }
        return stats
    
    def reset_stats(self) -> None:
        """Reset hit counts for all endpoints."""
        for endpoint in self.endpoints.values():
            endpoint.hit_count = 0
        self.logger.info("Reset endpoint statistics")
    
    def set_global_delay(self, delay_seconds: float) -> None:
        """Set global delay for all responses."""
        self.global_delay = delay_seconds
        self.logger.info(f"Set global delay: {delay_seconds}s")
    
    def set_error_rate(self, error_rate_percent: float) -> None:
        """Set global error rate percentage."""
        self.error_rate = max(0, min(100, error_rate_percent))
        self.logger.info(f"Set error rate: {self.error_rate}%")
    
    async def start(self) -> None:
        """Start the mock server."""
        if self.running:
            self.logger.warning("Mock server is already running")
            return
        
        try:
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            
            self.running = True
            self.server = runner
            
            self.logger.info(f"Mock server started on http://{self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start mock server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the mock server."""
        if not self.running or not self.server:
            self.logger.warning("Mock server is not running")
            return
        
        try:
            await self.server.cleanup()
            self.running = False
            self.server = None
            
            self.logger.info("Mock server stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop mock server: {e}")
            raise
    
    def create_test_pages(self) -> None:
        """Create common test pages for scraping tests."""
        # Simple HTML page
        simple_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <h1>Welcome to Test Page</h1>
            <p class="description">This is a test page for web scraping.</p>
            <div class="content">
                <ul>
                    <li class="item">Item 1</li>
                    <li class="item">Item 2</li>
                    <li class="item">Item 3</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        self.add_endpoint(MockEndpoint(
            path="/test-page",
            response=MockResponse(body=simple_html, content_type="text/html")
        ))
        
        # E-commerce product page
        product_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Product - Amazing Widget</title>
        </head>
        <body>
            <div class="product">
                <h1 class="product-title">Amazing Widget</h1>
                <div class="price">$29.99</div>
                <div class="description">
                    This is an amazing widget that does amazing things.
                </div>
                <div class="stock">In Stock</div>
                <div class="rating">4.5 stars</div>
            </div>
        </body>
        </html>
        """
        
        self.add_endpoint(MockEndpoint(
            path="/product/123",
            response=MockResponse(body=product_html, content_type="text/html")
        ))
        
        # News article page
        article_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Breaking News Article</title>
        </head>
        <body>
            <article>
                <h1 class="headline">Breaking: Important News Happened</h1>
                <div class="meta">
                    <span class="author">John Reporter</span>
                    <span class="date">2024-01-15</span>
                </div>
                <div class="content">
                    <p>This is the first paragraph of the news article.</p>
                    <p>This is the second paragraph with more details.</p>
                </div>
            </article>
        </body>
        </html>
        """
        
        self.add_endpoint(MockEndpoint(
            path="/news/article-1",
            response=MockResponse(body=article_html, content_type="text/html")
        ))
        
        # JSON API endpoint
        def json_response(request):
            return MockResponse(
                body=json.dumps({
                    "status": "success",
                    "data": {
                        "items": [
                            {"id": 1, "name": "Item 1", "value": 100},
                            {"id": 2, "name": "Item 2", "value": 200},
                            {"id": 3, "name": "Item 3", "value": 300}
                        ]
                    },
                    "timestamp": datetime.now().isoformat()
                }),
                content_type="application/json"
            )
        
        self.add_endpoint(MockEndpoint(
            path="/api/data",
            response_function=json_response
        ))
        
        # Dynamic content page
        def dynamic_content(request):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            random_number = random.randint(1, 1000)
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dynamic Content</title>
            </head>
            <body>
                <h1>Dynamic Content Page</h1>
                <div class="timestamp">{current_time}</div>
                <div class="random-number">{random_number}</div>
                <div class="counter">{request.path.count('/')}</div>
            </body>
            </html>
            """
            
            return MockResponse(body=html, content_type="text/html")
        
        self.add_endpoint(MockEndpoint(
            path="/dynamic",
            response_function=dynamic_content
        ))
        
        # Slow loading page
        self.add_endpoint(MockEndpoint(
            path="/slow",
            response=MockResponse(
                body="<html><body><h1>Slow Page</h1></body></html>",
                content_type="text/html",
                delay_seconds=3.0
            )
        ))
        
        # Error page
        self.add_endpoint(MockEndpoint(
            path="/error",
            response=MockResponse(
                status_code=500,
                body="<html><body><h1>Server Error</h1></body></html>",
                content_type="text/html"
            )
        ))
        
        self.logger.info("Created test pages")
    
    def get_url(self, path: str = "") -> str:
        """Get full URL for a path."""
        return f"http://{self.host}:{self.port}{path}"


# Convenience functions for common testing scenarios
async def create_test_server(port: int = 8888) -> MockServer:
    """Create and start a test server with common pages."""
    server = MockServer(port=port)
    server.create_test_pages()
    await server.start()
    return server


def create_quotes_page() -> str:
    """Create a quotes page similar to quotes.toscrape.com."""
    quotes = [
        {
            "text": "The world as we have created it is a process of our thinking.",
            "author": "Albert Einstein",
            "tags": ["change", "deep-thoughts", "thinking"]
        },
        {
            "text": "It is our choices that show what we truly are.",
            "author": "J.K. Rowling",
            "tags": ["abilities", "choices"]
        },
        {
            "text": "There are only two ways to live your life.",
            "author": "Albert Einstein",
            "tags": ["inspirational", "life", "live"]
        }
    ]
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quotes to Scrape</title>
        <style>
            .quote { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
            .text { font-style: italic; margin-bottom: 10px; }
            .author { font-weight: bold; }
            .tags { margin-top: 10px; }
            .tag { background: #eee; padding: 2px 5px; margin-right: 5px; }
        </style>
    </head>
    <body>
        <h1>Quotes to Scrape</h1>
    """
    
    for quote in quotes:
        tags_html = "".join(f'<span class="tag">{tag}</span>' for tag in quote["tags"])
        html += f"""
        <div class="quote">
            <div class="text">"{quote["text"]}"</div>
            <div class="author">by {quote["author"]}</div>
            <div class="tags">{tags_html}</div>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    return html
