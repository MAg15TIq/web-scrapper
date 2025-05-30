"""
URL Intelligence Agent for the self-aware web scraping system.

This agent analyzes URLs and websites to determine the best scraping approach.
"""
import asyncio
import logging
import time
import re
import json
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import uuid
from urllib.parse import urlparse, urljoin
import httpx
import tldextract
from bs4 import BeautifulSoup

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.intelligence import (
    ContentType, InputType, WebsiteType, AgentCapability,
    AgentProfile, ContentAnalysisResult
)


class URLIntelligenceAgent(Agent):
    """
    URL Intelligence Agent that analyzes URLs and websites to determine the best scraping approach.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new URL Intelligence Agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        super().__init__(agent_id=agent_id, agent_type="url_intelligence", coordinator_id=coordinator_id)
        
        # HTTP client for making requests
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        )
        
        # Technology detection patterns
        self.technology_patterns = self._initialize_technology_patterns()
        
        # Website type patterns
        self.website_patterns = self._initialize_website_patterns()
        
        # Cache for URL analysis results
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Register message handlers
        self.register_handler("analyze_url", self._handle_analyze_url)
        self.register_handler("detect_technologies", self._handle_detect_technologies)
        self.register_handler("check_robots_txt", self._handle_check_robots_txt)
        
        # Start periodic tasks
        self._start_periodic_tasks()
    
    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the URL Intelligence Agent."""
        asyncio.create_task(self._periodic_cache_cleanup())
    
    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up the analysis cache."""
        while self.running:
            self.logger.debug("Running periodic cache cleanup")
            
            # Remove cache entries older than 1 hour
            current_time = time.time()
            expired_keys = []
            
            for url, cache_entry in self.analysis_cache.items():
                if current_time - cache_entry.get("timestamp", 0) > 3600:
                    expired_keys.append(url)
            
            # Remove expired entries
            for key in expired_keys:
                del self.analysis_cache[key]
            
            self.logger.debug(f"Removed {len(expired_keys)} expired cache entries")
            
            # Sleep for 15 minutes
            await asyncio.sleep(900)
    
    def _initialize_technology_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for technology detection."""
        return {
            "wordpress": [
                "wp-content", "wp-includes", "wp-admin",
                "<meta name=\"generator\" content=\"WordPress"
            ],
            "drupal": [
                "Drupal.settings", "drupal.org", "/sites/all/",
                "<meta name=\"generator\" content=\"Drupal"
            ],
            "joomla": [
                "joomla", "/media/system/js/",
                "<meta name=\"generator\" content=\"Joomla"
            ],
            "magento": [
                "Mage.Cookies", "magento", "Magento_",
                "<script type=\"text/x-magento-init\">"
            ],
            "shopify": [
                "shopify", "Shopify.theme", "/cdn.shopify.com/"
            ],
            "woocommerce": [
                "woocommerce", "is-woocommerce", "wc-"
            ],
            "react": [
                "react.development.js", "react.production.min.js",
                "react-dom", "_reactRootContainer"
            ],
            "angular": [
                "ng-app", "ng-controller", "ng-model",
                "angular.js", "angular.min.js"
            ],
            "vue": [
                "vue.js", "vue.min.js", "v-bind", "v-model", "v-if", "v-for"
            ],
            "jquery": [
                "jquery.js", "jquery.min.js", "jQuery(", "$(document)"
            ],
            "bootstrap": [
                "bootstrap.css", "bootstrap.min.css", "bootstrap.js", "bootstrap.min.js"
            ],
            "cloudflare": [
                "cloudflare", "__cf", "cf-", "cf_chl_"
            ],
            "google_analytics": [
                "google-analytics.com", "ga('create'", "gtag("
            ],
            "google_tag_manager": [
                "googletagmanager.com", "gtm.js", "GTM-"
            ],
            "facebook_pixel": [
                "connect.facebook.net", "fbq(", "fbevents.js"
            ],
            "recaptcha": [
                "recaptcha", "g-recaptcha", "grecaptcha"
            ]
        }
    
    def _initialize_website_patterns(self) -> Dict[WebsiteType, List[str]]:
        """Initialize patterns for website type detection."""
        return {
            WebsiteType.ECOMMERCE: [
                "cart", "checkout", "product", "shop", "store", "price",
                "add to cart", "buy now", "shopping", "payment", "shipping"
            ],
            WebsiteType.NEWS: [
                "news", "article", "headline", "reporter", "editor",
                "breaking", "latest", "politics", "world", "sports"
            ],
            WebsiteType.BLOG: [
                "blog", "post", "author", "comment", "category",
                "tag", "archive", "read more", "published"
            ],
            WebsiteType.SOCIAL_MEDIA: [
                "profile", "friend", "follow", "like", "share",
                "comment", "post", "status", "timeline", "feed"
            ],
            WebsiteType.FORUM: [
                "forum", "thread", "post", "reply", "topic",
                "board", "discussion", "community", "member"
            ],
            WebsiteType.GOVERNMENT: [
                "government", "official", "agency", "department", "ministry",
                "public", "service", "citizen", "national", "federal"
            ],
            WebsiteType.ACADEMIC: [
                "university", "college", "academic", "research", "study",
                "course", "student", "faculty", "professor", "education"
            ],
            WebsiteType.CORPORATE: [
                "company", "corporate", "business", "enterprise", "industry",
                "service", "solution", "client", "partner", "investor"
            ]
        }
    
    async def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Analyze a URL to determine its characteristics.

        Args:
            url: The URL to analyze.

        Returns:
            A dictionary containing the analysis results.
        """
        self.logger.info(f"Analyzing URL: {url}")
        
        # Check cache first
        if url in self.analysis_cache:
            self.logger.info(f"Using cached analysis for URL: {url}")
            return self.analysis_cache[url]["result"]
        
        # Parse URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        
        # Extract domain information
        domain_info = tldextract.extract(url)
        
        # Fetch robots.txt
        robots_txt_info = await self._fetch_robots_txt(url)
        
        # Fetch URL content
        try:
            response = await self.client.get(url)
            content = response.text
            status_code = response.status_code
            headers = dict(response.headers)
            content_type = headers.get("content-type", "")
        except Exception as e:
            self.logger.error(f"Error fetching URL: {str(e)}")
            content = ""
            status_code = 0
            headers = {}
            content_type = ""
        
        # Detect technologies
        technologies = self._detect_technologies(content)
        
        # Determine website type
        website_type = self._determine_website_type(url, content)
        
        # Check for JavaScript requirements
        requires_javascript = self._requires_javascript(content, technologies)
        
        # Check for authentication
        requires_authentication = self._requires_authentication(url, content)
        
        # Check for anti-bot measures
        has_anti_bot = self._has_anti_bot(url, content, technologies)
        
        # Check for pagination
        has_pagination, pagination_info = self._detect_pagination(url, content)
        
        # Estimate complexity
        complexity = self._estimate_complexity(content, technologies)
        
        # Create result
        result = {
            "url": url,
            "domain": domain,
            "path": path,
            "domain_info": {
                "subdomain": domain_info.subdomain,
                "domain": domain_info.domain,
                "suffix": domain_info.suffix
            },
            "status_code": status_code,
            "content_type": content_type,
            "technologies": technologies,
            "website_type": website_type.value if website_type else None,
            "requires_javascript": requires_javascript,
            "requires_authentication": requires_authentication,
            "has_anti_bot": has_anti_bot,
            "has_pagination": has_pagination,
            "pagination_info": pagination_info,
            "complexity": complexity,
            "robots_txt": robots_txt_info,
            "headers": headers
        }
        
        # Cache the result
        self.analysis_cache[url] = {
            "result": result,
            "timestamp": time.time()
        }
        
        self.logger.info(f"URL analysis complete: {url}")
        return result
    
    async def _fetch_robots_txt(self, url: str) -> Dict[str, Any]:
        """
        Fetch and parse robots.txt for a URL.

        Args:
            url: The URL to analyze.

        Returns:
            A dictionary containing robots.txt information.
        """
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        try:
            response = await self.client.get(robots_url)
            if response.status_code == 200:
                content = response.text
                
                # Parse robots.txt
                disallowed_paths = []
                allowed_paths = []
                crawl_delay = None
                
                for line in content.split("\n"):
                    line = line.strip().lower()
                    
                    if line.startswith("disallow:"):
                        path = line[len("disallow:"):].strip()
                        if path:
                            disallowed_paths.append(path)
                    
                    elif line.startswith("allow:"):
                        path = line[len("allow:"):].strip()
                        if path:
                            allowed_paths.append(path)
                    
                    elif line.startswith("crawl-delay:"):
                        try:
                            delay = line[len("crawl-delay:"):].strip()
                            crawl_delay = float(delay)
                        except ValueError:
                            pass
                
                return {
                    "exists": True,
                    "url": robots_url,
                    "content": content,
                    "disallowed_paths": disallowed_paths,
                    "allowed_paths": allowed_paths,
                    "crawl_delay": crawl_delay,
                    "can_crawl": self._can_crawl_url(url, disallowed_paths, allowed_paths)
                }
            else:
                return {
                    "exists": False,
                    "url": robots_url,
                    "can_crawl": True
                }
        except Exception as e:
            self.logger.error(f"Error fetching robots.txt: {str(e)}")
            return {
                "exists": False,
                "url": robots_url,
                "error": str(e),
                "can_crawl": True
            }
    
    def _can_crawl_url(self, url: str, disallowed_paths: List[str], allowed_paths: List[str]) -> bool:
        """
        Determine if a URL can be crawled based on robots.txt rules.

        Args:
            url: The URL to check.
            disallowed_paths: List of disallowed paths from robots.txt.
            allowed_paths: List of allowed paths from robots.txt.

        Returns:
            True if the URL can be crawled, False otherwise.
        """
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Check if path is explicitly allowed
        for allowed in allowed_paths:
            if path.startswith(allowed):
                return True
        
        # Check if path is disallowed
        for disallowed in disallowed_paths:
            if path.startswith(disallowed):
                return False
        
        # Default to allowed
        return True
    
    def _detect_technologies(self, content: str) -> Dict[str, float]:
        """
        Detect technologies used on a website.

        Args:
            content: The HTML content of the website.

        Returns:
            A dictionary mapping technology names to confidence scores.
        """
        technologies = {}
        
        for tech_name, patterns in self.technology_patterns.items():
            confidence = 0.0
            matches = 0
            
            for pattern in patterns:
                if pattern in content:
                    matches += 1
            
            if matches > 0:
                # Calculate confidence based on number of matches
                confidence = min(1.0, matches / len(patterns))
                technologies[tech_name] = confidence
        
        return technologies
    
    def _determine_website_type(self, url: str, content: str) -> Optional[WebsiteType]:
        """
        Determine the type of website.

        Args:
            url: The URL of the website.
            content: The HTML content of the website.

        Returns:
            The website type, or None if not determinable.
        """
        # Extract domain and path
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()
        
        # Combine domain, path, and content for analysis
        text_to_analyze = f"{domain} {path} {content[:10000]}"
        
        # Count matches for each website type
        type_scores = {}
        
        for website_type, patterns in self.website_patterns.items():
            matches = 0
            for pattern in patterns:
                if pattern.lower() in text_to_analyze.lower():
                    matches += 1
            
            if matches > 0:
                score = matches / len(patterns)
                type_scores[website_type] = score
        
        # Return the type with the highest score
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return WebsiteType.UNKNOWN
    
    def _requires_javascript(self, content: str, technologies: Dict[str, float]) -> bool:
        """
        Determine if a website requires JavaScript.

        Args:
            content: The HTML content of the website.
            technologies: Detected technologies.

        Returns:
            True if JavaScript is required, False otherwise.
        """
        # Check for JavaScript frameworks
        js_frameworks = ["react", "angular", "vue", "jquery"]
        for framework in js_frameworks:
            if framework in technologies:
                return True
        
        # Check for common JavaScript patterns
        js_patterns = [
            "document.getElementById", "document.querySelector",
            "window.onload", "$(document).ready", "addEventListener",
            "fetch(", "axios.", ".ajax", "XMLHttpRequest",
            "data-bind", "ng-", "v-", "x-data", "x-bind"
        ]
        
        for pattern in js_patterns:
            if pattern in content:
                return True
        
        return False
    
    def _requires_authentication(self, url: str, content: str) -> bool:
        """
        Determine if a website requires authentication.

        Args:
            url: The URL of the website.
            content: The HTML content of the website.

        Returns:
            True if authentication is required, False otherwise.
        """
        # Check URL for authentication indicators
        auth_url_indicators = [
            "/login", "/signin", "/account", "/profile", "/dashboard",
            "/admin", "/member", "/secure", "/auth", "/private"
        ]
        
        for indicator in auth_url_indicators:
            if indicator in url.lower():
                return True
        
        # Check content for authentication indicators
        auth_content_indicators = [
            "login", "sign in", "username", "password", "forgot password",
            "register", "sign up", "create account", "log in", "logout"
        ]
        
        soup = BeautifulSoup(content, "html.parser")
        
        # Check for login forms
        forms = soup.find_all("form")
        for form in forms:
            form_text = form.get_text().lower()
            for indicator in auth_content_indicators:
                if indicator in form_text:
                    return True
            
            # Check for password fields
            if form.find("input", {"type": "password"}):
                return True
        
        return False
    
    def _has_anti_bot(self, url: str, content: str, technologies: Dict[str, float]) -> bool:
        """
        Determine if a website has anti-bot measures.

        Args:
            url: The URL of the website.
            content: The HTML content of the website.
            technologies: Detected technologies.

        Returns:
            True if anti-bot measures are detected, False otherwise.
        """
        # Check for anti-bot technologies
        anti_bot_techs = ["cloudflare", "recaptcha"]
        for tech in anti_bot_techs:
            if tech in technologies:
                return True
        
        # Check for common anti-bot patterns
        anti_bot_patterns = [
            "captcha", "robot", "human", "automated", "bot detection",
            "security check", "challenge", "verify", "cf-", "cf_chl_"
        ]
        
        for pattern in anti_bot_patterns:
            if pattern in content.lower():
                return True
        
        return False
    
    def _detect_pagination(self, url: str, content: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect pagination on a website.

        Args:
            url: The URL of the website.
            content: The HTML content of the website.

        Returns:
            A tuple containing (has_pagination, pagination_info).
        """
        pagination_info = {}
        
        # Check for common pagination indicators
        pagination_indicators = [
            "page=", "p=", "pg=", "?page", "&page",
            "pagination", "pager", "next", "prev", "previous",
            "load more", "show more", "infinite scroll"
        ]
        
        has_pagination = False
        for indicator in pagination_indicators:
            if indicator in content.lower() or indicator in url.lower():
                has_pagination = True
                break
        
        if not has_pagination:
            return False, pagination_info
        
        # Try to find pagination elements
        soup = BeautifulSoup(content, "html.parser")
        
        # Look for pagination containers
        pagination_elements = soup.select(".pagination, .pager, nav ul.pages, .page-numbers")
        if pagination_elements:
            pagination_info["elements_found"] = True
            
            # Extract page links
            page_links = []
            for element in pagination_elements:
                links = element.find_all("a")
                for link in links:
                    href = link.get("href")
                    if href:
                        page_links.append(urljoin(url, href))
            
            pagination_info["page_links"] = page_links
            
            # Try to find next/prev links
            next_link = soup.select_one("a.next, a[rel=next], a:contains('Next')")
            if next_link and next_link.get("href"):
                pagination_info["next_url"] = urljoin(url, next_link["href"])
            
            prev_link = soup.select_one("a.prev, a[rel=prev], a:contains('Previous')")
            if prev_link and prev_link.get("href"):
                pagination_info["prev_url"] = urljoin(url, prev_link["href"])
        
        # Look for page numbers in URL
        page_numbers = re.findall(r'page=(\d+)', url)
        if page_numbers:
            try:
                current_page = int(page_numbers[0])
                pagination_info["current_page"] = current_page
                
                # Try to find total pages
                page_of_pattern = re.search(r'[Pp]age\s+\d+\s+of\s+(\d+)', content)
                if page_of_pattern:
                    try:
                        total_pages = int(page_of_pattern.group(1))
                        pagination_info["total_pages"] = total_pages
                    except ValueError:
                        pass
            except ValueError:
                pass
        
        return True, pagination_info
    
    def _estimate_complexity(self, content: str, technologies: Dict[str, float]) -> float:
        """
        Estimate the complexity of a website.

        Args:
            content: The HTML content of the website.
            technologies: Detected technologies.

        Returns:
            A complexity score between 0.0 and 1.0.
        """
        complexity = 0.0
        
        # Base complexity from content length
        content_length = len(content)
        complexity += min(0.3, content_length / 100000)
        
        # Complexity from HTML structure
        soup = BeautifulSoup(content, "html.parser")
        
        # Count elements
        elements = soup.find_all()
        element_count = len(elements)
        complexity += min(0.2, element_count / 1000)
        
        # Count forms
        forms = soup.find_all("form")
        form_count = len(forms)
        complexity += min(0.1, form_count * 0.02)
        
        # Count scripts
        scripts = soup.find_all("script")
        script_count = len(scripts)
        complexity += min(0.2, script_count / 50)
        
        # Complexity from technologies
        tech_complexity = {
            "react": 0.1,
            "angular": 0.1,
            "vue": 0.1,
            "jquery": 0.05,
            "cloudflare": 0.05,
            "recaptcha": 0.05
        }
        
        for tech, value in tech_complexity.items():
            if tech in technologies:
                complexity += value
        
        # Cap at 1.0
        return min(1.0, complexity)
    
    async def _handle_analyze_url(self, message: Message) -> None:
        """
        Handle an analyze URL message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "url"):
            self.logger.warning("Received analyze_url message without url")
            return
        
        try:
            # Analyze the URL
            result = await self.analyze_url(message.url)
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error analyzing URL: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def _handle_detect_technologies(self, message: Message) -> None:
        """
        Handle a detect technologies message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "content"):
            self.logger.warning("Received detect_technologies message without content")
            return
        
        try:
            # Detect technologies
            technologies = self._detect_technologies(message.content)
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"technologies": technologies}
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error detecting technologies: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def _handle_check_robots_txt(self, message: Message) -> None:
        """
        Handle a check robots.txt message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "url"):
            self.logger.warning("Received check_robots_txt message without url")
            return
        
        try:
            # Fetch robots.txt
            result = await self._fetch_robots_txt(message.url)
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error checking robots.txt: {str(e)}")
            
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
        
        if task.type == TaskType.ANALYZE_URL:
            # Analyze URL
            url = task.parameters.get("url")
            if not url:
                raise ValueError("Missing url parameter")
            
            return await self.analyze_url(url)
        
        elif task.type == TaskType.DETECT_TECHNOLOGIES:
            # Detect technologies
            content = task.parameters.get("content")
            if not content:
                raise ValueError("Missing content parameter")
            
            technologies = self._detect_technologies(content)
            return {"technologies": technologies}
        
        elif task.type == TaskType.CHECK_ROBOTS_TXT:
            # Check robots.txt
            url = task.parameters.get("url")
            if not url:
                raise ValueError("Missing url parameter")
            
            return await self._fetch_robots_txt(url)
        
        else:
            raise ValueError(f"Unsupported task type: {task.type}")
