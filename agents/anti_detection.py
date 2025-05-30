"""
Anti-detection agent for the web scraping system.
"""
import asyncio
import logging
import time
import random
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
import httpx
from urllib.parse import urlparse
import platform
import uuid
import os
import secrets
from datetime import datetime, timedelta
from playwright.async_api import async_playwright
import fake_useragent
import requests
from bs4 import BeautifulSoup
import aiohttp

from agents.base import Agent
from models.task import Task, TaskType
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage


class BrowserFingerprint:
    """
    Class for generating and managing browser fingerprints.
    """
    def __init__(self):
        """Initialize the browser fingerprint generator."""
        self.user_agents = [
            # Windows Chrome
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            # Windows Firefox
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
            # Windows Edge
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
            # macOS Chrome
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            # macOS Safari
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            # Linux Chrome
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            # Mobile - iOS Safari
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            # Mobile - Android Chrome
            "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
        ]

        self.languages = [
            "en-US,en;q=0.9",
            "en-GB,en;q=0.9",
            "en-CA,en;q=0.9,fr-CA;q=0.8",
            "fr-FR,fr;q=0.9,en-US;q=0.8",
            "de-DE,de;q=0.9,en-US;q=0.8",
            "es-ES,es;q=0.9,en-US;q=0.8",
            "it-IT,it;q=0.9,en-US;q=0.8",
            "ja-JP,ja;q=0.9,en-US;q=0.8",
            "zh-CN,zh;q=0.9,en-US;q=0.8",
            "pt-BR,pt;q=0.9,en-US;q=0.8",
        ]

        self.platforms = [
            "Windows NT 10.0; Win64; x64",
            "Windows NT 10.0; WOW64",
            "Macintosh; Intel Mac OS X 10_15_7",
            "Macintosh; Intel Mac OS X 10_14_6",
            "X11; Linux x86_64",
            "X11; Ubuntu; Linux x86_64",
            "iPhone; CPU iPhone OS 14_6 like Mac OS X",
            "iPad; CPU OS 14_6 like Mac OS X",
            "Linux; Android 11; SM-G991B",
            "Linux; Android 10; SM-A505F",
        ]

        self.color_depths = [24, 32, 48]
        self.screen_resolutions = [
            "1920x1080",
            "1366x768",
            "1440x900",
            "1536x864",
            "2560x1440",
            "1280x720",
            "1600x900",
            "3840x2160",
            "1280x800",
            "1920x1200",
        ]

        self.timezones = [
            "UTC",
            "America/New_York",
            "America/Los_Angeles",
            "America/Chicago",
            "Europe/London",
            "Europe/Paris",
            "Europe/Berlin",
            "Asia/Tokyo",
            "Asia/Shanghai",
            "Australia/Sydney",
        ]

        self.logger = logging.getLogger("browser_fingerprint")

    def generate_fingerprint(self, consistent: bool = False, seed: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a browser fingerprint.

        Args:
            consistent: Whether to generate a consistent fingerprint based on the seed.
            seed: Seed for random number generation to create consistent fingerprints.

        Returns:
            A dictionary containing the browser fingerprint.
        """
        if consistent and seed:
            # Use the seed to initialize the random number generator
            random.seed(hashlib.md5(seed.encode()).hexdigest())

        user_agent = random.choice(self.user_agents)
        language = random.choice(self.languages)
        platform_str = random.choice(self.platforms)
        color_depth = random.choice(self.color_depths)
        screen_resolution = random.choice(self.screen_resolutions)
        timezone = random.choice(self.timezones)

        # Generate a canvas fingerprint hash (simulated)
        canvas_hash = hashlib.md5(f"{random.random()}".encode()).hexdigest() if not consistent else hashlib.md5(f"{seed}_canvas".encode()).hexdigest()

        # Generate a WebGL fingerprint hash (simulated)
        webgl_hash = hashlib.md5(f"{random.random()}".encode()).hexdigest() if not consistent else hashlib.md5(f"{seed}_webgl".encode()).hexdigest()

        # Generate a font list (simulated)
        font_count = random.randint(10, 30)

        # Reset the random seed if we used a custom one
        if consistent and seed:
            random.seed()

        return {
            "user_agent": user_agent,
            "language": language,
            "platform": platform_str,
            "color_depth": color_depth,
            "screen_resolution": screen_resolution,
            "timezone": timezone,
            "canvas_fingerprint": canvas_hash,
            "webgl_fingerprint": webgl_hash,
            "font_count": font_count,
            "do_not_track": random.choice([None, "1", "0"]),
            "cookie_enabled": random.choice([True, True, True, False]),  # Mostly enabled
            "touch_points": random.choice([0, 0, 0, 1, 2, 5]),  # Mostly non-touch
            "hardware_concurrency": random.choice([2, 4, 8, 12, 16]),
            "device_memory": random.choice([2, 4, 8, 16, 32]),
        }

    def get_headers_from_fingerprint(self, fingerprint: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate HTTP headers based on a browser fingerprint.

        Args:
            fingerprint: The browser fingerprint.

        Returns:
            A dictionary of HTTP headers.
        """
        headers = {
            "User-Agent": fingerprint["user_agent"],
            "Accept-Language": fingerprint["language"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }

        if fingerprint["do_not_track"] is not None:
            headers["DNT"] = fingerprint["do_not_track"]

        return headers


class RequestPatternManager:
    """
    Manager for varying request patterns to avoid detection.
    """
    def __init__(self):
        """Initialize the request pattern manager."""
        self.logger = logging.getLogger("request_pattern_manager")
        self.last_request_time = 0
        self.request_count = 0
        self.domain_stats = {}

    def calculate_delay(self, domain: str, aggressive: bool = False) -> float:
        """
        Calculate a delay before the next request to avoid detection.

        Args:
            domain: The domain being accessed.
            aggressive: Whether to use more aggressive (faster) scraping.

        Returns:
            The delay in seconds.
        """
        # Initialize domain stats if not present
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {
                "request_count": 0,
                "last_request_time": 0,
                "delays": [],
                "errors": 0,
                "success_rate": 1.0,
            }

        stats = self.domain_stats[domain]
        stats["request_count"] += 1

        # Base delay calculation
        if aggressive:
            base_delay = random.uniform(0.5, 1.5)
        else:
            base_delay = random.uniform(2.0, 5.0)

        # Adjust based on request count (gradual slowdown)
        count_factor = min(stats["request_count"] / 100, 1.0)
        delay = base_delay * (1 + count_factor)

        # Add randomness to appear more human-like
        humanized_delay = delay * random.uniform(0.8, 1.2)

        # Adjust based on success rate
        if stats["success_rate"] < 0.9:
            # If we're getting errors, slow down more
            humanized_delay *= (2.0 - stats["success_rate"])

        # Occasionally add a longer pause to simulate human behavior
        if random.random() < 0.05:  # 5% chance
            humanized_delay += random.uniform(5.0, 15.0)

        # Update stats
        stats["delays"].append(humanized_delay)
        if len(stats["delays"]) > 10:
            stats["delays"].pop(0)  # Keep only the last 10 delays

        self.last_request_time = time.time()
        stats["last_request_time"] = self.last_request_time

        return humanized_delay

    def update_success(self, domain: str, success: bool) -> None:
        """
        Update the success rate for a domain.

        Args:
            domain: The domain being accessed.
            success: Whether the request was successful.
        """
        if domain not in self.domain_stats:
            return

        stats = self.domain_stats[domain]

        # Update error count
        if not success:
            stats["errors"] += 1

        # Update success rate
        if stats["request_count"] > 0:
            stats["success_rate"] = (stats["request_count"] - stats["errors"]) / stats["request_count"]

    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """
        Get statistics for a domain.

        Args:
            domain: The domain to get statistics for.

        Returns:
            A dictionary of domain statistics.
        """
        return self.domain_stats.get(domain, {
            "request_count": 0,
            "last_request_time": 0,
            "delays": [],
            "errors": 0,
            "success_rate": 1.0,
        })


class BlockingDetector:
    """
    Detector for identifying when a website is blocking scraping.
    """
    def __init__(self):
        """Initialize the blocking detector."""
        self.logger = logging.getLogger("blocking_detector")
        self.blocking_indicators = [
            "captcha",
            "robot",
            "automated",
            "blocked",
            "suspicious",
            "unusual traffic",
            "too many requests",
            "access denied",
            "forbidden",
            "security check",
            "verify you are human",
        ]
        self.domain_blocks = {}

    def is_blocked(self, response_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a response indicates blocking.

        Args:
            response_data: The response data to check.

        Returns:
            A tuple of (is_blocked, reason).
        """
        status_code = response_data.get("status_code", 200)
        content = response_data.get("content", "")
        url = response_data.get("url", "")
        headers = response_data.get("headers", {})

        # Check status code
        if status_code in [403, 429, 503]:
            return True, f"Blocked status code: {status_code}"

        # Check for blocking indicators in content
        lower_content = content.lower() if isinstance(content, str) else ""
        for indicator in self.blocking_indicators:
            if indicator.lower() in lower_content:
                return True, f"Blocking indicator found: {indicator}"

        # Check for CAPTCHA presence
        if "captcha" in lower_content or "recaptcha" in lower_content:
            return True, "CAPTCHA detected"

        # Check for unusual redirects
        if response_data.get("redirect_count", 0) > 3:
            return True, "Too many redirects"

        # Check for unusual content length
        if len(content) < 500 and status_code == 200:
            # Suspiciously short content for a 200 response
            return True, "Suspiciously short content"

        # Not blocked
        return False, ""

    def record_blocking(self, domain: str, is_blocked: bool, reason: str = "") -> None:
        """
        Record blocking information for a domain.

        Args:
            domain: The domain being accessed.
            is_blocked: Whether the domain is blocking.
            reason: The reason for blocking.
        """
        if domain not in self.domain_blocks:
            self.domain_blocks[domain] = {
                "blocked_count": 0,
                "total_requests": 0,
                "last_blocked_time": 0,
                "reasons": [],
            }

        stats = self.domain_blocks[domain]
        stats["total_requests"] += 1

        if is_blocked:
            stats["blocked_count"] += 1
            stats["last_blocked_time"] = time.time()
            stats["reasons"].append(reason)

            # Keep only the last 10 reasons
            if len(stats["reasons"]) > 10:
                stats["reasons"].pop(0)

    def get_blocking_stats(self, domain: str) -> Dict[str, Any]:
        """
        Get blocking statistics for a domain.

        Args:
            domain: The domain to get statistics for.

        Returns:
            A dictionary of blocking statistics.
        """
        return self.domain_blocks.get(domain, {
            "blocked_count": 0,
            "total_requests": 0,
            "last_blocked_time": 0,
            "reasons": [],
        })

    def should_backoff(self, domain: str) -> Tuple[bool, float]:
        """
        Determine if we should back off from a domain due to blocking.

        Args:
            domain: The domain to check.

        Returns:
            A tuple of (should_backoff, backoff_time).
        """
        stats = self.get_blocking_stats(domain)

        if stats["total_requests"] == 0:
            return False, 0

        block_ratio = stats["blocked_count"] / stats["total_requests"]

        if block_ratio > 0.5 and stats["blocked_count"] >= 3:
            # More than 50% of requests are blocked and at least 3 blocks
            backoff_time = min(3600, 60 * (2 ** min(stats["blocked_count"] - 2, 6)))  # Exponential backoff, max 1 hour
            return True, backoff_time

        if stats["blocked_count"] >= 5:
            # At least 5 blocks, regardless of ratio
            backoff_time = min(7200, 120 * (2 ** min(stats["blocked_count"] - 4, 6)))  # Exponential backoff, max 2 hours
            return True, backoff_time

        return False, 0


class AntiDetectionAgent(Agent):
    """
    Agent responsible for handling anti-detection measures.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new anti-detection agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="anti_detection")
        self.coordinator_id = coordinator_id

        # Proxy management
        self.proxies: List[Dict[str, Any]] = []
        self.proxy_stats: Dict[str, Dict[str, Any]] = {}
        self.current_proxy_index = 0
        self.proxy_rotation_interval = 300  # 5 minutes
        self.proxy_health_check_interval = 60  # 1 minute
        self.proxy_failure_threshold = 3
        self.proxy_success_threshold = 5

        # User agent management
        self.user_agents: List[str] = []
        self.user_agent_stats: Dict[str, Dict[str, Any]] = {}
        self.current_user_agent_index = 0
        self.user_agent_rotation_interval = 1800  # 30 minutes

        # Browser fingerprint management
        self.fingerprints: List[Dict[str, Any]] = []
        self.fingerprint_stats: Dict[str, Dict[str, Any]] = {}
        self.current_fingerprint_index = 0
        self.fingerprint_rotation_interval = 3600  # 1 hour

        # Request pattern management
        self.request_patterns: List[Dict[str, Any]] = []
        self.current_pattern_index = 0
        self.pattern_rotation_interval = 1800  # 30 minutes

        # Register message handlers
        self.register_handler("get_proxy", self._handle_get_proxy)
        self.register_handler("get_user_agent", self._handle_get_user_agent)
        self.register_handler("get_fingerprint", self._handle_get_fingerprint)
        self.register_handler("update_proxy_stats", self._handle_update_proxy_stats)
        self.register_handler("update_fingerprint_stats", self._handle_update_fingerprint_stats)

        # Start periodic tasks
        self._start_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the anti-detection agent."""
        asyncio.create_task(self._periodic_proxy_rotation())
        asyncio.create_task(self._periodic_proxy_health_check())
        asyncio.create_task(self._periodic_fingerprint_rotation())
        asyncio.create_task(self._periodic_pattern_rotation())
        asyncio.create_task(self._periodic_stats_cleanup())

    async def _periodic_proxy_rotation(self) -> None:
        """Periodically rotate proxies based on performance."""
        while self.running:
            try:
                await asyncio.sleep(self.proxy_rotation_interval)
                if not self.running:
                    break

                await self._rotate_proxies()
            except Exception as e:
                self.logger.error(f"Error in proxy rotation: {str(e)}", exc_info=True)

    async def _periodic_proxy_health_check(self) -> None:
        """Periodically check proxy health."""
        while self.running:
            try:
                await asyncio.sleep(self.proxy_health_check_interval)
                if not self.running:
                    break

                await self._check_proxy_health()
            except Exception as e:
                self.logger.error(f"Error in proxy health check: {str(e)}", exc_info=True)

    async def _rotate_proxies(self) -> None:
        """Rotate proxies based on performance metrics."""
        if not self.proxies:
            return

        # Sort proxies by success rate and response time
        sorted_proxies = sorted(
            self.proxies,
            key=lambda p: (
                self.proxy_stats.get(p["id"], {}).get("success_rate", 0),
                -self.proxy_stats.get(p["id"], {}).get("avg_response_time", float("inf"))
            ),
            reverse=True
        )

        # Update proxy list
        self.proxies = sorted_proxies
        self.current_proxy_index = 0

    async def _check_proxy_health(self) -> None:
        """Check health of all proxies."""
        for proxy in self.proxies:
            try:
                async with aiohttp.ClientSession() as session:
                    start_time = time.time()
                    async with session.get(
                        "https://www.google.com",
                        proxy=f"http://{proxy['host']}:{proxy['port']}",
                        timeout=10
                    ) as response:
                        response_time = time.time() - start_time

                        if response.status == 200:
                            await self._update_proxy_stats(
                                proxy["id"],
                                success=True,
                                response_time=response_time
                            )
                        else:
                            await self._update_proxy_stats(
                                proxy["id"],
                                success=False,
                                response_time=response_time
                            )
            except Exception as e:
                self.logger.error(f"Error checking proxy {proxy['id']}: {str(e)}")
                await self._update_proxy_stats(proxy["id"], success=False)

    async def _update_proxy_stats(self, proxy_id: str, success: bool, response_time: Optional[float] = None) -> None:
        """Update proxy statistics."""
        if proxy_id not in self.proxy_stats:
            self.proxy_stats[proxy_id] = {
                "success_count": 0,
                "failure_count": 0,
                "total_response_time": 0,
                "request_count": 0
            }

        stats = self.proxy_stats[proxy_id]
        stats["request_count"] += 1

        if success:
            stats["success_count"] += 1
            if response_time:
                stats["total_response_time"] += response_time
        else:
            stats["failure_count"] += 1

        # Calculate metrics
        stats["success_rate"] = stats["success_count"] / stats["request_count"]
        if stats["success_count"] > 0:
            stats["avg_response_time"] = stats["total_response_time"] / stats["success_count"]

        # Check if proxy should be removed
        if (stats["failure_count"] >= self.proxy_failure_threshold and
            stats["success_count"] < self.proxy_success_threshold):
            self.proxies = [p for p in self.proxies if p["id"] != proxy_id]
            del self.proxy_stats[proxy_id]

    async def _periodic_fingerprint_rotation(self) -> None:
        """Periodically rotate browser fingerprints."""
        while self.running:
            try:
                await asyncio.sleep(self.fingerprint_rotation_interval)
                if not self.running:
                    break

                await self._rotate_fingerprints()
            except Exception as e:
                self.logger.error(f"Error in fingerprint rotation: {str(e)}", exc_info=True)

    async def _rotate_fingerprints(self) -> None:
        """Rotate fingerprints based on detection rate."""
        if not self.fingerprints:
            return

        # Sort fingerprints by detection rate
        sorted_fingerprints = sorted(
            self.fingerprints,
            key=lambda f: self.fingerprint_stats.get(f["id"], {}).get("detection_rate", 0)
        )

        # Update fingerprint list
        self.fingerprints = sorted_fingerprints
        self.current_fingerprint_index = 0

    async def _update_fingerprint_stats(self, fingerprint_id: str, detected: bool) -> None:
        """Update fingerprint statistics."""
        if fingerprint_id not in self.fingerprint_stats:
            self.fingerprint_stats[fingerprint_id] = {
                "detection_count": 0,
                "total_uses": 0
            }

        stats = self.fingerprint_stats[fingerprint_id]
        stats["total_uses"] += 1

        if detected:
            stats["detection_count"] += 1

        # Calculate detection rate
        stats["detection_rate"] = stats["detection_count"] / stats["total_uses"]

    async def _periodic_pattern_rotation(self) -> None:
        """Periodically rotate request patterns."""
        while self.running:
            try:
                await asyncio.sleep(self.pattern_rotation_interval)
                if not self.running:
                    break

                self.current_pattern_index = (self.current_pattern_index + 1) % len(self.request_patterns)
            except Exception as e:
                self.logger.error(f"Error in pattern rotation: {str(e)}", exc_info=True)

    async def _periodic_stats_cleanup(self) -> None:
        """Periodically clean up old statistics."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                if not self.running:
                    break

                # Clean up old proxy stats
                current_time = time.time()
                for proxy_id, stats in list(self.proxy_stats.items()):
                    if current_time - stats.get("last_updated", 0) > 86400:  # 24 hours
                        del self.proxy_stats[proxy_id]

                # Clean up old fingerprint stats
                for fingerprint_id, stats in list(self.fingerprint_stats.items()):
                    if current_time - stats.get("last_updated", 0) > 86400:  # 24 hours
                        del self.fingerprint_stats[fingerprint_id]

            except Exception as e:
                self.logger.error(f"Error in stats cleanup: {str(e)}", exc_info=True)

    async def _handle_get_proxy(self, message: Message) -> None:
        """Handle proxy request."""
        try:
            if not self.proxies:
                raise ValueError("No proxies available")

            proxy = self.proxies[self.current_proxy_index]
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"proxy": proxy}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error getting proxy: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="proxy_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_get_user_agent(self, message: Message) -> None:
        """
        Handle a user agent request.

        Args:
            message: The message containing user agent request parameters.
        """
        try:
            result = await self.get_user_agent(
                message.parameters.get("browser"),
                message.parameters.get("platform")
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error getting user agent: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="user_agent_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_get_fingerprint(self, message: Message) -> None:
        """Handle fingerprint request."""
        try:
            if not self.fingerprints:
                raise ValueError("No fingerprints available")

            fingerprint = self.fingerprints[self.current_fingerprint_index]
            self.current_fingerprint_index = (self.current_fingerprint_index + 1) % len(self.fingerprints)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"fingerprint": fingerprint}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error getting fingerprint: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="fingerprint_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_update_proxy_stats(self, message: Message) -> None:
        """Handle proxy statistics update."""
        try:
            proxy_id = message.data.get("proxy_id")
            success = message.data.get("success", False)
            response_time = message.data.get("response_time")

            if not proxy_id:
                raise ValueError("Missing proxy ID")

            await self._update_proxy_stats(proxy_id, success, response_time)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"status": "success"}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error updating proxy stats: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="proxy_stats_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_update_fingerprint_stats(self, message: Message) -> None:
        """Handle fingerprint statistics update."""
        try:
            fingerprint_id = message.data.get("fingerprint_id")
            detected = message.data.get("detected", False)

            if not fingerprint_id:
                raise ValueError("Missing fingerprint ID")

            await self._update_fingerprint_stats(fingerprint_id, detected)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"status": "success"}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error updating fingerprint stats: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="fingerprint_stats_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def get_user_agent(
        self,
        browser: Optional[str] = None,
        platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a user agent based on requirements.

        Args:
            browser: The browser to emulate.
            platform: The platform to emulate.

        Returns:
            A dictionary containing user agent information.
        """
        # Get user agent
        user_agent = random.choice(self.user_agents)

        # Update stats
        self.user_agent_stats[user_agent] = {
            "last_used": time.time(),
            "browser": browser,
            "platform": platform
        }

        return {
            "user_agent": user_agent,
            "browser": browser,
            "platform": platform
        }

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
        if task.type == TaskType.GENERATE_FINGERPRINT:
            domain = task.parameters.get("domain", "")
            consistent = task.parameters.get("consistent", False)

            # Generate a seed based on domain if consistent
            seed = hashlib.md5(domain.encode()).hexdigest() if consistent and domain else None

            # Create fingerprint generator
            fingerprint_generator = BrowserFingerprint()

            # Generate fingerprint
            fingerprint = fingerprint_generator.generate_fingerprint(consistent=consistent, seed=seed)

            # Generate headers
            headers = fingerprint_generator.get_headers_from_fingerprint(fingerprint)

            return {
                "fingerprint": fingerprint,
                "headers": headers,
                "domain": domain
            }
        elif task.type == TaskType.CHECK_BLOCKING:
            url = task.parameters.get("url")
            content = task.parameters.get("content", "")
            status_code = task.parameters.get("status_code", 200)
            headers = task.parameters.get("headers", {})

            if not url:
                raise ValueError("URL parameter is required for check_blocking task")

            # Create blocking detector
            detector = BlockingDetector()

            # Check if blocked
            is_blocked, reason = detector.is_blocked({
                "url": url,
                "content": content,
                "status_code": status_code,
                "headers": headers
            })

            # Get domain
            domain = urlparse(url).netloc

            # Record blocking
            detector.record_blocking(domain, is_blocked, reason)

            # Get blocking stats
            stats = detector.get_blocking_stats(domain)

            # Check if we should back off
            should_backoff, backoff_time = detector.should_backoff(domain)

            return {
                "is_blocked": is_blocked,
                "reason": reason,
                "domain": domain,
                "stats": stats,
                "should_backoff": should_backoff,
                "backoff_time": backoff_time
            }
        elif task.type == TaskType.GET_PROXY:
            domain = task.parameters.get("domain", "")
            country = task.parameters.get("country")

            # Get proxy
            proxy = await self._get_proxy(domain, country)

            return proxy
        else:
            raise ValueError(f"Unsupported task type for anti-detection agent: {task.type}")