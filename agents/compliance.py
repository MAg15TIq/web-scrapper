"""
Compliance agent for the web scraping system.
"""
import logging
import asyncio
import time
import os
import json
import re
import urllib.robotparser
import httpx
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from urllib.parse import urlparse
from collections import defaultdict
import aiohttp

from agents.base import Agent
from models.task import Task, TaskStatus, TaskType
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority, ComplianceMessage


class ComplianceAgent(Agent):
    """
    Agent responsible for ensuring compliance with legal and ethical scraping guidelines,
    including robots.txt parsing, rate limiting, and terms of service monitoring.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new compliance agent.
        
        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="compliance")
        self.coordinator_id = coordinator_id
        
        # Directory for storing compliance data
        self.data_dir = "data/compliance"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True
        )
        
        # Cache for robots.txt data
        self.robots_cache: Dict[str, Dict[str, Any]] = {}
        self.robots_cache_ttl = 3600  # 1 hour
        
        # Cache for terms of service data
        self.tos_cache: Dict[str, Dict[str, Any]] = {}
        self.tos_cache_ttl = 86400  # 24 hours
        
        # Rate limit configuration by domain
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Default rate limit settings
        self.default_rate_limit = {
            "max_requests_per_minute": 10,
            "max_concurrent_requests": 2,
            "respect_retry_after": True,
            "adaptive": True,
            "backoff_factor": 2.0
        }
        
        # Request history
        self.request_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.max_history_size = 10000
        
        # Register additional message handlers
        self.register_handler("compliance_config", self._handle_compliance_config_message)
        self.register_handler("compliance_check", self._handle_compliance_check)
        self.register_handler("rate_limit_check", self._handle_rate_limit_check)
        
        # Start periodic tasks
        self._start_periodic_tasks()
    
    def _start_periodic_tasks(self) -> None:
        """Start periodic compliance tasks."""
        asyncio.create_task(self._periodic_cache_cleanup())
        asyncio.create_task(self._periodic_tos_check())
    
    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up expired cache entries."""
        while self.running:
            try:
                # Clean up cache every hour
                await asyncio.sleep(3600)
                if not self.running:
                    break
                
                self.logger.info("Running periodic cache cleanup")
                self._cleanup_expired_cache()
            except Exception as e:
                self.logger.error(f"Error in periodic cache cleanup: {str(e)}", exc_info=True)
    
    async def _periodic_tos_check(self) -> None:
        """Periodically check for terms of service changes."""
        while self.running:
            try:
                # Check ToS every 6 hours
                await asyncio.sleep(21600)
                if not self.running:
                    break
                
                self.logger.info("Running periodic ToS check")
                await self._check_tos_changes()
            except Exception as e:
                self.logger.error(f"Error in periodic ToS check: {str(e)}", exc_info=True)
    
    def _cleanup_expired_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        
        # Clean up robots.txt cache
        self.robots_cache = {
            domain: data
            for domain, data in self.robots_cache.items()
            if current_time - data["timestamp"] < self.robots_cache_ttl
        }
        
        # Clean up ToS cache
        self.tos_cache = {
            domain: data
            for domain, data in self.tos_cache.items()
            if current_time - data["timestamp"] < self.tos_cache_ttl
        }
    
    async def _check_tos_changes(self) -> None:
        """Check for changes in terms of service."""
        for domain, tos_data in self.tos_cache.items():
            try:
                # Fetch current ToS
                current_tos = await self._fetch_tos(domain)
                
                # Compare with cached version
                if current_tos != tos_data["content"]:
                    self.logger.warning(f"ToS change detected for {domain}")
                    
                    # Update cache
                    self.tos_cache[domain] = {
                        "content": current_tos,
                        "timestamp": time.time(),
                        "hash": hash(current_tos)
                    }
                    
                    # Notify coordinator
                    await self._notify_tos_change(domain)
            except Exception as e:
                self.logger.error(f"Error checking ToS for {domain}: {str(e)}", exc_info=True)
    
    async def _fetch_tos(self, domain: str) -> str:
        """
        Fetch terms of service for a domain.
        
        Args:
            domain: The domain to fetch ToS for.
            
        Returns:
            The terms of service content.
        """
        # This is a placeholder for actual ToS fetching
        # In a real implementation, you would fetch the actual ToS
        return ""
    
    async def _notify_tos_change(self, domain: str) -> None:
        """
        Notify the coordinator about ToS changes.
        
        Args:
            domain: The domain with changed ToS.
        """
        message = ComplianceMessage(
            sender_id=self.agent_id,
            recipient_id=self.coordinator_id,
            type="tos_change",
            domain=domain,
            details={
                "timestamp": time.time(),
                "message": f"Terms of service changed for {domain}"
            }
        )
        self.outbox.put(message)
    
    async def start(self) -> None:
        """Start the agent and initialize compliance tasks."""
        # Start the agent
        await super().start()
    
    async def stop(self) -> None:
        """Stop the agent and clean up resources."""
        # Close HTTP client
        await self.client.aclose()
        
        # Stop the agent
        await super().stop()
    
    async def _send_message(self, message: Message) -> None:
        """
        Send a message to another agent.
        
        Args:
            message: The message to send.
        """
        # In a real implementation, this would use a message broker
        self.logger.debug(f"Sending message to {message.recipient_id}: {message.type}")
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the result of the task.
        """
        self.logger.info(f"Executing task: {task.type}")
        
        if task.type == TaskType.PARSE_ROBOTS_TXT:
            return await self._execute_parse_robots_txt(task)
        elif task.type == TaskType.CHECK_RATE_LIMITS:
            return await self._execute_check_rate_limits(task)
        elif task.type == TaskType.MONITOR_TOS:
            return await self._execute_monitor_tos(task)
        elif task.type == TaskType.CHECK_LEGAL_COMPLIANCE:
            return await self._execute_check_legal_compliance(task)
        elif task.type == TaskType.ENFORCE_ETHICAL_SCRAPING:
            return await self._execute_enforce_ethical_scraping(task)
        else:
            raise ValueError(f"Unsupported task type: {task.type}")
    
    async def _execute_parse_robots_txt(self, task: Task) -> Dict[str, Any]:
        """
        Execute a robots.txt parsing task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the parsing results.
        """
        params = task.parameters
        
        # Get required parameters
        domain = params.get("domain")
        user_agent = params.get("user_agent", "MyScraperBot")
        cache_duration = params.get("cache_duration", 86400)  # 24 hours in seconds
        respect_crawl_delay = params.get("respect_crawl_delay", True)
        follow_sitemaps = params.get("follow_sitemaps", True)
        
        if not domain:
            raise ValueError("Missing required parameter: domain")
        
        # Check if we have a cached version that's still valid
        if domain in self.robots_cache:
            cache_entry = self.robots_cache[domain]
            cache_age = time.time() - cache_entry["timestamp"]
            
            if cache_age < cache_duration:
                self.logger.info(f"Using cached robots.txt for {domain}")
                return cache_entry["data"]
        
        # Fetch robots.txt
        robots_url = f"https://{domain}/robots.txt"
        try:
            response = await self.client.get(robots_url)
            response.raise_for_status()
            robots_content = response.text
            
            # Parse robots.txt
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(robots_content.splitlines())
            
            # Extract disallowed paths for the user agent
            disallowed_paths = []
            sitemaps = []
            crawl_delay = None
            
            # Extract information from robots.txt
            # This is a simplified implementation
            for line in robots_content.splitlines():
                line = line.strip().lower()
                
                # Check for sitemaps
                if line.startswith("sitemap:") and follow_sitemaps:
                    sitemap_url = line[8:].strip()
                    sitemaps.append(sitemap_url)
                
                # Check for crawl delay
                if respect_crawl_delay and line.startswith("crawl-delay:"):
                    try:
                        delay_str = line[12:].strip()
                        crawl_delay = float(delay_str)
                    except ValueError:
                        self.logger.warning(f"Invalid crawl delay in robots.txt: {line}")
            
            # Check if specific paths are allowed
            test_paths = [
                "/",
                "/products",
                "/categories",
                "/search",
                "/api"
            ]
            
            path_permissions = {}
            for path in test_paths:
                path_permissions[path] = rp.can_fetch(user_agent, f"https://{domain}{path}")
            
            # Store in cache
            result = {
                "domain": domain,
                "user_agent": user_agent,
                "robots_url": robots_url,
                "disallowed_paths": disallowed_paths,
                "sitemaps": sitemaps,
                "crawl_delay": crawl_delay,
                "path_permissions": path_permissions,
                "has_robots_txt": True
            }
            
            self.robots_cache[domain] = {
                "timestamp": time.time(),
                "data": result
            }
            
            # If crawl delay is specified, update rate limits for this domain
            if crawl_delay is not None:
                requests_per_minute = int(60 / crawl_delay) if crawl_delay > 0 else 60
                self.rate_limits[domain] = {
                    "max_requests_per_minute": requests_per_minute,
                    "max_concurrent_requests": 1,
                    "respect_retry_after": True,
                    "adaptive": False,
                    "backoff_factor": 2.0,
                    "source": "robots_txt"
                }
            
            return result
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # No robots.txt found
                result = {
                    "domain": domain,
                    "user_agent": user_agent,
                    "robots_url": robots_url,
                    "disallowed_paths": [],
                    "sitemaps": [],
                    "crawl_delay": None,
                    "path_permissions": {path: True for path in test_paths},
                    "has_robots_txt": False
                }
                
                self.robots_cache[domain] = {
                    "timestamp": time.time(),
                    "data": result
                }
                
                return result
            else:
                raise
    
    async def _execute_check_rate_limits(self, task: Task) -> Dict[str, Any]:
        """
        Execute a rate limit checking task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the rate limit results.
        """
        params = task.parameters
        
        # Get required parameters
        domain = params.get("domain")
        
        if not domain:
            raise ValueError("Missing required parameter: domain")
        
        # Get rate limit parameters
        max_requests_per_minute = params.get("max_requests_per_minute")
        max_concurrent_requests = params.get("max_concurrent_requests")
        respect_retry_after = params.get("respect_retry_after")
        adaptive = params.get("adaptive")
        backoff_factor = params.get("backoff_factor")
        
        # If domain already has rate limits, use those as defaults
        domain_rate_limits = self.rate_limits.get(domain, self.default_rate_limit)
        
        # Update with any provided parameters
        if max_requests_per_minute is not None:
            domain_rate_limits["max_requests_per_minute"] = max_requests_per_minute
        
        if max_concurrent_requests is not None:
            domain_rate_limits["max_concurrent_requests"] = max_concurrent_requests
        
        if respect_retry_after is not None:
            domain_rate_limits["respect_retry_after"] = respect_retry_after
        
        if adaptive is not None:
            domain_rate_limits["adaptive"] = adaptive
        
        if backoff_factor is not None:
            domain_rate_limits["backoff_factor"] = backoff_factor
        
        # Store updated rate limits
        self.rate_limits[domain] = domain_rate_limits
        
        # Calculate derived values
        min_interval = 60.0 / domain_rate_limits["max_requests_per_minute"]
        
        return {
            "domain": domain,
            "rate_limits": domain_rate_limits,
            "min_interval_seconds": min_interval,
            "source": domain_rate_limits.get("source", "manual")
        }
    
    async def _execute_monitor_tos(self, task: Task) -> Dict[str, Any]:
        """
        Execute a terms of service monitoring task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the monitoring results.
        """
        params = task.parameters
        
        # Get required parameters
        domain = params.get("domain")
        tos_url = params.get("tos_url")
        
        if not domain:
            raise ValueError("Missing required parameter: domain")
        
        if not tos_url:
            # Try to guess the TOS URL
            tos_url = f"https://{domain}/terms"
        
        # Get optional parameters
        check_frequency = params.get("check_frequency", "weekly")
        notify_on_change = params.get("notify_on_change", True)
        keywords_to_monitor = params.get("keywords_to_monitor", [
            "scraping",
            "crawling",
            "automated",
            "bot",
            "data collection"
        ])
        
        # Check if we need to fetch the TOS
        fetch_tos = True
        if domain in self.tos_cache:
            cache_entry = self.tos_cache[domain]
            last_check = datetime.fromtimestamp(cache_entry["timestamp"])
            
            # Determine if we need to check based on frequency
            if check_frequency == "daily":
                next_check = last_check + timedelta(days=1)
            elif check_frequency == "weekly":
                next_check = last_check + timedelta(weeks=1)
            elif check_frequency == "monthly":
                next_check = last_check + timedelta(days=30)
            else:
                next_check = last_check + timedelta(days=1)
            
            if datetime.now() < next_check:
                fetch_tos = False
        
        if fetch_tos:
            try:
                # Fetch the TOS page
                response = await self.client.get(tos_url)
                response.raise_for_status()
                tos_content = response.text
                
                # Extract text content (simplified)
                # In a real implementation, use a proper HTML parser
                tos_text = re.sub(r'<[^>]+>', ' ', tos_content)
                tos_text = re.sub(r'\s+', ' ', tos_text).strip()
                
                # Check for keywords
                keyword_matches = {}
                for keyword in keywords_to_monitor:
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    matches = pattern.findall(tos_text)
                    if matches:
                        keyword_matches[keyword] = len(matches)
                
                # Check for changes if we have a previous version
                changes_detected = False
                if domain in self.tos_cache:
                    old_tos = self.tos_cache[domain]["tos_text"]
                    if old_tos != tos_text:
                        changes_detected = True
                        
                        if notify_on_change:
                            # Log the change
                            self.logger.warning(f"Terms of Service changed for {domain}")
                            
                            # In a real implementation, send a notification
                
                # Store in cache
                self.tos_cache[domain] = {
                    "timestamp": time.time(),
                    "tos_url": tos_url,
                    "tos_text": tos_text,
                    "keyword_matches": keyword_matches
                }
                
                return {
                    "domain": domain,
                    "tos_url": tos_url,
                    "keywords_found": keyword_matches,
                    "changes_detected": changes_detected,
                    "check_frequency": check_frequency,
                    "last_checked": datetime.now().isoformat()
                }
                
            except httpx.HTTPStatusError as e:
                return {
                    "domain": domain,
                    "tos_url": tos_url,
                    "error": f"HTTP error: {e.response.status_code}",
                    "check_frequency": check_frequency,
                    "last_checked": datetime.now().isoformat()
                }
        else:
            # Return cached data
            cache_entry = self.tos_cache[domain]
            return {
                "domain": domain,
                "tos_url": cache_entry["tos_url"],
                "keywords_found": cache_entry["keyword_matches"],
                "changes_detected": False,
                "check_frequency": check_frequency,
                "last_checked": datetime.fromtimestamp(cache_entry["timestamp"]).isoformat(),
                "source": "cache"
            }
    
    async def _execute_check_legal_compliance(self, task: Task) -> Dict[str, Any]:
        """
        Execute a legal compliance checking task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the compliance check results.
        """
        params = task.parameters
        
        # Get required parameters
        domain = params.get("domain")
        
        if not domain:
            raise ValueError("Missing required parameter: domain")
        
        # Get optional parameters
        jurisdiction = params.get("jurisdiction", "US")
        data_types = params.get("data_types", ["public"])
        check_copyright = params.get("check_copyright", True)
        check_gdpr = params.get("check_gdpr", True)
        check_ccpa = params.get("check_ccpa", True)
        
        # Perform compliance checks
        compliance_issues = []
        
        # Check for personal data handling if applicable
        if "personal" in data_types:
            if check_gdpr and jurisdiction in ["EU", "UK"]:
                compliance_issues.append({
                    "type": "gdpr",
                    "message": "Scraping personal data requires explicit consent under GDPR",
                    "severity": "high"
                })
            
            if check_ccpa and jurisdiction == "US" and "CA" in jurisdiction:
                compliance_issues.append({
                    "type": "ccpa",
                    "message": "Scraping personal data may be subject to CCPA requirements",
                    "severity": "high"
                })
        
        # Check copyright considerations
        if check_copyright:
            compliance_issues.append({
                "type": "copyright",
                "message": "Ensure content is used in accordance with fair use/fair dealing principles",
                "severity": "medium"
            })
        
        # Generate compliance recommendations
        recommendations = []
        
        if "personal" in data_types:
            recommendations.append("Implement data anonymization")
            recommendations.append("Establish data retention policies")
            recommendations.append("Provide opt-out mechanisms")
        
        recommendations.append("Document compliance measures")
        recommendations.append("Respect robots.txt directives")
        recommendations.append("Implement rate limiting")
        
        return {
            "domain": domain,
            "jurisdiction": jurisdiction,
            "data_types": data_types,
            "compliance_issues": compliance_issues,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_enforce_ethical_scraping(self, task: Task) -> Dict[str, Any]:
        """
        Execute an ethical scraping enforcement task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the enforcement results.
        """
        params = task.parameters
        
        # Get required parameters
        domain = params.get("domain")
        
        if not domain:
            raise ValueError("Missing required parameter: domain")
        
        # Get optional parameters
        guidelines = params.get("guidelines", [
            "respect_robots_txt",
            "limit_request_rate",
            "identify_bot",
            "minimize_impact",
            "respect_copyright"
        ])
        identify_as = params.get("identify_as", "MyScraperBot (contact@example.com)")
        abort_on_violation = params.get("abort_on_violation", True)
        
        # Check each guideline
        guideline_results = {}
        violations = []
        
        # Check robots.txt compliance
        if "respect_robots_txt" in guidelines:
            # Create a robots.txt parsing task
            robots_task = Task(
                type=TaskType.PARSE_ROBOTS_TXT,
                parameters={
                    "domain": domain,
                    "user_agent": identify_as.split(" ")[0] if " " in identify_as else identify_as
                }
            )
            
            robots_result = await self._execute_parse_robots_txt(robots_task)
            guideline_results["respect_robots_txt"] = {
                "compliant": robots_result["has_robots_txt"],
                "details": robots_result
            }
            
            if not robots_result["has_robots_txt"]:
                self.logger.info(f"No robots.txt found for {domain}")
            else:
                # Check if any test paths are disallowed
                disallowed_paths = [path for path, allowed in robots_result["path_permissions"].items() if not allowed]
                if disallowed_paths:
                    violations.append({
                        "guideline": "respect_robots_txt",
                        "message": f"Some paths are disallowed by robots.txt: {', '.join(disallowed_paths)}",
                        "severity": "high"
                    })
        
        # Check rate limiting
        if "limit_request_rate" in guidelines:
            # Create a rate limit checking task
            rate_limit_task = Task(
                type=TaskType.CHECK_RATE_LIMITS,
                parameters={
                    "domain": domain
                }
            )
            
            rate_limit_result = await self._execute_check_rate_limits(rate_limit_task)
            guideline_results["limit_request_rate"] = {
                "compliant": True,  # Assume compliant since we're setting the limits
                "details": rate_limit_result
            }
        
        # Check bot identification
        if "identify_bot" in guidelines:
            guideline_results["identify_bot"] = {
                "compliant": bool(identify_as),
                "details": {
                    "user_agent": identify_as
                }
            }
            
            if not identify_as:
                violations.append({
                    "guideline": "identify_bot",
                    "message": "Bot is not properly identified with a user agent",
                    "severity": "medium"
                })
        
        # Check impact minimization
        if "minimize_impact" in guidelines:
            # This is more of a recommendation than something we can check
            guideline_results["minimize_impact"] = {
                "compliant": True,
                "details": {
                    "recommendations": [
                        "Use conditional GET requests with If-Modified-Since",
                        "Cache results when possible",
                        "Limit concurrent connections",
                        "Schedule scraping during off-peak hours"
                    ]
                }
            }
        
        # Check copyright respect
        if "respect_copyright" in guidelines:
            guideline_results["respect_copyright"] = {
                "compliant": True,  # Assume compliant, this is more of a policy check
                "details": {
                    "recommendations": [
                        "Only use data in accordance with fair use principles",
                        "Respect copyright notices on the website",
                        "Consider licensing requirements for commercial use"
                    ]
                }
            }
        
        # Determine overall compliance
        is_compliant = len(violations) == 0
        
        # Check if we should abort
        should_abort = abort_on_violation and not is_compliant
        
        return {
            "domain": domain,
            "is_compliant": is_compliant,
            "guideline_results": guideline_results,
            "violations": violations,
            "should_abort": should_abort,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_compliance_config_message(self, message: Message) -> None:
        """
        Handle a compliance configuration message.
        
        Args:
            message: The message to handle.
        """
        if not hasattr(message, "config") or not message.config:
            self.logger.warning("Received compliance config message without config")
            return
        
        self.logger.info("Updating compliance configuration")
        
        # Update rate limits if provided
        if "rate_limits" in message.config:
            for domain, limits in message.config["rate_limits"].items():
                self.rate_limits[domain] = limits
        
        # Update default rate limit if provided
        if "default_rate_limit" in message.config:
            self.default_rate_limit.update(message.config["default_rate_limit"])
        
        # Send acknowledgment
        status_message = StatusMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            status="compliance_config_updated",
            details={"default_rate_limit": self.default_rate_limit}
        )
        self.outbox.put(status_message)
    
    async def _handle_compliance_check(self, message: Message) -> None:
        """
        Handle a compliance check message.
        
        Args:
            message: The message to handle.
        """
        if not hasattr(message, "url"):
            self.logger.warning("Received compliance check without URL")
            return
        
        result = await self.check_compliance(
            message.url,
            message.request_type if hasattr(message, "request_type") else "GET",
            message.headers if hasattr(message, "headers") else None
        )
        
        # Send response
        response = ComplianceMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            type="compliance_result",
            url=message.url,
            details=result
        )
        self.outbox.put(response)
    
    async def _handle_rate_limit_check(self, message: Message) -> None:
        """
        Handle a rate limit check message.
        
        Args:
            message: The message to handle.
        """
        if not hasattr(message, "domain"):
            self.logger.warning("Received rate limit check without domain")
            return
        
        result = await self._check_rate_limits(
            message.domain,
            message.url if hasattr(message, "url") else f"https://{message.domain}/"
        )
        
        # Send response
        response = ComplianceMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            type="rate_limit_result",
            domain=message.domain,
            details=result
        )
        self.outbox.put(response)

    async def check_compliance(self, url: str, request_type: str = "GET", headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Check if a request complies with all rules.
        
        Args:
            url: The URL to check.
            request_type: The type of request (GET, POST, etc.).
            headers: The request headers.
            
        Returns:
            A dictionary containing compliance check results.
        """
        domain = urlparse(url).netloc
        current_time = time.time()
        
        # Initialize result
        result = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "rate_limit": None
        }
        
        # Check robots.txt
        if self.compliance_rules["respect_robots_txt"]:
            robots_result = await self._check_robots_txt(url, request_type)
            if not robots_result["allowed"]:
                result["compliant"] = False
                result["violations"].append({
                    "type": "robots_txt",
                    "message": robots_result["message"]
                })
        
        # Check rate limits
        if self.compliance_rules["respect_rate_limits"]:
            rate_limit_result = await self._check_rate_limits(domain, url)
            if not rate_limit_result["allowed"]:
                result["compliant"] = False
                result["violations"].append({
                    "type": "rate_limit",
                    "message": rate_limit_result["message"]
                })
            result["rate_limit"] = rate_limit_result["next_allowed"]
        
        # Check user agent
        if self.compliance_rules["user_agent_required"]:
            if not headers or "User-Agent" not in headers:
                result["compliant"] = False
                result["violations"].append({
                    "type": "user_agent",
                    "message": "User-Agent header is required"
                })
        
        # Check terms of service
        if self.compliance_rules["respect_tos"]:
            tos_result = await self._check_tos(domain, url)
            if not tos_result["allowed"]:
                result["compliant"] = False
                result["violations"].append({
                    "type": "tos",
                    "message": tos_result["message"]
                })
        
        # Record request if compliant
        if result["compliant"]:
            self._record_request(domain, url, request_type, headers)
        
        return result
    
    async def _check_robots_txt(self, url: str, request_type: str) -> Dict[str, Any]:
        """
        Check if a request is allowed by robots.txt.
        
        Args:
            url: The URL to check.
            request_type: The type of request.
            
        Returns:
            A dictionary containing the check result.
        """
        domain = urlparse(url).netloc
        path = urlparse(url).path
        
        # Check cache
        if domain in self.robots_cache:
            cache_data = self.robots_cache[domain]
            if time.time() - cache_data["timestamp"] < self.robots_cache_ttl:
                return self._check_robots_rules(cache_data["rules"], url, request_type)
        
        try:
            # Fetch robots.txt
            robots_url = f"https://{domain}/robots.txt"
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse robots.txt
                        rules = self._parse_robots_txt(content)
                        
                        # Cache rules
                        self.robots_cache[domain] = {
                            "rules": rules,
                            "timestamp": time.time()
                        }
                        
                        return self._check_robots_rules(rules, url, request_type)
                    else:
                        # If robots.txt doesn't exist, assume everything is allowed
                        return {"allowed": True, "message": "No robots.txt found"}
        except Exception as e:
            self.logger.error(f"Error checking robots.txt for {url}: {str(e)}", exc_info=True)
            return {"allowed": True, "message": "Error checking robots.txt, assuming allowed"}
    
    def _parse_robots_txt(self, content: str) -> Dict[str, List[str]]:
        """
        Parse robots.txt content.
        
        Args:
            content: The robots.txt content.
            
        Returns:
            A dictionary of rules.
        """
        rules = {
            "allow": [],
            "disallow": [],
            "crawl_delay": None
        }
        
        current_user_agent = None
        
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == "user-agent":
                    current_user_agent = value
                elif current_user_agent in ["*", "scraper"]:
                    if key == "allow":
                        rules["allow"].append(value)
                    elif key == "disallow":
                        rules["disallow"].append(value)
                    elif key == "crawl-delay":
                        try:
                            rules["crawl_delay"] = float(value)
                        except ValueError:
                            pass
        
        return rules
    
    def _check_robots_rules(self, rules: Dict[str, List[str]], url: str, request_type: str) -> Dict[str, Any]:
        """
        Check if a request is allowed by robots.txt rules.
        
        Args:
            rules: The robots.txt rules.
            url: The URL to check.
            request_type: The type of request.
            
        Returns:
            A dictionary containing the check result.
        """
        path = urlparse(url).path
        
        # Check disallow rules first
        for pattern in rules["disallow"]:
            if self._match_robots_pattern(pattern, path):
                return {
                    "allowed": False,
                    "message": f"URL disallowed by robots.txt: {pattern}"
                }
        
        # Check allow rules
        for pattern in rules["allow"]:
            if self._match_robots_pattern(pattern, path):
                return {
                    "allowed": True,
                    "message": f"URL explicitly allowed by robots.txt: {pattern}"
                }
        
        # If no specific rules match, check crawl delay
        if rules["crawl_delay"] is not None:
            return {
                "allowed": True,
                "message": f"Crawl delay of {rules['crawl_delay']} seconds required"
            }
        
        return {"allowed": True, "message": "No specific rules found"}
    
    def _match_robots_pattern(self, pattern: str, path: str) -> bool:
        """
        Check if a path matches a robots.txt pattern.
        
        Args:
            pattern: The pattern to match.
            path: The path to check.
            
        Returns:
            True if the path matches the pattern, False otherwise.
        """
        # Convert pattern to regex
        pattern = pattern.replace("*", ".*")
        pattern = pattern.replace("?", ".")
        pattern = f"^{pattern}$"
        
        return bool(re.match(pattern, path))
    
    async def _check_rate_limits(self, domain: str, url: str) -> Dict[str, Any]:
        """
        Check if a request would exceed rate limits.
        
        Args:
            domain: The domain to check.
            url: The URL to check.
            
        Returns:
            A dictionary containing the check result.
        """
        current_time = time.time()
        
        # Get request history for domain
        history = self.request_history[domain]
        
        # Clean up old history
        history = [r for r in history if current_time - r["timestamp"] < 3600]
        self.request_history[domain] = history
        
        # Check concurrent requests
        concurrent = len([r for r in history if current_time - r["timestamp"] < self.compliance_rules["min_delay_between_requests"]])
        if concurrent >= self.compliance_rules["max_concurrent_requests"]:
            return {
                "allowed": False,
                "message": "Too many concurrent requests",
                "next_allowed": current_time + self.compliance_rules["min_delay_between_requests"]
            }
        
        # Check requests per hour
        hourly = len([r for r in history if current_time - r["timestamp"] < 3600])
        if hourly >= self.compliance_rules["max_requests_per_hour"]:
            return {
                "allowed": False,
                "message": "Hourly request limit exceeded",
                "next_allowed": min(r["timestamp"] + 3600 for r in history)
            }
        
        # Check requests per day
        daily = len([r for r in history if current_time - r["timestamp"] < 86400])
        if daily >= self.compliance_rules["max_requests_per_day"]:
            return {
                "allowed": False,
                "message": "Daily request limit exceeded",
                "next_allowed": min(r["timestamp"] + 86400 for r in history)
            }
        
        return {"allowed": True, "message": "Rate limits OK"}
    
    async def _check_tos(self, domain: str, url: str) -> Dict[str, Any]:
        """
        Check if a request complies with terms of service.
        
        Args:
            domain: The domain to check.
            url: The URL to check.
            
        Returns:
            A dictionary containing the check result.
        """
        # Check cache
        if domain in self.tos_cache:
            cache_data = self.tos_cache[domain]
            if time.time() - cache_data["timestamp"] < self.tos_cache_ttl:
                return self._check_tos_rules(cache_data["rules"], url)
        
        try:
            # Fetch ToS
            tos_content = await self._fetch_tos(domain)
            
            # Parse ToS
            rules = self._parse_tos(tos_content)
            
            # Cache rules
            self.tos_cache[domain] = {
                "rules": rules,
                "content": tos_content,
                "timestamp": time.time(),
                "hash": hash(tos_content)
            }
            
            return self._check_tos_rules(rules, url)
        except Exception as e:
            self.logger.error(f"Error checking ToS for {url}: {str(e)}", exc_info=True)
            return {"allowed": True, "message": "Error checking ToS, assuming allowed"}
    
    def _parse_tos(self, content: str) -> Dict[str, Any]:
        """
        Parse terms of service content.
        
        Args:
            content: The ToS content.
            
        Returns:
            A dictionary of rules.
        """
        # This is a placeholder for actual ToS parsing
        # In a real implementation, you would parse the actual ToS
        return {}
    
    def _check_tos_rules(self, rules: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Check if a request complies with ToS rules.
        
        Args:
            rules: The ToS rules.
            url: The URL to check.
            
        Returns:
            A dictionary containing the check result.
        """
        # This is a placeholder for actual ToS rule checking
        # In a real implementation, you would check against actual rules
        return {"allowed": True, "message": "No specific rules found"}
    
    def _record_request(self, domain: str, url: str, request_type: str, headers: Optional[Dict[str, str]]) -> None:
        """
        Record a request in the history.
        
        Args:
            domain: The domain of the request.
            url: The URL of the request.
            request_type: The type of request.
            headers: The request headers.
        """
        request = {
            "timestamp": time.time(),
            "url": url,
            "type": request_type,
            "headers": headers
        }
        
        self.request_history[domain].append(request)
        
        # Trim history if needed
        if len(self.request_history[domain]) > self.max_history_size:
            self.request_history[domain] = self.request_history[domain][-self.max_history_size:]
