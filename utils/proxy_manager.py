"""
Proxy manager utility for rotating proxies.
"""
import random
import time
import logging
from typing import Dict, Any, Optional, List, Set
import httpx


class Proxy:
    """
    Represents a proxy server with its status and performance metrics.
    """
    def __init__(self, url: str, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize a new proxy.
        
        Args:
            url: The proxy URL (e.g., "http://proxy.example.com:8080").
            username: Optional username for authentication.
            password: Optional password for authentication.
        """
        self.url = url
        self.username = username
        self.password = password
        self.is_active = True
        self.success_count = 0
        self.failure_count = 0
        self.last_used = 0.0
        self.average_response_time = 0.0
        self.consecutive_failures = 0
    
    def get_url(self) -> str:
        """
        Get the full proxy URL including authentication if provided.
        
        Returns:
            The full proxy URL.
        """
        if self.username and self.password:
            # Extract protocol and host
            protocol, rest = self.url.split("://", 1)
            return f"{protocol}://{self.username}:{self.password}@{rest}"
        return self.url
    
    def report_success(self, response_time: float) -> None:
        """
        Report a successful request through this proxy.
        
        Args:
            response_time: The time taken for the request in seconds.
        """
        self.success_count += 1
        self.last_used = time.time()
        self.consecutive_failures = 0
        
        # Update average response time
        if self.average_response_time == 0:
            self.average_response_time = response_time
        else:
            # Weighted average (70% old, 30% new)
            self.average_response_time = 0.7 * self.average_response_time + 0.3 * response_time
    
    def report_failure(self) -> None:
        """Report a failed request through this proxy."""
        self.failure_count += 1
        self.last_used = time.time()
        self.consecutive_failures += 1
        
        # Deactivate proxy if it fails too many times in a row
        if self.consecutive_failures >= 3:
            self.is_active = False


class ProxyManager:
    """
    Manages a pool of proxies and provides rotation and health checking.
    """
    def __init__(self):
        """Initialize a new proxy manager."""
        self.proxies: Dict[str, Proxy] = {}
        self.active_proxies: Set[str] = set()
        self.logger = logging.getLogger("proxy_manager")
    
    def add_proxy(self, url: str, username: Optional[str] = None, password: Optional[str] = None) -> None:
        """
        Add a proxy to the pool.
        
        Args:
            url: The proxy URL.
            username: Optional username for authentication.
            password: Optional password for authentication.
        """
        proxy = Proxy(url, username, password)
        self.proxies[url] = proxy
        self.active_proxies.add(url)
        self.logger.info(f"Added proxy: {url}")
    
    def remove_proxy(self, url: str) -> None:
        """
        Remove a proxy from the pool.
        
        Args:
            url: The proxy URL to remove.
        """
        if url in self.proxies:
            del self.proxies[url]
            self.active_proxies.discard(url)
            self.logger.info(f"Removed proxy: {url}")
    
    def get_proxy(self) -> Optional[Proxy]:
        """
        Get a proxy from the pool using a selection strategy.
        
        Returns:
            A proxy object, or None if no proxies are available.
        """
        if not self.active_proxies:
            self.logger.warning("No active proxies available")
            return None
        
        # Select a random proxy from the active ones
        proxy_url = random.choice(list(self.active_proxies))
        proxy = self.proxies[proxy_url]
        proxy.last_used = time.time()
        
        return proxy
    
    def report_proxy_result(self, proxy_url: str, success: bool, response_time: Optional[float] = None) -> None:
        """
        Report the result of using a proxy.
        
        Args:
            proxy_url: The URL of the proxy used.
            success: Whether the request was successful.
            response_time: The time taken for the request in seconds.
        """
        if proxy_url not in self.proxies:
            return
        
        proxy = self.proxies[proxy_url]
        
        if success:
            proxy.report_success(response_time or 0.0)
        else:
            proxy.report_failure()
            
            # Remove from active proxies if deactivated
            if not proxy.is_active:
                self.active_proxies.discard(proxy_url)
                self.logger.warning(f"Deactivated proxy due to consecutive failures: {proxy_url}")
    
    async def check_proxies(self, test_url: str = "https://httpbin.org/ip", timeout: float = 10.0) -> None:
        """
        Check the health of all proxies.
        
        Args:
            test_url: The URL to use for testing proxies.
            timeout: Timeout for the test requests in seconds.
        """
        self.logger.info(f"Checking health of {len(self.proxies)} proxies")
        
        for proxy_url, proxy in list(self.proxies.items()):
            try:
                # Create a client with the proxy
                async with httpx.AsyncClient(
                    proxies={"all://": proxy.get_url()},
                    timeout=timeout,
                    follow_redirects=True
                ) as client:
                    start_time = time.time()
                    response = await client.get(test_url)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        proxy.report_success(response_time)
                        self.active_proxies.add(proxy_url)
                        self.logger.debug(f"Proxy {proxy_url} is healthy (response time: {response_time:.2f}s)")
                    else:
                        proxy.report_failure()
                        self.logger.warning(f"Proxy {proxy_url} returned status code {response.status_code}")
                        
                        if not proxy.is_active:
                            self.active_proxies.discard(proxy_url)
            except Exception as e:
                proxy.report_failure()
                self.logger.warning(f"Error checking proxy {proxy_url}: {str(e)}")
                
                if not proxy.is_active:
                    self.active_proxies.discard(proxy_url)
        
        self.logger.info(f"Proxy health check completed. {len(self.active_proxies)} active proxies")
    
    def load_proxies_from_file(self, file_path: str) -> None:
        """
        Load proxies from a file.
        
        Args:
            file_path: Path to the file containing proxy information.
                       Each line should be in the format "url,username,password" or just "url".
        """
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    parts = line.split(",")
                    if len(parts) == 1:
                        self.add_proxy(parts[0])
                    elif len(parts) >= 3:
                        self.add_proxy(parts[0], parts[1], parts[2])
            
            self.logger.info(f"Loaded {len(self.proxies)} proxies from {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading proxies from {file_path}: {str(e)}")
    
    def save_proxies_to_file(self, file_path: str) -> None:
        """
        Save proxies to a file.
        
        Args:
            file_path: Path to save the proxy information to.
        """
        try:
            with open(file_path, "w") as f:
                for proxy_url, proxy in self.proxies.items():
                    if proxy.username and proxy.password:
                        f.write(f"{proxy_url},{proxy.username},{proxy.password}\n")
                    else:
                        f.write(f"{proxy_url}\n")
            
            self.logger.info(f"Saved {len(self.proxies)} proxies to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving proxies to {file_path}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the proxy pool.
        
        Returns:
            A dictionary containing statistics about the proxy pool.
        """
        total_proxies = len(self.proxies)
        active_proxies = len(self.active_proxies)
        
        # Calculate average success rate and response time
        success_rates = []
        response_times = []
        
        for proxy in self.proxies.values():
            total_requests = proxy.success_count + proxy.failure_count
            if total_requests > 0:
                success_rate = proxy.success_count / total_requests
                success_rates.append(success_rate)
            
            if proxy.average_response_time > 0:
                response_times.append(proxy.average_response_time)
        
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "total_proxies": total_proxies,
            "active_proxies": active_proxies,
            "inactive_proxies": total_proxies - active_proxies,
            "average_success_rate": avg_success_rate,
            "average_response_time": avg_response_time
        }
