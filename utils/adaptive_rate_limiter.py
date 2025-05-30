"""
Adaptive rate limiter for the web scraping system.
"""
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts request rates based on response patterns.
    
    This rate limiter monitors response times, error rates, and other metrics
    to dynamically adjust the rate at which requests are made to a domain.
    """
    def __init__(
        self,
        domain: str,
        initial_rate: float = 1.0,  # requests per second
        min_rate: float = 0.1,      # minimum requests per second (1 per 10 seconds)
        max_rate: float = 10.0,     # maximum requests per second
        window_size: int = 10,      # number of requests to consider for adaptation
        error_penalty: float = 0.5, # rate multiplier on error (0.5 = halve the rate)
        success_bonus: float = 0.1, # rate increase on success (0.1 = increase by 10%)
        backoff_time: float = 60.0  # seconds to back off after consecutive errors
    ):
        """
        Initialize a new adaptive rate limiter.
        
        Args:
            domain: Domain to rate limit.
            initial_rate: Initial request rate in requests per second.
            min_rate: Minimum request rate in requests per second.
            max_rate: Maximum request rate in requests per second.
            window_size: Number of requests to consider for adaptation.
            error_penalty: Rate multiplier on error.
            success_bonus: Rate increase on success.
            backoff_time: Seconds to back off after consecutive errors.
        """
        self.domain = domain
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.window_size = window_size
        self.error_penalty = error_penalty
        self.success_bonus = success_bonus
        self.backoff_time = backoff_time
        
        self.request_times: List[float] = []
        self.response_times: List[float] = []
        self.success_history: List[bool] = []
        self.consecutive_errors = 0
        self.last_request_time = 0
        self.backoff_until = 0
        self.total_requests = 0
        self.total_errors = 0
        
        self.logger = logging.getLogger(f"rate_limiter.{domain}")
    
    async def acquire(self) -> float:
        """
        Acquire permission to make a request.
        
        This method will block until it's safe to make a request
        according to the current rate limit.
        
        Returns:
            The time spent waiting in seconds.
        """
        # Check if we're in a backoff period
        if time.time() < self.backoff_until:
            wait_time = self.backoff_until - time.time()
            self.logger.info(f"In backoff period for {wait_time:.2f} more seconds")
            await asyncio.sleep(wait_time)
        
        # Calculate time to wait based on current rate
        now = time.time()
        time_since_last_request = now - self.last_request_time
        
        # Calculate delay needed to maintain the rate
        delay = (1.0 / self.current_rate) - time_since_last_request
        
        if delay > 0:
            await asyncio.sleep(delay)
            wait_time = delay
        else:
            wait_time = 0
        
        self.last_request_time = time.time()
        self.total_requests += 1
        
        return wait_time
    
    def update(self, success: bool, response_time: Optional[float] = None) -> None:
        """
        Update the rate limiter with the result of a request.
        
        Args:
            success: Whether the request was successful.
            response_time: Response time in seconds, if available.
        """
        # Record response time if provided
        if response_time is not None:
            self.response_times.append(response_time)
            if len(self.response_times) > self.window_size:
                self.response_times.pop(0)
        
        # Record success/failure
        self.success_history.append(success)
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
        
        # Update error count
        if not success:
            self.consecutive_errors += 1
            self.total_errors += 1
        else:
            self.consecutive_errors = 0
        
        # Adjust rate based on success/failure
        if success:
            # Increase rate on success, up to max_rate
            self.current_rate = min(
                self.current_rate * (1.0 + self.success_bonus),
                self.max_rate
            )
        else:
            # Decrease rate on failure, down to min_rate
            self.current_rate = max(
                self.current_rate * self.error_penalty,
                self.min_rate
            )
        
        # If we have too many consecutive errors, back off
        if self.consecutive_errors >= 3:
            self.backoff_until = time.time() + self.backoff_time
            self.logger.warning(
                f"Backing off for {self.backoff_time} seconds after "
                f"{self.consecutive_errors} consecutive errors"
            )
        
        # Log current state
        self.logger.debug(
            f"Rate updated: {self.current_rate:.2f} req/s, "
            f"Success rate: {self._get_success_rate():.2f}"
        )
    
    def _get_success_rate(self) -> float:
        """
        Get the current success rate.
        
        Returns:
            Success rate as a value between 0 and 1.
        """
        if not self.success_history:
            return 1.0
        
        return sum(1 for s in self.success_history if s) / len(self.success_history)
    
    def _get_avg_response_time(self) -> float:
        """
        Get the average response time.
        
        Returns:
            Average response time in seconds.
        """
        if not self.response_times:
            return 0.0
        
        return sum(self.response_times) / len(self.response_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the rate limiter.
        
        Returns:
            A dictionary of statistics.
        """
        return {
            "domain": self.domain,
            "current_rate": self.current_rate,
            "success_rate": self._get_success_rate(),
            "avg_response_time": self._get_avg_response_time(),
            "consecutive_errors": self.consecutive_errors,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "in_backoff": time.time() < self.backoff_until,
            "backoff_remaining": max(0, self.backoff_until - time.time())
        }
    
    def reset(self) -> None:
        """Reset the rate limiter to its initial state."""
        self.current_rate = self.min_rate
        self.request_times = []
        self.response_times = []
        self.success_history = []
        self.consecutive_errors = 0
        self.backoff_until = 0
        self.logger.info(f"Rate limiter for {self.domain} reset")


class RateLimiterRegistry:
    """
    Registry for managing multiple rate limiters.
    """
    def __init__(self):
        """Initialize a new rate limiter registry."""
        self.rate_limiters: Dict[str, AdaptiveRateLimiter] = {}
        self.logger = logging.getLogger("rate_limiter_registry")
    
    def get_or_create(
        self,
        domain: str,
        initial_rate: float = 1.0,
        min_rate: float = 0.1,
        max_rate: float = 10.0,
        window_size: int = 10,
        error_penalty: float = 0.5,
        success_bonus: float = 0.1,
        backoff_time: float = 60.0
    ) -> AdaptiveRateLimiter:
        """
        Get an existing rate limiter or create a new one.
        
        Args:
            domain: Domain to rate limit.
            initial_rate: Initial request rate in requests per second.
            min_rate: Minimum request rate in requests per second.
            max_rate: Maximum request rate in requests per second.
            window_size: Number of requests to consider for adaptation.
            error_penalty: Rate multiplier on error.
            success_bonus: Rate increase on success.
            backoff_time: Seconds to back off after consecutive errors.
            
        Returns:
            The rate limiter.
        """
        if domain not in self.rate_limiters:
            self.rate_limiters[domain] = AdaptiveRateLimiter(
                domain=domain,
                initial_rate=initial_rate,
                min_rate=min_rate,
                max_rate=max_rate,
                window_size=window_size,
                error_penalty=error_penalty,
                success_bonus=success_bonus,
                backoff_time=backoff_time
            )
            self.logger.info(f"Created new rate limiter for {domain}")
        
        return self.rate_limiters[domain]
    
    def get(self, domain: str) -> Optional[AdaptiveRateLimiter]:
        """
        Get an existing rate limiter.
        
        Args:
            domain: Domain to get rate limiter for.
            
        Returns:
            The rate limiter, or None if it doesn't exist.
        """
        return self.rate_limiters.get(domain)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all rate limiters.
        
        Returns:
            A dictionary mapping domains to their statistics.
        """
        return {domain: limiter.get_stats() for domain, limiter in self.rate_limiters.items()}
    
    def reset_all(self) -> None:
        """Reset all rate limiters."""
        for limiter in self.rate_limiters.values():
            limiter.reset()
        
        self.logger.info("Reset all rate limiters")


# Global registry instance
registry = RateLimiterRegistry()
