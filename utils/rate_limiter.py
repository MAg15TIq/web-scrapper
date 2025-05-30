"""
Rate limiter utility for controlling request rates.
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, Deque
from collections import deque


class RateLimiter:
    """
    Rate limiter for controlling the rate of requests to a domain.
    """
    def __init__(self, max_calls: int = 1, period: float = 1.0):
        """
        Initialize a new rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period.
            period: Time period in seconds.
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: Deque[float] = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire permission to make a request.
        This method will block until a request can be made without exceeding the rate limit.
        """
        async with self.lock:
            # Remove calls that are outside the time window
            current_time = time.time()
            while self.calls and current_time - self.calls[0] > self.period:
                self.calls.popleft()
            
            # If we've reached the maximum number of calls, wait until we can make another
            if len(self.calls) >= self.max_calls:
                wait_time = self.period - (current_time - self.calls[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Update current time after sleeping
                    current_time = time.time()
                    # Remove calls that are now outside the time window
                    while self.calls and current_time - self.calls[0] > self.period:
                        self.calls.popleft()
            
            # Add the current call
            self.calls.append(current_time)
    
    def reset(self) -> None:
        """Reset the rate limiter by clearing all recorded calls."""
        self.calls.clear()


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts the rate based on server responses.
    """
    def __init__(self, initial_max_calls: int = 1, initial_period: float = 1.0,
                 min_max_calls: int = 1, max_max_calls: int = 10):
        """
        Initialize a new adaptive rate limiter.
        
        Args:
            initial_max_calls: Initial maximum number of calls allowed in the period.
            initial_period: Initial time period in seconds.
            min_max_calls: Minimum value for max_calls.
            max_max_calls: Maximum value for max_calls.
        """
        super().__init__(max_calls=initial_max_calls, period=initial_period)
        self.min_max_calls = min_max_calls
        self.max_max_calls = max_max_calls
        self.success_count = 0
        self.failure_count = 0
        self.adjustment_threshold = 10  # Number of requests before adjusting the rate
    
    def report_success(self) -> None:
        """Report a successful request."""
        self.success_count += 1
        self._adjust_rate()
    
    def report_failure(self, status_code: int) -> None:
        """
        Report a failed request.
        
        Args:
            status_code: HTTP status code of the failed request.
        """
        self.failure_count += 1
        
        # If we get a 429 (Too Many Requests), immediately reduce the rate
        if status_code == 429:
            self._reduce_rate(factor=2.0)
        
        self._adjust_rate()
    
    def _adjust_rate(self) -> None:
        """Adjust the rate based on success and failure counts."""
        total = self.success_count + self.failure_count
        
        if total >= self.adjustment_threshold:
            success_ratio = self.success_count / total
            
            if success_ratio > 0.95:
                # More than 95% success, increase the rate
                self._increase_rate()
            elif success_ratio < 0.75:
                # Less than 75% success, decrease the rate
                self._reduce_rate()
            
            # Reset counters
            self.success_count = 0
            self.failure_count = 0
    
    def _increase_rate(self, factor: float = 1.2) -> None:
        """
        Increase the rate by increasing max_calls.
        
        Args:
            factor: Factor to multiply max_calls by.
        """
        new_max_calls = min(int(self.max_calls * factor) + 1, self.max_max_calls)
        if new_max_calls > self.max_calls:
            self.max_calls = new_max_calls
    
    def _reduce_rate(self, factor: float = 1.5) -> None:
        """
        Reduce the rate by decreasing max_calls or increasing the period.
        
        Args:
            factor: Factor to divide max_calls by or multiply period by.
        """
        if self.max_calls > self.min_max_calls:
            # Reduce max_calls
            self.max_calls = max(int(self.max_calls / factor), self.min_max_calls)
        else:
            # Increase period
            self.period *= factor
        
        # Clear calls to apply the new rate immediately
        self.calls.clear()
