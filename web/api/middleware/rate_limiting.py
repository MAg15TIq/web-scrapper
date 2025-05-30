"""
Rate Limiting Middleware
Implements rate limiting for API requests to prevent abuse.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.web_config import get_web_config


# Configure logging
logger = logging.getLogger("rate_limit_middleware")


class TokenBucket:
    """Token bucket algorithm implementation for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Rate at which tokens are added (tokens per second)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        
        self.last_refill = now
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available."""
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class SlidingWindowCounter:
    """Sliding window counter for rate limiting."""
    
    def __init__(self, window_size: int, max_requests: int):
        """
        Initialize sliding window counter.
        
        Args:
            window_size: Window size in seconds
            max_requests: Maximum requests allowed in window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed within rate limit."""
        now = time.time()
        
        # Remove old requests outside the window
        while self.requests and self.requests[0] <= now - self.window_size:
            self.requests.popleft()
        
        # Check if we're within the limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def get_reset_time(self) -> float:
        """Get time when the oldest request will expire."""
        if not self.requests:
            return 0.0
        
        return self.requests[0] + self.window_size


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for API requests."""
    
    def __init__(self, app):
        super().__init__(app)
        self.config = get_web_config()
        
        # Rate limit storage
        self.client_buckets: Dict[str, TokenBucket] = {}
        self.client_windows: Dict[str, SlidingWindowCounter] = {}
        
        # Rate limit rules
        self.rate_limits = {
            "default": {
                "requests_per_minute": self.config.rate_limit.requests_per_minute,
                "requests_per_hour": self.config.rate_limit.requests_per_hour,
                "burst_size": self.config.rate_limit.burst_size
            },
            "auth": {
                "requests_per_minute": 10,
                "requests_per_hour": 100,
                "burst_size": 5
            },
            "websocket": {
                "requests_per_minute": 120,
                "requests_per_hour": 3600,
                "burst_size": 20
            }
        }
        
        # Exempt endpoints
        self.exempt_endpoints = {
            "/health",
            "/",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json"
        }
        
        # Cleanup task
        self._cleanup_task = None
        if self.config.rate_limit.enabled:
            self._start_cleanup_task()
        
        logger.info(f"Rate limiting middleware initialized (enabled: {self.config.rate_limit.enabled})")
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting middleware."""
        # Skip rate limiting if disabled
        if not self.config.rate_limit.enabled:
            return await call_next(request)
        
        # Skip rate limiting for exempt endpoints
        if self._is_exempt_endpoint(request.url.path):
            return await call_next(request)
        
        try:
            # Get client identifier
            client_id = self._get_client_id(request)
            
            # Get rate limit rule for endpoint
            rule_name = self._get_rate_limit_rule(request.url.path)
            rate_limit = self.rate_limits[rule_name]
            
            # Check rate limits
            is_allowed, headers = self._check_rate_limits(client_id, rate_limit)
            
            if not is_allowed:
                return self._create_rate_limit_response(headers)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            for key, value in headers.items():
                response.headers[key] = str(value)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Continue processing if rate limiting fails
            return await call_next(request)
    
    def _is_exempt_endpoint(self, path: str) -> bool:
        """Check if endpoint is exempt from rate limiting."""
        return path in self.exempt_endpoints
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user.get('id', 'unknown')}"
        
        # Fall back to IP address
        client_ip = request.client.host
        
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"
    
    def _get_rate_limit_rule(self, path: str) -> str:
        """Get rate limit rule name for endpoint."""
        if path.startswith("/api/v1/auth/"):
            return "auth"
        elif path.startswith("/ws/"):
            return "websocket"
        else:
            return "default"
    
    def _check_rate_limits(self, client_id: str, rate_limit: Dict) -> Tuple[bool, Dict[str, int]]:
        """Check if client is within rate limits."""
        # Initialize client rate limiters if not exists
        if client_id not in self.client_buckets:
            self.client_buckets[client_id] = TokenBucket(
                capacity=rate_limit["burst_size"],
                refill_rate=rate_limit["requests_per_minute"] / 60.0
            )
        
        if client_id not in self.client_windows:
            self.client_windows[client_id] = SlidingWindowCounter(
                window_size=3600,  # 1 hour
                max_requests=rate_limit["requests_per_hour"]
            )
        
        bucket = self.client_buckets[client_id]
        window = self.client_windows[client_id]
        
        # Check burst limit (token bucket)
        burst_allowed = bucket.consume(1)
        
        # Check hourly limit (sliding window)
        hourly_allowed = window.is_allowed()
        
        # Determine if request is allowed
        is_allowed = burst_allowed and hourly_allowed
        
        # Calculate headers
        headers = {
            "X-RateLimit-Limit-Minute": rate_limit["requests_per_minute"],
            "X-RateLimit-Limit-Hour": rate_limit["requests_per_hour"],
            "X-RateLimit-Remaining-Burst": int(bucket.tokens),
            "X-RateLimit-Remaining-Hour": max(0, rate_limit["requests_per_hour"] - len(window.requests))
        }
        
        if not is_allowed:
            if not burst_allowed:
                headers["X-RateLimit-Reset-Burst"] = int(time.time() + bucket.get_wait_time(1))
            if not hourly_allowed:
                headers["X-RateLimit-Reset-Hour"] = int(window.get_reset_time())
        
        return is_allowed, headers
    
    def _create_rate_limit_response(self, headers: Dict[str, int]) -> JSONResponse:
        """Create rate limit exceeded response."""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "details": {
                        "retry_after": headers.get("X-RateLimit-Reset-Burst", headers.get("X-RateLimit-Reset-Hour", 60))
                    }
                }
            },
            headers={k: str(v) for k, v in headers.items()}
        )
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(300)  # Run every 5 minutes
                    self._cleanup_old_entries()
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_task())
    
    def _cleanup_old_entries(self):
        """Clean up old rate limiting entries."""
        current_time = time.time()
        cleanup_threshold = 3600  # 1 hour
        
        # Clean up token buckets
        expired_buckets = []
        for client_id, bucket in self.client_buckets.items():
            if current_time - bucket.last_refill > cleanup_threshold:
                expired_buckets.append(client_id)
        
        for client_id in expired_buckets:
            del self.client_buckets[client_id]
        
        # Clean up sliding windows
        expired_windows = []
        for client_id, window in self.client_windows.items():
            # Remove old requests
            while window.requests and window.requests[0] <= current_time - window.window_size:
                window.requests.popleft()
            
            # Remove empty windows
            if not window.requests:
                expired_windows.append(client_id)
        
        for client_id in expired_windows:
            del self.client_windows[client_id]
        
        if expired_buckets or expired_windows:
            logger.debug(f"Cleaned up {len(expired_buckets)} buckets and {len(expired_windows)} windows")
    
    def get_rate_limit_stats(self) -> Dict[str, any]:
        """Get rate limiting statistics."""
        return {
            "active_clients": len(self.client_buckets),
            "total_buckets": len(self.client_buckets),
            "total_windows": len(self.client_windows),
            "rate_limits": self.rate_limits,
            "enabled": self.config.rate_limit.enabled
        }
    
    def reset_client_limits(self, client_id: str) -> bool:
        """Reset rate limits for a specific client."""
        reset_count = 0
        
        if client_id in self.client_buckets:
            del self.client_buckets[client_id]
            reset_count += 1
        
        if client_id in self.client_windows:
            del self.client_windows[client_id]
            reset_count += 1
        
        return reset_count > 0
    
    def __del__(self):
        """Cleanup when middleware is destroyed."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
