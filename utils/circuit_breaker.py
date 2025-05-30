"""
Circuit breaker pattern implementation for the web scraping system.
"""
import time
import logging
import asyncio
from enum import Enum
from typing import Callable, Any, Dict, Optional, Awaitable


class CircuitState(str, Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"  # Normal operation, requests are allowed
    OPEN = "open"      # Failing state, requests are blocked
    HALF_OPEN = "half_open"  # Testing state, limited requests allowed


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent repeated failures.
    
    The circuit breaker pattern is used to detect failures and prevent
    the application from repeatedly trying to execute an operation that's
    likely to fail, allowing it to continue without waiting for the failure
    to be fixed or wasting resources while the failure persists.
    """
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        timeout: float = 30.0,
        exclude_exceptions: Optional[list] = None
    ):
        """
        Initialize a new circuit breaker.
        
        Args:
            name: Name of the circuit breaker.
            failure_threshold: Number of failures before opening the circuit.
            recovery_timeout: Time in seconds to wait before trying to recover.
            timeout: Timeout for operations in seconds.
            exclude_exceptions: List of exception types that should not count as failures.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        self.exclude_exceptions = exclude_exceptions or []
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
    
    async def execute(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to execute.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The result of the function.
            
        Raises:
            Exception: If the circuit is open or the function fails.
        """
        self._check_state()
        
        try:
            # Create a task with timeout
            task = asyncio.create_task(func(*args, **kwargs))
            result = await asyncio.wait_for(task, timeout=self.timeout)
            
            # Success, reset failure count
            self._on_success()
            return result
        
        except asyncio.TimeoutError:
            self._on_failure(asyncio.TimeoutError(f"Operation timed out after {self.timeout} seconds"))
            raise
        
        except Exception as e:
            # Check if this exception should be excluded
            if any(isinstance(e, exc_type) for exc_type in self.exclude_exceptions):
                self.logger.info(f"Excluded exception: {str(e)}")
                return None
            
            self._on_failure(e)
            raise
    
    def _check_state(self) -> None:
        """
        Check the current state of the circuit breaker.
        
        Raises:
            Exception: If the circuit is open.
        """
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.logger.info(f"Circuit {self.name} transitioning from OPEN to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit {self.name} is OPEN until {self.last_failure_time + self.recovery_timeout}")
    
    def _on_success(self) -> None:
        """Handle a successful operation."""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.logger.info(f"Circuit {self.name} recovered, transitioning from HALF_OPEN to CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
    
    def _on_failure(self, exception: Exception) -> None:
        """
        Handle a failed operation.
        
        Args:
            exception: The exception that occurred.
        """
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.logger.warning(
                f"Circuit {self.name} tripped after {self.failure_count} failures, "
                f"transitioning from CLOSED to OPEN"
            )
            self.state = CircuitState.OPEN
        
        elif self.state == CircuitState.HALF_OPEN:
            self.logger.warning(
                f"Circuit {self.name} failed during recovery, "
                f"transitioning from HALF_OPEN to OPEN"
            )
            self.state = CircuitState.OPEN
        
        self.logger.error(f"Circuit {self.name} operation failed: {str(exception)}")
    
    def reset(self) -> None:
        """Reset the circuit breaker to its initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.logger.info(f"Circuit {self.name} manually reset to CLOSED state")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the circuit breaker.
        
        Returns:
            A dictionary containing the circuit breaker state.
        """
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "recovery_timeout": self.recovery_timeout
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    """
    def __init__(self):
        """Initialize a new circuit breaker registry."""
        self.circuit_breakers = {}
        self.logger = logging.getLogger("circuit_breaker_registry")
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        timeout: float = 30.0,
        exclude_exceptions: Optional[list] = None
    ) -> CircuitBreaker:
        """
        Get an existing circuit breaker or create a new one.
        
        Args:
            name: Name of the circuit breaker.
            failure_threshold: Number of failures before opening the circuit.
            recovery_timeout: Time in seconds to wait before trying to recover.
            timeout: Timeout for operations in seconds.
            exclude_exceptions: List of exception types that should not count as failures.
            
        Returns:
            The circuit breaker.
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                timeout=timeout,
                exclude_exceptions=exclude_exceptions
            )
            self.logger.info(f"Created new circuit breaker: {name}")
        
        return self.circuit_breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get an existing circuit breaker.
        
        Args:
            name: Name of the circuit breaker.
            
        Returns:
            The circuit breaker, or None if it doesn't exist.
        """
        return self.circuit_breakers.get(name)
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.reset()
        
        self.logger.info("Reset all circuit breakers")
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the states of all circuit breakers.
        
        Returns:
            A dictionary mapping circuit breaker names to their states.
        """
        return {name: cb.get_state() for name, cb in self.circuit_breakers.items()}


# Global registry instance
registry = CircuitBreakerRegistry()
