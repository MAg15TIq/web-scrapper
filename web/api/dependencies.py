"""
FastAPI Dependencies for Multi-Agent Web Scraping System
Common dependencies used across API endpoints.
"""

import logging
from typing import Optional, Dict, Any, Generator
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt
from passlib.context import CryptContext

from config.web_config import get_web_config
from cli.agent_communication import AgentCommunicationLayer
from web.dashboard.agent_monitor import AgentMonitor
from web.scheduler.job_manager import JobManager


# Configure logging
logger = logging.getLogger("api_dependencies")

# Security
security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Global instances (will be properly initialized in main.py)
_agent_manager: Optional[AgentCommunicationLayer] = None
_agent_monitor: Optional[AgentMonitor] = None
_job_manager: Optional[JobManager] = None
_database_session: Optional[Session] = None


class AuthenticationError(Exception):
    """Authentication error."""
    pass


class AuthorizationError(Exception):
    """Authorization error."""
    pass


def get_config():
    """Get web configuration."""
    return get_web_config()


def get_agent_manager() -> AgentCommunicationLayer:
    """Get agent communication layer."""
    global _agent_manager
    
    if _agent_manager is None:
        _agent_manager = AgentCommunicationLayer()
    
    return _agent_manager


def get_agent_monitor(request: Request) -> AgentMonitor:
    """Get agent monitor from application state."""
    if hasattr(request.app.state, 'agent_monitor'):
        return request.app.state.agent_monitor
    
    # Fallback: create new instance
    logger.warning("Agent monitor not found in app state, creating new instance")
    return AgentMonitor()


def get_job_manager(request: Request) -> JobManager:
    """Get job manager from application state."""
    if hasattr(request.app.state, 'job_manager'):
        return request.app.state.job_manager
    
    # Fallback: create new instance
    logger.warning("Job manager not found in app state, creating new instance")
    return JobManager()


def get_database() -> Generator[Session, None, None]:
    """Get database session."""
    # This is a placeholder for actual database session management
    # In a real implementation, you would use SQLAlchemy session management
    
    class MockSession:
        """Mock database session for development."""
        
        def query(self, *args, **kwargs):
            return self
        
        def filter(self, *args, **kwargs):
            return self
        
        def first(self):
            return None
        
        def all(self):
            return []
        
        def add(self, obj):
            pass
        
        def commit(self):
            pass
        
        def rollback(self):
            pass
        
        def close(self):
            pass
    
    session = MockSession()
    try:
        yield session
    finally:
        session.close()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    config = get_config()
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.security.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        config.get_secret_key(),
        algorithm=config.security.algorithm
    )
    
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return payload."""
    config = get_config()
    
    try:
        payload = jwt.decode(
            token,
            config.get_secret_key(),
            algorithms=[config.security.algorithm]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.JWTError:
        raise AuthenticationError("Invalid token")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_database)
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user."""
    config = get_config()
    
    # If authentication is disabled, return mock user
    if not config.enable_auth:
        return {
            "id": "mock_user",
            "username": "developer",
            "email": "dev@example.com",
            "is_active": True,
            "is_admin": True
        }
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # In a real implementation, you would fetch the user from the database
        # For now, return mock user data
        user = {
            "id": user_id,
            "username": payload.get("username", "unknown"),
            "email": payload.get("email", "unknown@example.com"),
            "is_active": True,
            "is_admin": payload.get("is_admin", False)
        }
        
        return user
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current active user."""
    if not current_user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get current admin user."""
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def check_rate_limit(request: Request) -> bool:
    """Check rate limit for request."""
    config = get_config()
    
    if not config.enable_rate_limiting:
        return True
    
    # Simple in-memory rate limiting (in production, use Redis)
    client_ip = request.client.host
    current_time = datetime.now()
    
    # This is a simplified implementation
    # In production, you would use a proper rate limiting library
    return True


def validate_file_upload(file_size: int, file_extension: str) -> bool:
    """Validate file upload."""
    config = get_config()
    
    if not config.enable_file_uploads:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="File uploads are disabled"
        )
    
    if file_size > config.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {config.max_upload_size} bytes"
        )
    
    if file_extension.lower() not in config.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File extension '{file_extension}' is not allowed"
        )
    
    return True


def get_pagination_params(
    page: int = 1,
    size: int = 20,
    max_size: int = 100
) -> Dict[str, int]:
    """Get pagination parameters."""
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be greater than 0"
        )
    
    if size < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Size must be greater than 0"
        )
    
    if size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Size cannot exceed {max_size}"
        )
    
    offset = (page - 1) * size
    
    return {
        "page": page,
        "size": size,
        "offset": offset,
        "limit": size
    }


def validate_agent_type(agent_type: str) -> str:
    """Validate agent type."""
    valid_agent_types = [
        "orchestrator",
        "scraper",
        "parser",
        "storage",
        "javascript",
        "authentication",
        "anti_detection",
        "data_transformation",
        "error_recovery",
        "data_extractor"
    ]
    
    if agent_type not in valid_agent_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid agent type. Must be one of: {', '.join(valid_agent_types)}"
        )
    
    return agent_type


def validate_job_status(status: str) -> str:
    """Validate job status."""
    valid_statuses = [
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "paused"
    ]
    
    if status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid job status. Must be one of: {', '.join(valid_statuses)}"
        )
    
    return status


def get_request_context(request: Request) -> Dict[str, Any]:
    """Get request context information."""
    return {
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent", ""),
        "method": request.method,
        "url": str(request.url),
        "timestamp": datetime.utcnow().isoformat()
    }
