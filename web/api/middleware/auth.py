"""
Authentication Middleware
Handles authentication and authorization for API requests.
"""

import logging
import time
from typing import Optional, List
from datetime import datetime

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt

from config.web_config import get_web_config


# Configure logging
logger = logging.getLogger("auth_middleware")


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API requests."""
    
    def __init__(self, app):
        super().__init__(app)
        self.config = get_web_config()
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/",
            "/health",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/static"
        }
        
        # Admin-only endpoints
        self.admin_endpoints = {
            "/api/v1/agents/system/metrics",
            "/api/v1/monitoring/logs",
            "/api/v1/monitoring/alerts"
        }
        
        logger.info("Authentication middleware initialized")
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""
        start_time = time.time()
        
        try:
            # Skip authentication for public endpoints
            if self._is_public_endpoint(request.url.path):
                response = await call_next(request)
                return self._add_security_headers(response)
            
            # Skip authentication if disabled in config
            if not self.config.enable_auth:
                response = await call_next(request)
                return self._add_security_headers(response)
            
            # Extract and validate token
            token = self._extract_token(request)
            if not token:
                return self._create_auth_error_response("Missing authentication token")
            
            # Verify token and get user info
            user_info = self._verify_token(token)
            if not user_info:
                return self._create_auth_error_response("Invalid or expired token")
            
            # Check if user is active
            if not user_info.get("is_active", False):
                return self._create_auth_error_response("User account is inactive")
            
            # Check admin permissions for admin endpoints
            if self._is_admin_endpoint(request.url.path):
                if not user_info.get("is_admin", False):
                    return self._create_permission_error_response("Admin access required")
            
            # Add user info to request state
            request.state.user = user_info
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            response = self._add_security_headers(response)
            
            # Log successful request
            process_time = time.time() - start_time
            logger.debug(
                f"Authenticated request: {request.method} {request.url.path} "
                f"by {user_info.get('username')} in {process_time:.3f}s"
            )
            
            return response
            
        except HTTPException as e:
            # Handle HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": e.status_code,
                        "message": e.detail,
                        "type": "authentication_error"
                    }
                }
            )
        
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Authentication middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": 500,
                        "message": "Internal authentication error",
                        "type": "server_error"
                    }
                }
            )
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (doesn't require authentication)."""
        # Exact match
        if path in self.public_endpoints:
            return True
        
        # Prefix match for static files and docs
        public_prefixes = ["/static/", "/api/docs", "/api/redoc"]
        for prefix in public_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    def _is_admin_endpoint(self, path: str) -> bool:
        """Check if endpoint requires admin permissions."""
        return path in self.admin_endpoints
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from request headers."""
        # Check Authorization header
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            return authorization.split(" ")[1]
        
        # Check query parameter (for WebSocket connections)
        token = request.query_params.get("token")
        if token:
            return token
        
        # Check cookie
        token = request.cookies.get("access_token")
        if token:
            return token
        
        return None
    
    def _verify_token(self, token: str) -> Optional[dict]:
        """Verify JWT token and return user information."""
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.config.get_secret_key(),
                algorithms=[self.config.security.algorithm]
            )
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                logger.warning("Token has expired")
                return None
            
            # Extract user information
            user_info = {
                "id": payload.get("sub"),
                "username": payload.get("username"),
                "email": payload.get("email"),
                "is_active": payload.get("is_active", True),
                "is_admin": payload.get("is_admin", False)
            }
            
            return user_info
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def _create_auth_error_response(self, message: str) -> JSONResponse:
        """Create authentication error response."""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": 401,
                    "message": message,
                    "type": "authentication_error"
                }
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    def _create_permission_error_response(self, message: str) -> JSONResponse:
        """Create permission error response."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": {
                    "code": 403,
                    "message": message,
                    "type": "authorization_error"
                }
            }
        )
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response."""
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # CORS headers (if not already set)
        if "Access-Control-Allow-Origin" not in response.headers:
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        # Cache control for API responses
        if response.headers.get("content-type", "").startswith("application/json"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


class RoleBasedAccessControl:
    """Role-based access control helper."""
    
    def __init__(self):
        self.role_permissions = {
            "admin": [
                "read:all",
                "write:all",
                "delete:all",
                "manage:users",
                "manage:system"
            ],
            "user": [
                "read:own",
                "write:own",
                "read:public"
            ],
            "viewer": [
                "read:public"
            ]
        }
    
    def has_permission(self, user_role: str, required_permission: str) -> bool:
        """Check if user role has required permission."""
        user_permissions = self.role_permissions.get(user_role, [])
        return required_permission in user_permissions
    
    def get_user_permissions(self, user_role: str) -> List[str]:
        """Get all permissions for a user role."""
        return self.role_permissions.get(user_role, [])


class SessionManager:
    """Session management for authenticated users."""
    
    def __init__(self):
        self.active_sessions = {}
        self.session_timeout = 3600  # 1 hour
    
    def create_session(self, user_id: str, token: str) -> str:
        """Create a new user session."""
        session_id = f"session_{user_id}_{int(time.time())}"
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "token": token,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "ip_address": None,
            "user_agent": None
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if session is still active."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        last_activity = session["last_activity"]
        
        # Check if session has expired
        if (datetime.now() - last_activity).total_seconds() > self.session_timeout:
            self.invalidate_session(session_id)
            return False
        
        # Update last activity
        session["last_activity"] = datetime.now()
        return True
    
    def invalidate_session(self, session_id: str):
        """Invalidate a user session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if (current_time - session["last_activity"]).total_seconds() > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
        
        return len(expired_sessions)


# Global instances
rbac = RoleBasedAccessControl()
session_manager = SessionManager()
