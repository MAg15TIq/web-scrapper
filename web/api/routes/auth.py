"""
Authentication API Routes
Handles user authentication, authorization, and session management.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr, validator

from web.api.dependencies import (
    get_current_user, get_current_active_user, get_config,
    create_access_token, verify_password, get_password_hash
)
from config.web_config import WebConfig


# Configure logging
logger = logging.getLogger("auth_api")

# Create router
router = APIRouter()


# Pydantic models
class UserRegistration(BaseModel):
    """User registration model."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError("Username must contain only alphanumeric characters")
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(BaseModel):
    """User login model."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")


class UserProfile(BaseModel):
    """User profile model."""
    id: str
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login: Optional[datetime]
    preferences: Dict[str, Any]


class PasswordChange(BaseModel):
    """Password change model."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


@router.post("/register", response_model=UserProfile)
async def register_user(
    user_data: UserRegistration,
    config: WebConfig = Depends(get_config)
):
    """
    Register a new user.
    
    Creates a new user account with the provided information.
    Returns the user profile upon successful registration.
    """
    try:
        logger.info(f"Registering new user: {user_data.username}")
        
        # Check if registration is enabled
        if not config.enable_auth:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User registration is disabled"
            )
        
        # In a real implementation, you would:
        # 1. Check if username/email already exists
        # 2. Hash the password
        # 3. Store user in database
        # 4. Send verification email if required
        
        # For now, create a mock user
        hashed_password = get_password_hash(user_data.password)
        
        user_profile = UserProfile(
            id=f"user_{user_data.username}",
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            is_active=True,
            is_admin=False,
            created_at=datetime.now(),
            last_login=None,
            preferences={}
        )
        
        logger.info(f"User registered successfully: {user_data.username}")
        return user_profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    config: WebConfig = Depends(get_config)
):
    """
    Authenticate user and return access token.
    
    Validates user credentials and returns a JWT access token
    for authenticated requests.
    """
    try:
        logger.info(f"Login attempt for user: {form_data.username}")
        
        # In a real implementation, you would:
        # 1. Query user from database
        # 2. Verify password hash
        # 3. Check if user is active
        # 4. Update last login timestamp
        
        # For now, use mock authentication
        if form_data.username == "admin" and form_data.password == "admin123":
            user_data = {
                "sub": "admin",
                "username": "admin",
                "email": "admin@example.com",
                "is_admin": True
            }
        elif form_data.username == "user" and form_data.password == "user123":
            user_data = {
                "sub": "user",
                "username": "user",
                "email": "user@example.com",
                "is_admin": False
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=config.security.access_token_expire_minutes)
        access_token = create_access_token(
            data=user_data,
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in successfully: {form_data.username}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=config.security.access_token_expire_minutes * 60,
            refresh_token=None  # Could implement refresh tokens
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get current user profile.
    
    Returns the profile information for the currently authenticated user.
    """
    try:
        logger.debug(f"Getting profile for user: {current_user.get('username')}")
        
        # In a real implementation, you would query the database for full user info
        return UserProfile(
            id=current_user["id"],
            username=current_user["username"],
            email=current_user["email"],
            full_name=current_user.get("full_name"),
            is_active=current_user["is_active"],
            is_admin=current_user["is_admin"],
            created_at=datetime.now() - timedelta(days=30),  # Mock creation date
            last_login=datetime.now() - timedelta(hours=1),   # Mock last login
            preferences=current_user.get("preferences", {})
        )
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@router.put("/profile", response_model=UserProfile)
async def update_user_profile(
    profile_update: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Update user profile.
    
    Updates the profile information for the currently authenticated user.
    """
    try:
        logger.info(f"Updating profile for user: {current_user.get('username')}")
        
        # Validate allowed fields
        allowed_fields = ["full_name", "email", "preferences"]
        update_data = {k: v for k, v in profile_update.items() if k in allowed_fields}
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields to update"
            )
        
        # In a real implementation, you would update the database
        # For now, return updated mock profile
        updated_user = current_user.copy()
        updated_user.update(update_data)
        
        return UserProfile(
            id=updated_user["id"],
            username=updated_user["username"],
            email=updated_user.get("email", current_user["email"]),
            full_name=updated_user.get("full_name"),
            is_active=updated_user["is_active"],
            is_admin=updated_user["is_admin"],
            created_at=datetime.now() - timedelta(days=30),
            last_login=datetime.now() - timedelta(hours=1),
            preferences=updated_user.get("preferences", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Change user password.
    
    Changes the password for the currently authenticated user.
    """
    try:
        logger.info(f"Password change request for user: {current_user.get('username')}")
        
        # In a real implementation, you would:
        # 1. Verify current password
        # 2. Hash new password
        # 3. Update password in database
        # 4. Optionally invalidate existing tokens
        
        # For now, just validate the request
        if password_data.current_password == password_data.new_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password"
            )
        
        # Mock password verification
        # In reality, you would verify against stored hash
        
        logger.info(f"Password changed successfully for user: {current_user.get('username')}")
        
        return {
            "success": True,
            "message": "Password changed successfully",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.post("/logout")
async def logout_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Logout user.
    
    Invalidates the current user session and token.
    """
    try:
        logger.info(f"Logout request for user: {current_user.get('username')}")
        
        # In a real implementation, you would:
        # 1. Add token to blacklist
        # 2. Clear session data
        # 3. Update last logout timestamp
        
        return {
            "success": True,
            "message": "Logged out successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/refresh")
async def refresh_token(
    refresh_token: str = Form(...),
    config: WebConfig = Depends(get_config)
):
    """
    Refresh access token.
    
    Generates a new access token using a valid refresh token.
    """
    try:
        logger.debug("Token refresh request")
        
        # In a real implementation, you would:
        # 1. Validate refresh token
        # 2. Check if token is not blacklisted
        # 3. Generate new access token
        # 4. Optionally rotate refresh token
        
        # For now, return a mock response
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Token refresh not implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )
