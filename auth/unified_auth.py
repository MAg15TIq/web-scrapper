"""
Unified Authentication & Session Management System
Provides consistent authentication across CLI and Web interfaces.
"""
import os
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
import json
import uuid

from config.unified_config import get_unified_config_manager, ComponentType


class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"


class AuthMethod(str, Enum):
    """Authentication methods."""
    PASSWORD = "password"
    TOKEN = "token"
    API_KEY = "api_key"
    SESSION = "session"


class SessionStatus(str, Enum):
    """Session status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    INVALID = "invalid"


class User(BaseModel):
    """User model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: Optional[str] = None
    password_hash: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    api_keys: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    
    def check_password(self, password: str) -> bool:
        """Check if the provided password matches the user's password."""
        if not self.password_hash:
            return False
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def set_password(self, password: str) -> None:
        """Set the user's password."""
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return self.is_admin or permission in self.permissions
    
    def add_permission(self, permission: str) -> None:
        """Add a permission to the user."""
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: str) -> None:
        """Remove a permission from the user."""
        if permission in self.permissions:
            self.permissions.remove(permission)


class Session(BaseModel):
    """Session model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    token: str
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime
    last_accessed: datetime = Field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.ACTIVE
    client_info: Dict[str, Any] = Field(default_factory=dict)
    permissions: List[str] = Field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if the session is valid."""
        return (
            self.status == SessionStatus.ACTIVE and
            self.expires_at > datetime.now()
        )
    
    def refresh(self, duration_hours: int = 24) -> None:
        """Refresh the session expiration."""
        self.expires_at = datetime.now() + timedelta(hours=duration_hours)
        self.last_accessed = datetime.now()
    
    def revoke(self) -> None:
        """Revoke the session."""
        self.status = SessionStatus.REVOKED


class UnifiedAuthManager:
    """Unified authentication and session manager."""
    
    def __init__(self):
        """Initialize the unified auth manager."""
        self.logger = logging.getLogger("unified_auth")
        self.config_manager = get_unified_config_manager()
        
        # Authentication settings
        self.secret_key = self._get_secret_key()
        self.token_expiry_hours = 24
        self.session_expiry_hours = 24
        self.max_sessions_per_user = 5
        
        # Storage
        self.users_file = Path("auth/users.json")
        self.sessions_file = Path("auth/sessions.json")
        
        # Ensure auth directory exists
        self.users_file.parent.mkdir(exist_ok=True)
        
        # In-memory storage (for development)
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, List[str]] = {}
        
        # Load existing data
        self._load_users()
        self._load_sessions()
        
        # Create default admin user if none exists
        self._ensure_default_admin()
        
        self.logger.info("Unified authentication manager initialized")
    
    def _get_secret_key(self) -> str:
        """Get or generate the JWT secret key."""
        secret_key = os.getenv("JWT_SECRET_KEY")
        if not secret_key:
            # Generate a new secret key
            secret_key = str(uuid.uuid4()) + str(uuid.uuid4())
            self.logger.warning("No JWT_SECRET_KEY found, generated temporary key")
        return secret_key
    
    def _load_users(self) -> None:
        """Load users from storage."""
        try:
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    users_data = json.load(f)
                    for user_data in users_data.values():
                        user = User(**user_data)
                        self._users[user.id] = user
                self.logger.info(f"Loaded {len(self._users)} users")
        except Exception as e:
            self.logger.error(f"Failed to load users: {e}")
    
    def _load_sessions(self) -> None:
        """Load sessions from storage."""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    sessions_data = json.load(f)
                    for session_data in sessions_data.values():
                        # Convert datetime strings back to datetime objects
                        for field in ['created_at', 'expires_at', 'last_accessed']:
                            if field in session_data:
                                session_data[field] = datetime.fromisoformat(session_data[field])
                        
                        session = Session(**session_data)
                        self._sessions[session.id] = session
                        
                        # Update user sessions mapping
                        if session.user_id not in self._user_sessions:
                            self._user_sessions[session.user_id] = []
                        self._user_sessions[session.user_id].append(session.id)
                
                self.logger.info(f"Loaded {len(self._sessions)} sessions")
        except Exception as e:
            self.logger.error(f"Failed to load sessions: {e}")
    
    def _ensure_default_admin(self) -> None:
        """Ensure a default admin user exists."""
        admin_users = [user for user in self._users.values() if user.is_admin]
        
        if not admin_users:
            # Create default admin user
            admin_user = User(
                username="admin",
                email="admin@webscraper.local",
                role=UserRole.ADMIN,
                is_admin=True,
                permissions=["*"]  # All permissions
            )
            admin_user.set_password("admin123")  # Default password
            
            self._users[admin_user.id] = admin_user
            self._save_users()
            
            self.logger.warning(
                "Created default admin user (username: admin, password: admin123). "
                "Please change the password immediately!"
            )
    
    def _save_users(self) -> None:
        """Save users to storage."""
        try:
            users_data = {}
            for user_id, user in self._users.items():
                users_data[user_id] = user.model_dump()
            
            with open(self.users_file, 'w') as f:
                json.dump(users_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save users: {e}")
    
    def _save_sessions(self) -> None:
        """Save sessions to storage."""
        try:
            sessions_data = {}
            for session_id, session in self._sessions.items():
                sessions_data[session_id] = session.model_dump()
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save sessions: {e}")

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        try:
            # Find user by username
            user = None
            for u in self._users.values():
                if u.username == username:
                    user = u
                    break

            if not user:
                self.logger.warning(f"Authentication failed: user '{username}' not found")
                return None

            if not user.is_active:
                self.logger.warning(f"Authentication failed: user '{username}' is inactive")
                return None

            if not user.check_password(password):
                self.logger.warning(f"Authentication failed: invalid password for user '{username}'")
                return None

            # Update last login
            user.last_login = datetime.now()
            self._save_users()

            self.logger.info(f"User '{username}' authenticated successfully")
            return user

        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None

    def create_session(self, user: User, client_info: Optional[Dict[str, Any]] = None) -> Session:
        """Create a new session for a user."""
        try:
            # Clean up expired sessions for this user
            self._cleanup_user_sessions(user.id)

            # Check session limit
            user_sessions = self._user_sessions.get(user.id, [])
            if len(user_sessions) >= self.max_sessions_per_user:
                # Remove oldest session
                oldest_session_id = user_sessions[0]
                self._revoke_session(oldest_session_id)

            # Create new session
            session = Session(
                user_id=user.id,
                token=self._generate_jwt_token(user),
                expires_at=datetime.now() + timedelta(hours=self.session_expiry_hours),
                client_info=client_info or {},
                permissions=user.permissions.copy()
            )

            # Store session
            self._sessions[session.id] = session

            # Update user sessions mapping
            if user.id not in self._user_sessions:
                self._user_sessions[user.id] = []
            self._user_sessions[user.id].append(session.id)

            self._save_sessions()

            self.logger.info(f"Session created for user '{user.username}' (session: {session.id})")
            return session

        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise

    def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate a session by ID."""
        try:
            session = self._sessions.get(session_id)
            if not session:
                return None

            if not session.is_valid():
                if session.status == SessionStatus.ACTIVE:
                    session.status = SessionStatus.EXPIRED
                    self._save_sessions()
                return None

            # Update last accessed time
            session.last_accessed = datetime.now()
            self._save_sessions()

            return session

        except Exception as e:
            self.logger.error(f"Session validation error: {e}")
            return None

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a JWT token and return user info."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])

            # Check if user still exists and is active
            user_id = payload.get('user_id')
            user = self._users.get(user_id)

            if not user or not user.is_active:
                return None

            return {
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'is_admin': user.is_admin,
                'permissions': user.permissions
            }

        except jwt.ExpiredSignatureError:
            self.logger.debug("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.debug(f"Invalid token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return None

    def _generate_jwt_token(self, user: User) -> str:
        """Generate a JWT token for a user."""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'is_admin': user.is_admin,
            'iat': datetime.now(),
            'exp': datetime.now() + timedelta(hours=self.token_expiry_hours)
        }

        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def _cleanup_user_sessions(self, user_id: str) -> None:
        """Clean up expired sessions for a user."""
        user_sessions = self._user_sessions.get(user_id, [])
        active_sessions = []

        for session_id in user_sessions:
            session = self._sessions.get(session_id)
            if session and session.is_valid():
                active_sessions.append(session_id)
            elif session:
                # Remove expired session
                del self._sessions[session_id]

        self._user_sessions[user_id] = active_sessions
        self._save_sessions()

    def _revoke_session(self, session_id: str) -> None:
        """Revoke a specific session."""
        session = self._sessions.get(session_id)
        if session:
            session.revoke()
            self._save_sessions()

    def revoke_user_sessions(self, user_id: str) -> None:
        """Revoke all sessions for a user."""
        user_sessions = self._user_sessions.get(user_id, [])
        for session_id in user_sessions:
            self._revoke_session(session_id)

        self._user_sessions[user_id] = []
        self.logger.info(f"All sessions revoked for user {user_id}")

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None

    def create_user(self, username: str, password: str, email: Optional[str] = None,
                   role: UserRole = UserRole.USER) -> User:
        """Create a new user."""
        # Check if username already exists
        if self.get_user_by_username(username):
            raise ValueError(f"Username '{username}' already exists")

        user = User(
            username=username,
            email=email,
            role=role,
            is_admin=(role == UserRole.ADMIN)
        )
        user.set_password(password)

        self._users[user.id] = user
        self._save_users()

        self.logger.info(f"User '{username}' created successfully")
        return user


# Global unified auth manager instance
_unified_auth_manager: Optional[UnifiedAuthManager] = None


def get_unified_auth_manager() -> UnifiedAuthManager:
    """Get the global unified auth manager instance."""
    global _unified_auth_manager
    if _unified_auth_manager is None:
        _unified_auth_manager = UnifiedAuthManager()
    return _unified_auth_manager
