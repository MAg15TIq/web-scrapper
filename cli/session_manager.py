"""
Session Manager for Enhanced CLI
Handles session persistence, command history, and state management.
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict


@dataclass
class CommandHistoryEntry:
    """Command history entry."""
    timestamp: datetime
    command: str
    parsed_command: Optional[Dict[str, Any]]
    execution_time: Optional[float]
    success: bool
    error_message: Optional[str] = None
    session_id: Optional[str] = None


class SessionState(BaseModel):
    """Session state model."""
    
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity time")
    profile: str = Field("default", description="Active profile")
    current_directory: str = Field(".", description="Current working directory")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Session environment variables")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context data")
    active_agents: List[str] = Field(default_factory=list, description="Currently active agents")
    command_count: int = Field(0, description="Number of commands executed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionManager:
    """Session manager for CLI state and history."""
    
    def __init__(self, session_dir: Optional[str] = None):
        """Initialize the session manager."""
        self.logger = logging.getLogger("session_manager")
        
        # Set session directory
        if session_dir:
            self.session_dir = Path(session_dir)
        else:
            self.session_dir = Path.home() / ".webscraper_cli" / "sessions"
        
        # Ensure session directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # History file
        self.history_file = self.session_dir / "command_history.json"
        
        # Current session
        self.current_session: Optional[SessionState] = None
        self.current_session_id: Optional[str] = None
        
        # Command history
        self.command_history: List[CommandHistoryEntry] = []
        self.max_history_size = 1000
        
        # Load existing history
        self._load_command_history()
        
        self.logger.info(f"Session manager initialized with directory: {self.session_dir}")
    
    def start_session(self, profile: str = "default") -> str:
        """Start a new session."""
        try:
            session_id = str(uuid.uuid4())
            
            self.current_session = SessionState(
                session_id=session_id,
                profile=profile,
                current_directory=os.getcwd()
            )
            
            self.current_session_id = session_id
            
            # Save session state
            self._save_session_state()
            
            self.logger.info(f"New session started: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error starting session: {e}")
            raise
    
    def end_session(self, session_id: Optional[str] = None) -> bool:
        """End the current or specified session."""
        try:
            target_session_id = session_id or self.current_session_id
            
            if not target_session_id:
                self.logger.warning("No active session to end")
                return False
            
            # Save final session state
            if self.current_session and self.current_session.session_id == target_session_id:
                self._save_session_state()
                self.current_session = None
                self.current_session_id = None
            
            # Archive session file
            session_file = self.session_dir / f"session_{target_session_id}.json"
            if session_file.exists():
                archive_file = self.session_dir / "archived" / f"session_{target_session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                archive_file.parent.mkdir(exist_ok=True)
                session_file.rename(archive_file)
            
            self.logger.info(f"Session ended: {target_session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ending session: {e}")
            return False
    
    def restore_session(self, session_id: str) -> bool:
        """Restore a previous session."""
        try:
            session_file = self.session_dir / f"session_{session_id}.json"
            
            if not session_file.exists():
                self.logger.error(f"Session file not found: {session_id}")
                return False
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Parse datetime fields
            if 'created_at' in session_data:
                session_data['created_at'] = datetime.fromisoformat(session_data['created_at'])
            if 'last_activity' in session_data:
                session_data['last_activity'] = datetime.fromisoformat(session_data['last_activity'])
            
            self.current_session = SessionState(**session_data)
            self.current_session_id = session_id
            
            # Update last activity
            self.current_session.last_activity = datetime.now()
            self._save_session_state()
            
            self.logger.info(f"Session restored: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring session: {e}")
            return False
    
    def add_command_to_history(self, command: str, parsed_command: Optional[Dict[str, Any]] = None,
                             execution_time: Optional[float] = None, success: bool = True,
                             error_message: Optional[str] = None):
        """Add a command to the history."""
        try:
            entry = CommandHistoryEntry(
                timestamp=datetime.now(),
                command=command,
                parsed_command=parsed_command,
                execution_time=execution_time,
                success=success,
                error_message=error_message,
                session_id=self.current_session_id
            )
            
            self.command_history.append(entry)
            
            # Update session command count
            if self.current_session:
                self.current_session.command_count += 1
                self.current_session.last_activity = datetime.now()
                self._save_session_state()
            
            # Trim history if too large
            if len(self.command_history) > self.max_history_size:
                self.command_history = self.command_history[-self.max_history_size:]
            
            # Save history
            self._save_command_history()
            
            self.logger.debug(f"Command added to history: {command[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Error adding command to history: {e}")
    
    def get_command_history(self, limit: Optional[int] = None, session_id: Optional[str] = None) -> List[CommandHistoryEntry]:
        """Get command history."""
        history = self.command_history
        
        # Filter by session if specified
        if session_id:
            history = [entry for entry in history if entry.session_id == session_id]
        
        # Apply limit
        if limit:
            history = history[-limit:]
        
        return history
    
    def search_command_history(self, query: str, limit: int = 10) -> List[CommandHistoryEntry]:
        """Search command history."""
        query_lower = query.lower()
        matches = []
        
        for entry in reversed(self.command_history):
            if query_lower in entry.command.lower():
                matches.append(entry)
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_session_context(self, key: Optional[str] = None) -> Any:
        """Get session context data."""
        if not self.current_session:
            return None
        
        if key:
            return self.current_session.context.get(key)
        
        return self.current_session.context
    
    def set_session_context(self, key: str, value: Any) -> bool:
        """Set session context data."""
        try:
            if not self.current_session:
                self.logger.warning("No active session for setting context")
                return False
            
            self.current_session.context[key] = value
            self.current_session.last_activity = datetime.now()
            self._save_session_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting session context: {e}")
            return False
    
    def update_session_state(self, **kwargs) -> bool:
        """Update session state."""
        try:
            if not self.current_session:
                self.logger.warning("No active session to update")
                return False
            
            for key, value in kwargs.items():
                if hasattr(self.current_session, key):
                    setattr(self.current_session, key, value)
            
            self.current_session.last_activity = datetime.now()
            self._save_session_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating session state: {e}")
            return False
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions."""
        active_sessions = []
        
        try:
            for session_file in self.session_dir.glob("session_*.json"):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Check if session is recent (within last 24 hours)
                last_activity = datetime.fromisoformat(session_data.get('last_activity', '1970-01-01'))
                if datetime.now() - last_activity < timedelta(hours=24):
                    active_sessions.append({
                        'session_id': session_data['session_id'],
                        'created_at': session_data['created_at'],
                        'last_activity': session_data['last_activity'],
                        'profile': session_data.get('profile', 'default'),
                        'command_count': session_data.get('command_count', 0)
                    })
            
            return sorted(active_sessions, key=lambda x: x['last_activity'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting active sessions: {e}")
            return []
    
    def cleanup_old_sessions(self, days: int = 7) -> int:
        """Clean up old session files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cleaned_count = 0
            
            for session_file in self.session_dir.glob("session_*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    last_activity = datetime.fromisoformat(session_data.get('last_activity', '1970-01-01'))
                    
                    if last_activity < cutoff_date:
                        # Move to archive
                        archive_dir = self.session_dir / "archived"
                        archive_dir.mkdir(exist_ok=True)
                        
                        archive_file = archive_dir / f"old_{session_file.name}"
                        session_file.rename(archive_file)
                        cleaned_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error processing session file {session_file}: {e}")
            
            self.logger.info(f"Cleaned up {cleaned_count} old sessions")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old sessions: {e}")
            return 0
    
    def _save_session_state(self):
        """Save current session state to file."""
        if not self.current_session:
            return
        
        try:
            session_file = self.session_dir / f"session_{self.current_session.session_id}.json"
            
            with open(session_file, 'w') as f:
                json.dump(self.current_session.dict(), f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving session state: {e}")
    
    def _load_command_history(self):
        """Load command history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
                
                self.command_history = []
                for entry_data in history_data:
                    entry_data['timestamp'] = datetime.fromisoformat(entry_data['timestamp'])
                    self.command_history.append(CommandHistoryEntry(**entry_data))
                
                self.logger.info(f"Loaded {len(self.command_history)} command history entries")
                
        except Exception as e:
            self.logger.warning(f"Could not load command history: {e}")
            self.command_history = []
    
    def _save_command_history(self):
        """Save command history to file."""
        try:
            history_data = []
            for entry in self.command_history:
                entry_dict = asdict(entry)
                entry_dict['timestamp'] = entry.timestamp.isoformat()
                history_data.append(entry_dict)
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving command history: {e}")
    
    @property
    def current_session_info(self) -> Optional[Dict[str, Any]]:
        """Get current session information."""
        if not self.current_session:
            return None
        
        return {
            'session_id': self.current_session.session_id,
            'created_at': self.current_session.created_at.isoformat(),
            'last_activity': self.current_session.last_activity.isoformat(),
            'profile': self.current_session.profile,
            'command_count': self.current_session.command_count,
            'active_agents': self.current_session.active_agents
        }
