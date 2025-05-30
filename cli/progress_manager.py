"""
Progress Manager for Enhanced CLI
Handles advanced progress tracking, notifications, and real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading

from rich.console import Console
from rich.progress import (
    Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
)
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class NotificationType(Enum):
    """Notification type enumeration."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TaskInfo:
    """Task information structure."""
    task_id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    total: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    agent_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    subtasks: List['TaskInfo'] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class Notification:
    """Notification structure."""
    id: str
    type: NotificationType
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    task_id: Optional[str] = None
    auto_dismiss: bool = True
    dismiss_after: int = 5  # seconds


class ProgressManager:
    """Advanced progress manager with real-time updates and notifications."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the progress manager."""
        self.console = console or Console()
        self.logger = logging.getLogger("progress_manager")
        
        # Task tracking
        self.tasks: Dict[str, TaskInfo] = {}
        self.active_tasks: List[str] = []
        
        # Progress display
        self.progress_display: Optional[Progress] = None
        self.live_display: Optional[Live] = None
        self.rich_task_ids: Dict[str, TaskID] = {}
        
        # Notifications
        self.notifications: List[Notification] = []
        self.max_notifications = 10
        self.notification_callbacks: List[Callable[[Notification], None]] = []
        
        # Display settings
        self.show_live_updates = True
        self.show_agent_colors = True
        self.compact_mode = False
        
        # Agent color mapping
        self.agent_colors = {
            'Scraper Agent': 'green',
            'Parser Agent': 'blue',
            'Storage Agent': 'magenta',
            'JavaScript Agent': 'yellow',
            'Anti-Detection Agent': 'red',
            'Data Transformation Agent': 'cyan',
            'Error Recovery Agent': 'bright_red',
            'Master Intelligence Agent': 'gold1',
            'Coordinator Agent': 'bright_blue'
        }
        
        self.logger.info("Progress manager initialized")
    
    def create_task(self, task_id: str, name: str, description: str, 
                   total: Optional[float] = None, agent_type: Optional[str] = None) -> TaskInfo:
        """Create a new task for tracking."""
        task_info = TaskInfo(
            task_id=task_id,
            name=name,
            description=description,
            total=total,
            agent_type=agent_type
        )
        
        self.tasks[task_id] = task_info
        self.logger.debug(f"Created task: {task_id} - {name}")
        
        return task_info
    
    def start_task(self, task_id: str) -> bool:
        """Start tracking a task."""
        if task_id not in self.tasks:
            self.logger.error(f"Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        # Add to active tasks
        if task_id not in self.active_tasks:
            self.active_tasks.append(task_id)
        
        # Add to progress display if active
        if self.progress_display and task_id not in self.rich_task_ids:
            color = self.agent_colors.get(task.agent_type, 'white')
            task_description = f"[{color}]{task.name}[/{color}]: {task.description}"
            
            rich_task_id = self.progress_display.add_task(
                task_description,
                total=task.total
            )
            self.rich_task_ids[task_id] = rich_task_id
        
        # Send notification
        self.add_notification(
            NotificationType.INFO,
            "Task Started",
            f"Started {task.name}",
            task_id=task_id
        )
        
        self.logger.info(f"Started task: {task_id}")
        return True
    
    def update_task(self, task_id: str, progress: Optional[float] = None, 
                   description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update task progress and information."""
        if task_id not in self.tasks:
            self.logger.error(f"Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        # Update progress
        if progress is not None:
            task.progress = progress
            
            # Update rich progress if active
            if self.progress_display and task_id in self.rich_task_ids:
                if task.total:
                    completed = (progress / 100.0) * task.total if progress <= 100 else task.total
                    self.progress_display.update(self.rich_task_ids[task_id], completed=completed)
                else:
                    self.progress_display.update(self.rich_task_ids[task_id], advance=1)
        
        # Update description
        if description:
            task.description = description
            
            if self.progress_display and task_id in self.rich_task_ids:
                color = self.agent_colors.get(task.agent_type, 'white')
                new_description = f"[{color}]{task.name}[/{color}]: {description}"
                self.progress_display.update(self.rich_task_ids[task_id], description=new_description)
        
        # Update metadata
        if metadata:
            task.metadata.update(metadata)
        
        return True
    
    def complete_task(self, task_id: str, success: bool = True, error_message: Optional[str] = None) -> bool:
        """Mark a task as completed."""
        if task_id not in self.tasks:
            self.logger.error(f"Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.end_time = datetime.now()
        task.error_message = error_message
        
        # Remove from active tasks
        if task_id in self.active_tasks:
            self.active_tasks.remove(task_id)
        
        # Update progress display
        if self.progress_display and task_id in self.rich_task_ids:
            if success:
                self.progress_display.update(self.rich_task_ids[task_id], completed=task.total or 100)
            else:
                # Mark as failed in display
                color = 'red'
                description = f"[{color}]{task.name}[/{color}]: FAILED"
                self.progress_display.update(self.rich_task_ids[task_id], description=description)
        
        # Send notification
        notification_type = NotificationType.SUCCESS if success else NotificationType.ERROR
        title = "Task Completed" if success else "Task Failed"
        message = f"{'Completed' if success else 'Failed'} {task.name}"
        if error_message:
            message += f": {error_message}"
        
        self.add_notification(notification_type, title, message, task_id=task_id)
        
        self.logger.info(f"{'Completed' if success else 'Failed'} task: {task_id}")
        return True
    
    def start_live_display(self) -> None:
        """Start live progress display."""
        if self.live_display:
            return
        
        # Create progress display
        self.progress_display = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False
        )
        
        # Create layout
        layout = self._create_live_layout()
        
        # Start live display
        self.live_display = Live(layout, console=self.console, refresh_per_second=4)
        self.live_display.start()
        
        self.logger.info("Started live progress display")
    
    def stop_live_display(self) -> None:
        """Stop live progress display."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
        
        if self.progress_display:
            self.progress_display = None
        
        self.rich_task_ids.clear()
        self.logger.info("Stopped live progress display")
    
    def _create_live_layout(self) -> Layout:
        """Create the live display layout."""
        layout = Layout()
        
        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress"),
            Layout(name="notifications", size=8),
            Layout(name="footer", size=3)
        )
        
        # Header
        header_text = Text("🚀 Multi-Agent Web Scraping Progress", style="bold bright_blue", justify="center")
        layout["header"].update(Panel(header_text, border_style="bright_blue"))
        
        # Progress section
        layout["progress"].update(self.progress_display)
        
        # Notifications section
        layout["notifications"].update(self._create_notifications_panel())
        
        # Footer
        footer_text = Text(f"Active Tasks: {len(self.active_tasks)} | Total Tasks: {len(self.tasks)}", 
                          style="dim", justify="center")
        layout["footer"].update(Panel(footer_text, border_style="dim"))
        
        return layout
    
    def _create_notifications_panel(self) -> Panel:
        """Create notifications panel."""
        if not self.notifications:
            return Panel(
                Text("No notifications", style="dim", justify="center"),
                title="Notifications",
                border_style="yellow"
            )
        
        # Create notifications table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Time", style="dim", width=8)
        table.add_column("Type", width=8)
        table.add_column("Message", ratio=1)
        
        # Show recent notifications
        recent_notifications = self.notifications[-5:]
        
        for notification in recent_notifications:
            # Color based on type
            type_colors = {
                NotificationType.INFO: "blue",
                NotificationType.SUCCESS: "green",
                NotificationType.WARNING: "yellow",
                NotificationType.ERROR: "red"
            }
            
            color = type_colors.get(notification.type, "white")
            time_str = notification.timestamp.strftime("%H:%M:%S")
            type_str = f"[{color}]{notification.type.value.upper()}[/{color}]"
            message = f"[bold]{notification.title}[/bold]: {notification.message}"
            
            table.add_row(time_str, type_str, message)
        
        return Panel(table, title="Recent Notifications", border_style="yellow")
    
    def add_notification(self, notification_type: NotificationType, title: str, 
                        message: str, task_id: Optional[str] = None) -> None:
        """Add a notification."""
        notification = Notification(
            id=f"notif_{int(time.time() * 1000)}",
            type=notification_type,
            title=title,
            message=message,
            task_id=task_id
        )
        
        self.notifications.append(notification)
        
        # Trim notifications if too many
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        # Call notification callbacks
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                self.logger.error(f"Error in notification callback: {e}")
        
        self.logger.debug(f"Added notification: {title}")
    
    def add_notification_callback(self, callback: Callable[[Notification], None]) -> None:
        """Add a notification callback."""
        self.notification_callbacks.append(callback)
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of all tasks."""
        summary = {
            'total_tasks': len(self.tasks),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
            'failed_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
            'tasks_by_agent': {},
            'average_completion_time': None
        }
        
        # Tasks by agent
        for task in self.tasks.values():
            if task.agent_type:
                if task.agent_type not in summary['tasks_by_agent']:
                    summary['tasks_by_agent'][task.agent_type] = 0
                summary['tasks_by_agent'][task.agent_type] += 1
        
        # Average completion time
        completed_tasks = [t for t in self.tasks.values() 
                          if t.status == TaskStatus.COMPLETED and t.start_time and t.end_time]
        
        if completed_tasks:
            total_time = sum((t.end_time - t.start_time).total_seconds() for t in completed_tasks)
            summary['average_completion_time'] = total_time / len(completed_tasks)
        
        return summary
    
    def get_active_tasks(self) -> List[TaskInfo]:
        """Get list of active tasks."""
        return [self.tasks[task_id] for task_id in self.active_tasks if task_id in self.tasks]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.status = TaskStatus.CANCELLED
        task.end_time = datetime.now()
        
        # Remove from active tasks
        if task_id in self.active_tasks:
            self.active_tasks.remove(task_id)
        
        # Update display
        if self.progress_display and task_id in self.rich_task_ids:
            self.progress_display.remove_task(self.rich_task_ids[task_id])
            del self.rich_task_ids[task_id]
        
        self.add_notification(
            NotificationType.WARNING,
            "Task Cancelled",
            f"Cancelled {task.name}",
            task_id=task_id
        )
        
        return True
    
    def clear_completed_tasks(self) -> int:
        """Clear completed tasks from tracking."""
        completed_task_ids = [
            task_id for task_id, task in self.tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        
        for task_id in completed_task_ids:
            del self.tasks[task_id]
            if task_id in self.rich_task_ids:
                del self.rich_task_ids[task_id]
        
        return len(completed_task_ids)
