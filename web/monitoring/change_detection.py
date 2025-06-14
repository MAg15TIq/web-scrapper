"""
Change Detection & Monitoring System for Phase 2 Enhancement

This module provides website change detection with diff algorithms,
content freshness monitoring, and automatic re-scraping triggers.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import difflib
import re

import httpx
from bs4 import BeautifulSoup


# Configure logging
logger = logging.getLogger("change_detection")


class ChangeType(Enum):
    """Types of changes that can be detected."""
    CONTENT_ADDED = "content_added"
    CONTENT_REMOVED = "content_removed"
    CONTENT_MODIFIED = "content_modified"
    STRUCTURE_CHANGED = "structure_changed"
    NEW_ELEMENTS = "new_elements"
    REMOVED_ELEMENTS = "removed_elements"
    ATTRIBUTE_CHANGED = "attribute_changed"
    TEXT_CHANGED = "text_changed"


class MonitoringFrequency(Enum):
    """Monitoring frequency options."""
    REAL_TIME = "real_time"  # Continuous monitoring
    HIGH = "high"           # Every 5 minutes
    MEDIUM = "medium"       # Every 30 minutes
    LOW = "low"            # Every 2 hours
    DAILY = "daily"        # Once per day
    CUSTOM = "custom"      # Custom interval


@dataclass
class ChangeEvent:
    """Represents a detected change."""
    change_id: str
    url: str
    change_type: ChangeType
    timestamp: datetime
    description: str
    old_value: Optional[str]
    new_value: Optional[str]
    xpath: Optional[str]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class MonitoringTarget:
    """Configuration for a monitoring target."""
    target_id: str
    url: str
    name: str
    frequency: MonitoringFrequency
    custom_interval: Optional[int]  # seconds for custom frequency
    selectors: List[str]  # CSS selectors to monitor
    ignore_patterns: List[str]  # Patterns to ignore
    threshold: float  # Change threshold (0.0 - 1.0)
    enabled: bool
    last_check: Optional[datetime]
    last_content_hash: Optional[str]
    change_history: List[str]  # List of change IDs


class ChangeDetectionEngine:
    """
    Enhanced Change Detection & Monitoring System for Phase 2.
    
    Features:
    - Website change detection with diff algorithms
    - Content freshness monitoring
    - Automatic re-scraping triggers based on changes
    - Alert system for significant content modifications
    - Intelligent change classification
    """
    
    def __init__(self):
        self.logger = logging.getLogger("change_detection")
        
        # Monitoring targets and state
        self.monitoring_targets: Dict[str, MonitoringTarget] = {}
        self.change_events: Dict[str, ChangeEvent] = {}
        self.change_history: deque = deque(maxlen=10000)
        
        # Content snapshots for comparison
        self.content_snapshots: Dict[str, Dict[str, Any]] = {}
        
        # HTTP client for fetching content
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        
        # Background monitoring tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
        # Change detection algorithms
        self.diff_algorithms = {
            'text': self._text_diff_algorithm,
            'structure': self._structure_diff_algorithm,
            'semantic': self._semantic_diff_algorithm,
            'visual': self._visual_diff_algorithm
        }
        
        # Performance metrics
        self.metrics = {
            "total_checks": 0,
            "changes_detected": 0,
            "false_positives": 0,
            "monitoring_targets": 0,
            "average_check_time": 0.0
        }
        
        self.logger.info("Change detection engine initialized")
    
    async def start(self):
        """Start the change detection engine."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring for existing targets
        for target_id in self.monitoring_targets:
            await self._start_monitoring_target(target_id)
        
        # Start background tasks
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._cleanup_old_data())
        
        self.logger.info("Change detection engine started")
    
    async def stop(self):
        """Stop the change detection engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        self.monitoring_tasks.clear()
        
        # Close HTTP client
        await self.http_client.aclose()
        
        self.logger.info("Change detection engine stopped")
    
    async def add_monitoring_target(self, target_config: Dict[str, Any]) -> str:
        """Add a new monitoring target."""
        try:
            target = MonitoringTarget(
                target_id=target_config.get('target_id', f"target_{int(time.time())}"),
                url=target_config['url'],
                name=target_config.get('name', target_config['url']),
                frequency=MonitoringFrequency(target_config.get('frequency', 'medium')),
                custom_interval=target_config.get('custom_interval'),
                selectors=target_config.get('selectors', []),
                ignore_patterns=target_config.get('ignore_patterns', []),
                threshold=target_config.get('threshold', 0.1),
                enabled=target_config.get('enabled', True),
                last_check=None,
                last_content_hash=None,
                change_history=[]
            )
            
            self.monitoring_targets[target.target_id] = target
            self.metrics["monitoring_targets"] += 1
            
            # Take initial snapshot
            await self._take_content_snapshot(target.target_id)
            
            # Start monitoring if engine is running
            if self.is_running and target.enabled:
                await self._start_monitoring_target(target.target_id)
            
            self.logger.info(f"Monitoring target added: {target.target_id} - {target.url}")
            return target.target_id
            
        except Exception as e:
            self.logger.error(f"Error adding monitoring target: {e}")
            raise
    
    async def remove_monitoring_target(self, target_id: str) -> bool:
        """Remove a monitoring target."""
        try:
            if target_id not in self.monitoring_targets:
                return False
            
            # Stop monitoring task
            if target_id in self.monitoring_tasks:
                self.monitoring_tasks[target_id].cancel()
                del self.monitoring_tasks[target_id]
            
            # Remove target and data
            del self.monitoring_targets[target_id]
            
            if target_id in self.content_snapshots:
                del self.content_snapshots[target_id]
            
            self.metrics["monitoring_targets"] = max(0, self.metrics["monitoring_targets"] - 1)
            
            self.logger.info(f"Monitoring target removed: {target_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing monitoring target: {e}")
            return False
    
    async def check_for_changes(self, target_id: str) -> List[ChangeEvent]:
        """Manually check for changes on a specific target."""
        try:
            if target_id not in self.monitoring_targets:
                raise ValueError(f"Monitoring target not found: {target_id}")
            
            target = self.monitoring_targets[target_id]
            start_time = time.time()
            
            # Fetch current content
            current_content = await self._fetch_content(target.url)
            if not current_content:
                return []
            
            # Get previous snapshot
            previous_snapshot = self.content_snapshots.get(target_id)
            if not previous_snapshot:
                # First check - take snapshot and return
                await self._take_content_snapshot(target_id, current_content)
                return []
            
            # Detect changes
            changes = await self._detect_changes(target, previous_snapshot, current_content)
            
            # Update snapshot if significant changes detected
            if changes:
                await self._take_content_snapshot(target_id, current_content)
                
                # Store change events
                for change in changes:
                    self.change_events[change.change_id] = change
                    self.change_history.append(change.change_id)
                    target.change_history.append(change.change_id)
            
            # Update target metadata
            target.last_check = datetime.now()
            
            # Update metrics
            check_time = time.time() - start_time
            self.metrics["total_checks"] += 1
            self.metrics["changes_detected"] += len(changes)
            
            # Update average check time
            current_avg = self.metrics["average_check_time"]
            total_checks = self.metrics["total_checks"]
            self.metrics["average_check_time"] = (current_avg * (total_checks - 1) + check_time) / total_checks
            
            if changes:
                self.logger.info(f"Changes detected for {target_id}: {len(changes)} changes")
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error checking for changes on {target_id}: {e}")
            return []
    
    async def get_change_events(self, target_id: Optional[str] = None, 
                               since: Optional[datetime] = None,
                               change_types: Optional[List[ChangeType]] = None) -> List[ChangeEvent]:
        """Get change events with optional filtering."""
        try:
            events = []
            
            # Get relevant change IDs
            if target_id:
                if target_id in self.monitoring_targets:
                    change_ids = self.monitoring_targets[target_id].change_history
                else:
                    return []
            else:
                change_ids = list(self.change_history)
            
            # Filter events
            for change_id in change_ids:
                if change_id in self.change_events:
                    event = self.change_events[change_id]
                    
                    # Filter by timestamp
                    if since and event.timestamp < since:
                        continue
                    
                    # Filter by change type
                    if change_types and event.change_type not in change_types:
                        continue
                    
                    events.append(event)
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda e: e.timestamp, reverse=True)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting change events: {e}")
            return []
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and metrics."""
        try:
            active_targets = sum(1 for t in self.monitoring_targets.values() if t.enabled)
            
            # Calculate recent activity
            recent_changes = len([
                e for e in self.change_events.values()
                if (datetime.now() - e.timestamp).total_seconds() < 3600  # Last hour
            ])
            
            return {
                "is_running": self.is_running,
                "total_targets": len(self.monitoring_targets),
                "active_targets": active_targets,
                "active_monitoring_tasks": len(self.monitoring_tasks),
                "total_change_events": len(self.change_events),
                "recent_changes": recent_changes,
                "metrics": self.metrics,
                "targets": {
                    target_id: {
                        "name": target.name,
                        "url": target.url,
                        "frequency": target.frequency.value,
                        "enabled": target.enabled,
                        "last_check": target.last_check.isoformat() if target.last_check else None,
                        "change_count": len(target.change_history)
                    }
                    for target_id, target in self.monitoring_targets.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting monitoring status: {e}")
            return {"error": str(e)}

    # ===== PRIVATE METHODS =====

    async def _start_monitoring_target(self, target_id: str):
        """Start monitoring a specific target."""
        try:
            if target_id in self.monitoring_tasks:
                return  # Already monitoring

            target = self.monitoring_targets[target_id]
            if not target.enabled:
                return

            # Create monitoring task
            task = asyncio.create_task(self._monitoring_loop(target_id))
            self.monitoring_tasks[target_id] = task

            self.logger.info(f"Started monitoring: {target_id}")

        except Exception as e:
            self.logger.error(f"Error starting monitoring for {target_id}: {e}")

    async def _monitoring_loop(self, target_id: str):
        """Main monitoring loop for a target."""
        while self.is_running and target_id in self.monitoring_targets:
            try:
                target = self.monitoring_targets[target_id]

                if not target.enabled:
                    break

                # Check for changes
                changes = await self.check_for_changes(target_id)

                # Trigger alerts or actions if changes detected
                if changes:
                    await self._handle_changes_detected(target_id, changes)

                # Calculate next check interval
                interval = self._get_check_interval(target.frequency, target.custom_interval)

                # Wait for next check
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop for {target_id}: {e}")
                await asyncio.sleep(60)  # Wait before retrying

        # Clean up
        if target_id in self.monitoring_tasks:
            del self.monitoring_tasks[target_id]

    def _get_check_interval(self, frequency: MonitoringFrequency, custom_interval: Optional[int]) -> int:
        """Get check interval in seconds based on frequency."""
        intervals = {
            MonitoringFrequency.REAL_TIME: 30,      # 30 seconds
            MonitoringFrequency.HIGH: 300,          # 5 minutes
            MonitoringFrequency.MEDIUM: 1800,       # 30 minutes
            MonitoringFrequency.LOW: 7200,          # 2 hours
            MonitoringFrequency.DAILY: 86400,       # 24 hours
            MonitoringFrequency.CUSTOM: custom_interval or 1800
        }

        return intervals.get(frequency, 1800)

    async def _fetch_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch content from a URL."""
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()

            content = {
                'url': url,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'content': response.text,
                'timestamp': datetime.now(),
                'content_hash': hashlib.md5(response.text.encode()).hexdigest()
            }

            return content

        except Exception as e:
            self.logger.error(f"Error fetching content from {url}: {e}")
            return None

    async def _take_content_snapshot(self, target_id: str, content: Optional[Dict[str, Any]] = None):
        """Take a content snapshot for a target."""
        try:
            target = self.monitoring_targets[target_id]

            if not content:
                content = await self._fetch_content(target.url)

            if not content:
                return

            # Parse HTML for structural analysis
            soup = BeautifulSoup(content['content'], 'html.parser')

            # Create comprehensive snapshot
            snapshot = {
                'target_id': target_id,
                'timestamp': content['timestamp'],
                'content_hash': content['content_hash'],
                'raw_content': content['content'],
                'parsed_content': {
                    'title': soup.title.string if soup.title else None,
                    'text_content': soup.get_text(strip=True),
                    'element_count': len(soup.find_all()),
                    'link_count': len(soup.find_all('a')),
                    'image_count': len(soup.find_all('img')),
                    'form_count': len(soup.find_all('form'))
                },
                'monitored_elements': {}
            }

            # Extract content for monitored selectors
            for selector in target.selectors:
                try:
                    elements = soup.select(selector)
                    snapshot['monitored_elements'][selector] = [
                        {
                            'text': elem.get_text(strip=True),
                            'html': str(elem),
                            'attributes': dict(elem.attrs) if hasattr(elem, 'attrs') else {}
                        }
                        for elem in elements[:10]  # Limit to first 10 elements
                    ]
                except Exception as e:
                    self.logger.warning(f"Error extracting selector {selector}: {e}")

            # Store snapshot
            self.content_snapshots[target_id] = snapshot

            # Update target hash
            target.last_content_hash = content['content_hash']

        except Exception as e:
            self.logger.error(f"Error taking snapshot for {target_id}: {e}")

    async def _detect_changes(self, target: MonitoringTarget,
                             previous_snapshot: Dict[str, Any],
                             current_content: Dict[str, Any]) -> List[ChangeEvent]:
        """Detect changes between snapshots."""
        changes = []

        try:
            # Parse current content
            current_soup = BeautifulSoup(current_content['content'], 'html.parser')

            # Create current snapshot structure
            current_snapshot = {
                'parsed_content': {
                    'title': current_soup.title.string if current_soup.title else None,
                    'text_content': current_soup.get_text(strip=True),
                    'element_count': len(current_soup.find_all()),
                    'link_count': len(current_soup.find_all('a')),
                    'image_count': len(current_soup.find_all('img')),
                    'form_count': len(current_soup.find_all('form'))
                },
                'monitored_elements': {}
            }

            # Extract current monitored elements
            for selector in target.selectors:
                try:
                    elements = current_soup.select(selector)
                    current_snapshot['monitored_elements'][selector] = [
                        {
                            'text': elem.get_text(strip=True),
                            'html': str(elem),
                            'attributes': dict(elem.attrs) if hasattr(elem, 'attrs') else {}
                        }
                        for elem in elements[:10]
                    ]
                except Exception:
                    current_snapshot['monitored_elements'][selector] = []

            # Check for content hash change
            if previous_snapshot['content_hash'] != current_content['content_hash']:
                # Detailed change detection
                changes.extend(await self._analyze_content_changes(
                    target, previous_snapshot, current_snapshot
                ))

            # Filter changes based on threshold and ignore patterns
            filtered_changes = []
            for change in changes:
                if self._should_report_change(change, target):
                    filtered_changes.append(change)

            return filtered_changes

        except Exception as e:
            self.logger.error(f"Error detecting changes: {e}")
            return []

    async def _analyze_content_changes(self, target: MonitoringTarget,
                                      previous: Dict[str, Any],
                                      current: Dict[str, Any]) -> List[ChangeEvent]:
        """Analyze detailed content changes."""
        changes = []
        timestamp = datetime.now()

        try:
            # Check title changes
            prev_title = previous['parsed_content'].get('title')
            curr_title = current['parsed_content'].get('title')

            if prev_title != curr_title:
                changes.append(ChangeEvent(
                    change_id=f"change_{int(time.time())}_{len(changes)}",
                    url=target.url,
                    change_type=ChangeType.TEXT_CHANGED,
                    timestamp=timestamp,
                    description="Page title changed",
                    old_value=prev_title,
                    new_value=curr_title,
                    xpath="//title",
                    confidence=1.0,
                    metadata={"element": "title"}
                ))

            # Check structural changes
            prev_counts = previous['parsed_content']
            curr_counts = current['parsed_content']

            for count_type in ['element_count', 'link_count', 'image_count', 'form_count']:
                prev_count = prev_counts.get(count_type, 0)
                curr_count = curr_counts.get(count_type, 0)

                if abs(prev_count - curr_count) > 0:
                    change_type = ChangeType.STRUCTURE_CHANGED
                    if curr_count > prev_count:
                        change_type = ChangeType.NEW_ELEMENTS
                    elif curr_count < prev_count:
                        change_type = ChangeType.REMOVED_ELEMENTS

                    changes.append(ChangeEvent(
                        change_id=f"change_{int(time.time())}_{len(changes)}",
                        url=target.url,
                        change_type=change_type,
                        timestamp=timestamp,
                        description=f"{count_type.replace('_', ' ').title()} changed from {prev_count} to {curr_count}",
                        old_value=str(prev_count),
                        new_value=str(curr_count),
                        xpath=None,
                        confidence=0.8,
                        metadata={"metric": count_type}
                    ))

            # Check monitored elements
            for selector in target.selectors:
                prev_elements = previous.get('monitored_elements', {}).get(selector, [])
                curr_elements = current.get('monitored_elements', {}).get(selector, [])

                # Compare element content
                if len(prev_elements) != len(curr_elements):
                    changes.append(ChangeEvent(
                        change_id=f"change_{int(time.time())}_{len(changes)}",
                        url=target.url,
                        change_type=ChangeType.STRUCTURE_CHANGED,
                        timestamp=timestamp,
                        description=f"Number of elements matching '{selector}' changed",
                        old_value=str(len(prev_elements)),
                        new_value=str(len(curr_elements)),
                        xpath=selector,
                        confidence=0.9,
                        metadata={"selector": selector}
                    ))

                # Compare element text content
                for i, (prev_elem, curr_elem) in enumerate(zip(prev_elements, curr_elements)):
                    if prev_elem.get('text') != curr_elem.get('text'):
                        changes.append(ChangeEvent(
                            change_id=f"change_{int(time.time())}_{len(changes)}",
                            url=target.url,
                            change_type=ChangeType.CONTENT_MODIFIED,
                            timestamp=timestamp,
                            description=f"Content changed in element {i+1} of '{selector}'",
                            old_value=prev_elem.get('text', ''),
                            new_value=curr_elem.get('text', ''),
                            xpath=f"{selector}[{i+1}]",
                            confidence=0.95,
                            metadata={"selector": selector, "element_index": i}
                        ))

            return changes

        except Exception as e:
            self.logger.error(f"Error analyzing content changes: {e}")
            return []

    def _should_report_change(self, change: ChangeEvent, target: MonitoringTarget) -> bool:
        """Determine if a change should be reported based on filters."""
        try:
            # Check ignore patterns
            for pattern in target.ignore_patterns:
                if re.search(pattern, change.description, re.IGNORECASE):
                    return False

                if change.new_value and re.search(pattern, change.new_value, re.IGNORECASE):
                    return False

            # Check confidence threshold
            if change.confidence < target.threshold:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking change filters: {e}")
            return True

    async def _handle_changes_detected(self, target_id: str, changes: List[ChangeEvent]):
        """Handle detected changes (trigger alerts, re-scraping, etc.)."""
        try:
            target = self.monitoring_targets[target_id]

            # Log significant changes
            significant_changes = [c for c in changes if c.confidence > 0.8]
            if significant_changes:
                self.logger.info(
                    f"Significant changes detected for {target.name}: "
                    f"{len(significant_changes)} changes"
                )

            # Here you could trigger:
            # - Re-scraping jobs
            # - Notifications/alerts
            # - Webhook calls
            # - Real-time updates to connected clients

            # Example: Trigger re-scraping for significant changes
            for change in significant_changes:
                if change.change_type in [ChangeType.CONTENT_MODIFIED, ChangeType.NEW_ELEMENTS]:
                    # Could trigger a re-scraping job here
                    self.logger.info(f"Would trigger re-scraping for {target_id} due to {change.change_type.value}")

        except Exception as e:
            self.logger.error(f"Error handling changes for {target_id}: {e}")

    async def _metrics_collector(self):
        """Background task to collect metrics."""
        while self.is_running:
            try:
                # Update metrics
                self.metrics["monitoring_targets"] = len(self.monitoring_targets)

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                self.logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(300)

    async def _cleanup_old_data(self):
        """Background task to clean up old data."""
        while self.is_running:
            try:
                # Clean up old change events (keep last 30 days)
                cutoff_time = datetime.now() - timedelta(days=30)

                old_change_ids = [
                    change_id for change_id, event in self.change_events.items()
                    if event.timestamp < cutoff_time
                ]

                for change_id in old_change_ids:
                    del self.change_events[change_id]

                # Clean up change history in targets
                for target in self.monitoring_targets.values():
                    target.change_history = [
                        change_id for change_id in target.change_history
                        if change_id in self.change_events
                    ]

                self.logger.info(f"Cleaned up {len(old_change_ids)} old change events")

                await asyncio.sleep(3600)  # Clean up every hour

            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)


# Global change detection engine instance
change_detection_engine = ChangeDetectionEngine()
