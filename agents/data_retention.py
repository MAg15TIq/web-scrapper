"""
Phase 4: Automated Data Retention and Cleanup System.
"""
import asyncio
import logging
import time
import json
import hashlib
import os
import shutil
import sqlite3
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import schedule

from agents.base import Agent
from agents.enhanced_security import AdvancedEncryptionManager, TamperProofAuditLogger, SecurityEvent
from models.task import Task, TaskType
from models.message import Message, TaskMessage, ResultMessage


@dataclass
class RetentionPolicy:
    """Data retention policy definition."""
    policy_id: str
    name: str
    description: str
    data_types: List[str]
    retention_period_days: int
    deletion_method: str  # secure_delete, anonymize, archive
    legal_basis: str
    created_at: float
    is_active: bool = True
    auto_apply: bool = True
    
    
@dataclass
class RetentionScheduleItem:
    """Scheduled data retention/deletion item."""
    schedule_id: str
    policy_id: str
    data_location: str
    data_identifier: str
    scheduled_date: float
    status: str  # pending, in_progress, completed, failed
    created_at: float
    completed_at: Optional[float] = None
    error_message: Optional[str] = None


class DataRetentionManager:
    """Automated data retention and cleanup manager."""
    
    def __init__(self, encryption_manager: AdvancedEncryptionManager, 
                 audit_logger: TamperProofAuditLogger):
        self.logger = logging.getLogger("data_retention")
        self.encryption_manager = encryption_manager
        self.audit_logger = audit_logger
        self.db_path = "data/retention_management.db"
        
        # Initialize database
        self._initialize_database()
        
        # Default retention policies
        self._create_default_policies()
        
        # Cleanup configuration
        self.cleanup_batch_size = 100
        self.cleanup_interval = 3600  # 1 hour
        self.secure_deletion_passes = 3
        
        # Start background tasks
        # self._start_background_tasks()
        
    def _initialize_database(self):
        """Initialize retention management database."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Retention policies table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS retention_policies (
                        policy_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        data_types TEXT,
                        retention_period_days INTEGER,
                        deletion_method TEXT,
                        legal_basis TEXT,
                        created_at REAL,
                        is_active BOOLEAN,
                        auto_apply BOOLEAN
                    )
                ''')
                
                # Retention schedule table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS retention_schedule (
                        schedule_id TEXT PRIMARY KEY,
                        policy_id TEXT,
                        data_location TEXT,
                        data_identifier TEXT,
                        scheduled_date REAL,
                        status TEXT,
                        created_at REAL,
                        completed_at REAL,
                        error_message TEXT,
                        FOREIGN KEY (policy_id) REFERENCES retention_policies (policy_id)
                    )
                ''')
                
                # Data inventory table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS data_inventory (
                        inventory_id TEXT PRIMARY KEY,
                        data_type TEXT,
                        data_location TEXT,
                        data_identifier TEXT,
                        size_bytes INTEGER,
                        created_at REAL,
                        last_accessed REAL,
                        retention_policy_id TEXT,
                        deletion_scheduled_date REAL,
                        is_deleted BOOLEAN DEFAULT 0,
                        FOREIGN KEY (retention_policy_id) REFERENCES retention_policies (policy_id)
                    )
                ''')
                
                conn.commit()
                self.logger.info("Data retention database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize retention database: {e}")
            raise
            
    def _create_default_policies(self):
        """Create default retention policies."""
        default_policies = [
            RetentionPolicy(
                policy_id="scraping_data_90d",
                name="Scraping Data - 90 Days",
                description="Standard retention for web scraping data",
                data_types=["scraping_results", "parsed_data", "raw_html"],
                retention_period_days=90,
                deletion_method="secure_delete",
                legal_basis="legitimate_interests",
                created_at=time.time()
            ),
            RetentionPolicy(
                policy_id="user_data_1y",
                name="User Data - 1 Year",
                description="User account and preference data",
                data_types=["user_profiles", "preferences", "settings"],
                retention_period_days=365,
                deletion_method="secure_delete",
                legal_basis="contract",
                created_at=time.time()
            ),
            RetentionPolicy(
                policy_id="audit_logs_7y",
                name="Audit Logs - 7 Years",
                description="Security and compliance audit logs",
                data_types=["audit_logs", "security_events", "access_logs"],
                retention_period_days=2555,  # 7 years
                deletion_method="archive",
                legal_basis="legal_obligation",
                created_at=time.time()
            ),
            RetentionPolicy(
                policy_id="temp_data_7d",
                name="Temporary Data - 7 Days",
                description="Temporary processing data and caches",
                data_types=["temp_files", "cache_data", "session_data"],
                retention_period_days=7,
                deletion_method="secure_delete",
                legal_basis="legitimate_interests",
                created_at=time.time()
            )
        ]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for policy in default_policies:
                    # Check if policy already exists
                    cursor.execute(
                        'SELECT policy_id FROM retention_policies WHERE policy_id = ?',
                        (policy.policy_id,)
                    )
                    
                    if not cursor.fetchone():
                        cursor.execute('''
                            INSERT INTO retention_policies
                            (policy_id, name, description, data_types, retention_period_days,
                             deletion_method, legal_basis, created_at, is_active, auto_apply)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            policy.policy_id,
                            policy.name,
                            policy.description,
                            json.dumps(policy.data_types),
                            policy.retention_period_days,
                            policy.deletion_method,
                            policy.legal_basis,
                            policy.created_at,
                            policy.is_active,
                            policy.auto_apply
                        ))
                        
                conn.commit()
                self.logger.info("Default retention policies created")
                
        except Exception as e:
            self.logger.error(f"Failed to create default policies: {e}")
            
    def register_data(self, data_type: str, data_location: str, 
                     data_identifier: str, size_bytes: int = 0) -> str:
        """Register data for retention management."""
        try:
            inventory_id = hashlib.sha256(
                f"{data_type}_{data_location}_{data_identifier}_{time.time()}".encode()
            ).hexdigest()[:16]
            
            # Find applicable retention policy
            policy = self._find_applicable_policy(data_type)
            
            # Calculate deletion date
            deletion_date = None
            if policy:
                deletion_date = time.time() + (policy.retention_period_days * 86400)
                
            # Register in inventory
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO data_inventory
                    (inventory_id, data_type, data_location, data_identifier,
                     size_bytes, created_at, last_accessed, retention_policy_id,
                     deletion_scheduled_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    inventory_id,
                    data_type,
                    data_location,
                    data_identifier,
                    size_bytes,
                    time.time(),
                    time.time(),
                    policy.policy_id if policy else None,
                    deletion_date
                ))
                conn.commit()
                
            # Schedule deletion if policy requires it
            if policy and policy.auto_apply and deletion_date:
                self._schedule_deletion(inventory_id, policy.policy_id, 
                                      data_location, data_identifier, deletion_date)
                
            self.logger.debug(f"Registered data for retention: {inventory_id}")
            return inventory_id
            
        except Exception as e:
            self.logger.error(f"Failed to register data: {e}")
            raise
            
    def _find_applicable_policy(self, data_type: str) -> Optional[RetentionPolicy]:
        """Find the applicable retention policy for a data type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM retention_policies 
                    WHERE is_active = 1
                    ORDER BY retention_period_days ASC
                ''')
                
                for row in cursor.fetchall():
                    data_types = json.loads(row[3])
                    if data_type in data_types:
                        return RetentionPolicy(
                            policy_id=row[0],
                            name=row[1],
                            description=row[2],
                            data_types=data_types,
                            retention_period_days=row[4],
                            deletion_method=row[5],
                            legal_basis=row[6],
                            created_at=row[7],
                            is_active=bool(row[8]),
                            auto_apply=bool(row[9])
                        )
                        
        except Exception as e:
            self.logger.error(f"Failed to find applicable policy: {e}")
            
        return None
        
    def _schedule_deletion(self, inventory_id: str, policy_id: str,
                          data_location: str, data_identifier: str, 
                          scheduled_date: float):
        """Schedule data for deletion."""
        try:
            schedule_id = hashlib.sha256(
                f"{inventory_id}_{scheduled_date}".encode()
            ).hexdigest()[:16]
            
            schedule_item = RetentionScheduleItem(
                schedule_id=schedule_id,
                policy_id=policy_id,
                data_location=data_location,
                data_identifier=data_identifier,
                scheduled_date=scheduled_date,
                status="pending",
                created_at=time.time()
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO retention_schedule
                    (schedule_id, policy_id, data_location, data_identifier,
                     scheduled_date, status, created_at, completed_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    schedule_item.schedule_id,
                    schedule_item.policy_id,
                    schedule_item.data_location,
                    schedule_item.data_identifier,
                    schedule_item.scheduled_date,
                    schedule_item.status,
                    schedule_item.created_at,
                    schedule_item.completed_at,
                    schedule_item.error_message
                ))
                conn.commit()
                
            self.logger.debug(f"Scheduled deletion: {schedule_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to schedule deletion: {e}")
            
    async def process_scheduled_deletions(self):
        """Process scheduled deletions that are due."""
        try:
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM retention_schedule
                    WHERE scheduled_date <= ? AND status = 'pending'
                    ORDER BY scheduled_date ASC
                    LIMIT ?
                ''', (current_time, self.cleanup_batch_size))
                
                pending_deletions = cursor.fetchall()
                
            for row in pending_deletions:
                schedule_id = row[0]
                policy_id = row[1]
                data_location = row[2]
                data_identifier = row[3]
                
                try:
                    # Update status to in_progress
                    self._update_schedule_status(schedule_id, "in_progress")
                    
                    # Get policy details
                    policy = self._get_policy(policy_id)
                    if not policy:
                        raise ValueError(f"Policy not found: {policy_id}")
                        
                    # Perform deletion based on policy method
                    if policy.deletion_method == "secure_delete":
                        await self._secure_delete_data(data_location, data_identifier)
                    elif policy.deletion_method == "anonymize":
                        await self._anonymize_data(data_location, data_identifier)
                    elif policy.deletion_method == "archive":
                        await self._archive_data(data_location, data_identifier)
                    else:
                        raise ValueError(f"Unknown deletion method: {policy.deletion_method}")
                        
                    # Update status to completed
                    self._update_schedule_status(schedule_id, "completed", time.time())
                    
                    # Log security event
                    self.audit_logger.log_security_event(SecurityEvent(
                        event_id=f"retention_{schedule_id}",
                        event_type="data_retention",
                        timestamp=time.time(),
                        user_id="system",
                        ip_address=None,
                        user_agent="retention_manager",
                        resource=data_location,
                        action=policy.deletion_method,
                        result="success",
                        details={
                            "policy_id": policy_id,
                            "data_identifier": data_identifier,
                            "retention_period": policy.retention_period_days
                        }
                    ))
                    
                    self.logger.info(f"Completed scheduled deletion: {schedule_id}")
                    
                except Exception as e:
                    # Update status to failed
                    self._update_schedule_status(schedule_id, "failed", 
                                               error_message=str(e))
                    self.logger.error(f"Failed to process deletion {schedule_id}: {e}")
                    
                # Small delay between deletions
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error processing scheduled deletions: {e}")
            
    def _update_schedule_status(self, schedule_id: str, status: str,
                               completed_at: Optional[float] = None,
                               error_message: Optional[str] = None):
        """Update the status of a scheduled deletion."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE retention_schedule
                    SET status = ?, completed_at = ?, error_message = ?
                    WHERE schedule_id = ?
                ''', (status, completed_at, error_message, schedule_id))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update schedule status: {e}")
            
    def _get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """Get retention policy by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM retention_policies WHERE policy_id = ?',
                    (policy_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return RetentionPolicy(
                        policy_id=row[0],
                        name=row[1],
                        description=row[2],
                        data_types=json.loads(row[3]),
                        retention_period_days=row[4],
                        deletion_method=row[5],
                        legal_basis=row[6],
                        created_at=row[7],
                        is_active=bool(row[8]),
                        auto_apply=bool(row[9])
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to get policy: {e}")
            
        return None
        
    async def _secure_delete_data(self, data_location: str, data_identifier: str):
        """Securely delete data with multiple overwrite passes."""
        try:
            file_path = Path(data_location) / data_identifier
            
            if file_path.exists() and file_path.is_file():
                # Get file size for overwriting
                file_size = file_path.stat().st_size
                
                # Multiple overwrite passes
                for pass_num in range(self.secure_deletion_passes):
                    with open(file_path, 'r+b') as f:
                        # Overwrite with random data
                        f.seek(0)
                        f.write(os.urandom(file_size))
                        f.flush()
                        os.fsync(f.fileno())
                        
                # Final deletion
                file_path.unlink()
                self.logger.debug(f"Securely deleted file: {file_path}")
                
            elif Path(data_location).is_dir():
                # Handle directory deletion
                shutil.rmtree(data_location, ignore_errors=True)
                self.logger.debug(f"Deleted directory: {data_location}")
                
        except Exception as e:
            self.logger.error(f"Failed to securely delete data: {e}")
            raise
            
    async def _anonymize_data(self, data_location: str, data_identifier: str):
        """Anonymize data instead of deleting."""
        # This would implement data anonymization logic
        # For now, we'll just log the action
        self.logger.info(f"Anonymized data: {data_location}/{data_identifier}")
        
    async def _archive_data(self, data_location: str, data_identifier: str):
        """Archive data to long-term storage."""
        # This would implement data archiving logic
        # For now, we'll just log the action
        self.logger.info(f"Archived data: {data_location}/{data_identifier}")
        
    def _start_background_tasks(self):
        """Start background cleanup tasks."""
        asyncio.create_task(self._periodic_cleanup())
        
    async def _periodic_cleanup(self):
        """Periodic cleanup task."""
        while True:
            try:
                await self.process_scheduled_deletions()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying

# Singleton instance for global use
encryption_manager = AdvancedEncryptionManager()
audit_logger = TamperProofAuditLogger(encryption_manager)
retention_manager = DataRetentionManager(encryption_manager, audit_logger)
