"""
Phase 4: GDPR Compliance and Data Retention System.
"""
import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import os

from agents.base import Agent
from agents.enhanced_security import PIIDetector, PIIDetectionResult
from models.task import Task, TaskType
from models.message import Message, TaskMessage, ResultMessage


class DataProcessingLawfulBasis(Enum):
    """GDPR lawful basis for processing personal data."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRights(Enum):
    """GDPR data subject rights."""
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    RESTRICT_PROCESSING = "restrict_processing"
    DATA_PORTABILITY = "data_portability"
    OBJECT = "object"
    AUTOMATED_DECISION_MAKING = "automated_decision_making"


@dataclass
class DataProcessingRecord:
    """Record of personal data processing activity."""
    record_id: str
    data_subject_id: Optional[str]
    data_categories: List[str]
    processing_purposes: List[str]
    lawful_basis: DataProcessingLawfulBasis
    consent_timestamp: Optional[float]
    retention_period: int  # days
    created_at: float
    last_updated: float
    is_active: bool = True
    
    
@dataclass
class DataSubjectRequest:
    """Data subject request under GDPR."""
    request_id: str
    data_subject_id: str
    request_type: DataSubjectRights
    request_details: str
    submitted_at: float
    status: str  # pending, in_progress, completed, rejected
    response_due_date: float
    completed_at: Optional[float] = None
    response_details: Optional[str] = None


@dataclass
class ConsentRecord:
    """Record of data subject consent."""
    consent_id: str
    data_subject_id: str
    consent_text: str
    consent_version: str
    granted_at: float
    withdrawn_at: Optional[float] = None
    is_active: bool = True
    processing_purposes: List[str] = field(default_factory=list)
    data_categories: List[str] = field(default_factory=list)


class GDPRComplianceManager:
    """Comprehensive GDPR compliance management system."""
    
    def __init__(self):
        self.logger = logging.getLogger("gdpr_compliance")
        self.db_path = "data/gdpr_compliance.db"
        self.pii_detector = PIIDetector()
        
        # Initialize database
        self._initialize_database()
        
        # Compliance configuration
        self.data_retention_policies = {
            "scraping_data": 90,      # days
            "user_data": 365,         # days
            "audit_logs": 2555,       # 7 years
            "consent_records": 2555,  # 7 years
            "marketing_data": 1095,   # 3 years
        }
        
        # Response timeframes (in days)
        self.response_timeframes = {
            DataSubjectRights.ACCESS: 30,
            DataSubjectRights.RECTIFICATION: 30,
            DataSubjectRights.ERASURE: 30,
            DataSubjectRights.RESTRICT_PROCESSING: 30,
            DataSubjectRights.DATA_PORTABILITY: 30,
            DataSubjectRights.OBJECT: 30,
            DataSubjectRights.AUTOMATED_DECISION_MAKING: 30,
        }
        
    def _initialize_database(self):
        """Initialize GDPR compliance database."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Processing records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processing_records (
                        record_id TEXT PRIMARY KEY,
                        data_subject_id TEXT,
                        data_categories TEXT,
                        processing_purposes TEXT,
                        lawful_basis TEXT,
                        consent_timestamp REAL,
                        retention_period INTEGER,
                        created_at REAL,
                        last_updated REAL,
                        is_active BOOLEAN
                    )
                ''')
                
                # Data subject requests table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS data_subject_requests (
                        request_id TEXT PRIMARY KEY,
                        data_subject_id TEXT,
                        request_type TEXT,
                        request_details TEXT,
                        submitted_at REAL,
                        status TEXT,
                        response_due_date REAL,
                        completed_at REAL,
                        response_details TEXT
                    )
                ''')
                
                # Consent records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS consent_records (
                        consent_id TEXT PRIMARY KEY,
                        data_subject_id TEXT,
                        consent_text TEXT,
                        consent_version TEXT,
                        granted_at REAL,
                        withdrawn_at REAL,
                        is_active BOOLEAN,
                        processing_purposes TEXT,
                        data_categories TEXT
                    )
                ''')
                
                # Data retention schedule table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS retention_schedule (
                        schedule_id TEXT PRIMARY KEY,
                        data_type TEXT,
                        data_location TEXT,
                        retention_period INTEGER,
                        deletion_date REAL,
                        status TEXT,
                        created_at REAL
                    )
                ''')
                
                conn.commit()
                self.logger.info("GDPR compliance database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize GDPR database: {e}")
            raise
            
    def record_data_processing(self, 
                             data_subject_id: Optional[str],
                             data_categories: List[str],
                             processing_purposes: List[str],
                             lawful_basis: DataProcessingLawfulBasis,
                             consent_timestamp: Optional[float] = None,
                             retention_period: Optional[int] = None) -> str:
        """Record a data processing activity."""
        try:
            record_id = hashlib.sha256(
                f"{data_subject_id}_{time.time()}_{hash(tuple(data_categories))}".encode()
            ).hexdigest()[:16]
            
            # Determine retention period
            if not retention_period:
                retention_period = self._determine_retention_period(data_categories, processing_purposes)
                
            record = DataProcessingRecord(
                record_id=record_id,
                data_subject_id=data_subject_id,
                data_categories=data_categories,
                processing_purposes=processing_purposes,
                lawful_basis=lawful_basis,
                consent_timestamp=consent_timestamp,
                retention_period=retention_period,
                created_at=time.time(),
                last_updated=time.time()
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO processing_records 
                    (record_id, data_subject_id, data_categories, processing_purposes,
                     lawful_basis, consent_timestamp, retention_period, created_at,
                     last_updated, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.record_id,
                    record.data_subject_id,
                    json.dumps(record.data_categories),
                    json.dumps(record.processing_purposes),
                    record.lawful_basis.value,
                    record.consent_timestamp,
                    record.retention_period,
                    record.created_at,
                    record.last_updated,
                    record.is_active
                ))
                conn.commit()
                
            self.logger.info(f"Recorded data processing activity: {record_id}")
            return record_id
            
        except Exception as e:
            self.logger.error(f"Failed to record data processing: {e}")
            raise
            
    def _determine_retention_period(self, 
                                  data_categories: List[str], 
                                  processing_purposes: List[str]) -> int:
        """Determine appropriate retention period based on data and purpose."""
        max_retention = 90  # Default
        
        # Check against retention policies
        for category in data_categories:
            if category in self.data_retention_policies:
                max_retention = max(max_retention, self.data_retention_policies[category])
                
        for purpose in processing_purposes:
            if purpose in self.data_retention_policies:
                max_retention = max(max_retention, self.data_retention_policies[purpose])
                
        return max_retention
        
    def submit_data_subject_request(self,
                                  data_subject_id: str,
                                  request_type: DataSubjectRights,
                                  request_details: str) -> str:
        """Submit a data subject request."""
        try:
            request_id = hashlib.sha256(
                f"{data_subject_id}_{request_type.value}_{time.time()}".encode()
            ).hexdigest()[:16]
            
            submitted_at = time.time()
            response_due_date = submitted_at + (self.response_timeframes[request_type] * 86400)
            
            request = DataSubjectRequest(
                request_id=request_id,
                data_subject_id=data_subject_id,
                request_type=request_type,
                request_details=request_details,
                submitted_at=submitted_at,
                status="pending",
                response_due_date=response_due_date
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO data_subject_requests
                    (request_id, data_subject_id, request_type, request_details,
                     submitted_at, status, response_due_date, completed_at, response_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    request.request_id,
                    request.data_subject_id,
                    request.request_type.value,
                    request.request_details,
                    request.submitted_at,
                    request.status,
                    request.response_due_date,
                    request.completed_at,
                    request.response_details
                ))
                conn.commit()
                
            self.logger.info(f"Submitted data subject request: {request_id}")
            return request_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit data subject request: {e}")
            raise
            
    def process_erasure_request(self, request_id: str) -> Dict[str, Any]:
        """Process a data erasure (right to be forgotten) request."""
        try:
            # Get request details
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM data_subject_requests WHERE request_id = ?
                ''', (request_id,))
                row = cursor.fetchone()
                
                if not row:
                    raise ValueError(f"Request not found: {request_id}")
                    
                data_subject_id = row[1]
                
            # Find all processing records for this data subject
            processing_records = self._get_processing_records_for_subject(data_subject_id)
            
            # Check if erasure is legally possible
            erasure_blocks = self._check_erasure_blocks(processing_records)
            
            if erasure_blocks:
                return {
                    "status": "rejected",
                    "reason": "Legal obligations prevent erasure",
                    "details": erasure_blocks
                }
                
            # Perform erasure
            erasure_results = self._perform_data_erasure(data_subject_id)
            
            # Update request status
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE data_subject_requests 
                    SET status = ?, completed_at = ?, response_details = ?
                    WHERE request_id = ?
                ''', (
                    "completed",
                    time.time(),
                    json.dumps(erasure_results),
                    request_id
                ))
                conn.commit()
                
            return {
                "status": "completed",
                "erasure_results": erasure_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process erasure request: {e}")
            raise
            
    def _get_processing_records_for_subject(self, data_subject_id: str) -> List[Dict[str, Any]]:
        """Get all processing records for a data subject."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM processing_records 
                WHERE data_subject_id = ? AND is_active = 1
            ''', (data_subject_id,))
            
            records = []
            for row in cursor.fetchall():
                records.append({
                    "record_id": row[0],
                    "data_categories": json.loads(row[2]),
                    "processing_purposes": json.loads(row[3]),
                    "lawful_basis": row[4],
                    "retention_period": row[6]
                })
                
            return records
            
    def _check_erasure_blocks(self, processing_records: List[Dict[str, Any]]) -> List[str]:
        """Check if there are legal blocks to data erasure."""
        blocks = []
        
        for record in processing_records:
            lawful_basis = record["lawful_basis"]
            
            # Legal obligation blocks erasure
            if lawful_basis == DataProcessingLawfulBasis.LEGAL_OBLIGATION.value:
                blocks.append(f"Legal obligation: {record['processing_purposes']}")
                
            # Public task may block erasure
            if lawful_basis == DataProcessingLawfulBasis.PUBLIC_TASK.value:
                blocks.append(f"Public task: {record['processing_purposes']}")
                
            # Check for audit/compliance requirements
            if any(purpose in ["audit", "compliance", "legal"] 
                   for purpose in record["processing_purposes"]):
                blocks.append(f"Compliance requirement: {record['processing_purposes']}")
                
        return blocks
        
    def _perform_data_erasure(self, data_subject_id: str) -> Dict[str, Any]:
        """Perform actual data erasure for a data subject."""
        results = {
            "databases_updated": 0,
            "files_deleted": 0,
            "records_anonymized": 0,
            "errors": []
        }
        
        try:
            # Mark processing records as inactive
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE processing_records 
                    SET is_active = 0, last_updated = ?
                    WHERE data_subject_id = ?
                ''', (time.time(), data_subject_id))
                results["databases_updated"] += cursor.rowcount
                conn.commit()
                
            # TODO: Implement actual data deletion from various systems
            # This would include:
            # - Deleting from main application database
            # - Removing from file storage
            # - Anonymizing in analytics systems
            # - Notifying third-party processors
            
            self.logger.info(f"Completed data erasure for subject: {data_subject_id}")
            
        except Exception as e:
            results["errors"].append(str(e))
            self.logger.error(f"Error during data erasure: {e}")
            
        return results
        
    def generate_compliance_report(self, start_date: float, end_date: float) -> Dict[str, Any]:
        """Generate GDPR compliance report for a date range."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Processing activities summary
                cursor.execute('''
                    SELECT lawful_basis, COUNT(*) 
                    FROM processing_records 
                    WHERE created_at BETWEEN ? AND ?
                    GROUP BY lawful_basis
                ''', (start_date, end_date))
                processing_by_basis = dict(cursor.fetchall())
                
                # Data subject requests summary
                cursor.execute('''
                    SELECT request_type, status, COUNT(*)
                    FROM data_subject_requests
                    WHERE submitted_at BETWEEN ? AND ?
                    GROUP BY request_type, status
                ''', (start_date, end_date))
                requests_summary = {}
                for request_type, status, count in cursor.fetchall():
                    if request_type not in requests_summary:
                        requests_summary[request_type] = {}
                    requests_summary[request_type][status] = count
                    
                # Overdue requests
                cursor.execute('''
                    SELECT COUNT(*) FROM data_subject_requests
                    WHERE response_due_date < ? AND status != 'completed'
                ''', (time.time(),))
                overdue_requests = cursor.fetchone()[0]
                
                # Consent withdrawals
                cursor.execute('''
                    SELECT COUNT(*) FROM consent_records
                    WHERE withdrawn_at BETWEEN ? AND ?
                ''', (start_date, end_date))
                consent_withdrawals = cursor.fetchone()[0]
                
            report = {
                "report_period": {
                    "start_date": datetime.fromtimestamp(start_date).isoformat(),
                    "end_date": datetime.fromtimestamp(end_date).isoformat()
                },
                "processing_activities": {
                    "by_lawful_basis": processing_by_basis,
                    "total_activities": sum(processing_by_basis.values())
                },
                "data_subject_requests": {
                    "by_type_and_status": requests_summary,
                    "overdue_requests": overdue_requests
                },
                "consent_management": {
                    "withdrawals": consent_withdrawals
                },
                "compliance_score": self._calculate_compliance_score(
                    overdue_requests, len(requests_summary)
                )
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            raise
            
    def _calculate_compliance_score(self, overdue_requests: int, total_requests: int) -> float:
        """Calculate a compliance score based on request handling."""
        if total_requests == 0:
            return 1.0
            
        on_time_rate = max(0, (total_requests - overdue_requests) / total_requests)
        return round(on_time_rate, 2)
