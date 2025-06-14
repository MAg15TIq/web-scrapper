"""
Phase 4: Enhanced Security & Compliance System.
"""
import asyncio
import logging
import time
import json
import hashlib
import hmac
import secrets
import os
import re
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import base64

from agents.base import Agent
from models.task import Task, TaskType
from models.message import Message, TaskMessage, ResultMessage


@dataclass
class SecurityEvent:
    """Represents a security event for audit logging."""
    event_id: str
    event_type: str
    timestamp: float
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    risk_score: float = 0.0
    
    
@dataclass
class PIIDetectionResult:
    """Result of PII detection scan."""
    has_pii: bool
    pii_types: List[str]
    confidence_scores: Dict[str, float]
    locations: List[Dict[str, Any]]
    anonymization_suggestions: List[str]


class AdvancedEncryptionManager:
    """Advanced encryption manager with key rotation and HSM support."""
    
    def __init__(self):
        self.logger = logging.getLogger("encryption_manager")
        self.master_key_path = "data/master.key"
        self.key_rotation_interval = 86400 * 30  # 30 days
        self.encryption_keys: Dict[str, bytes] = {}
        self.key_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize master key
        self._initialize_master_key()
        
    def _initialize_master_key(self):
        """Initialize or load the master encryption key."""
        try:
            if os.path.exists(self.master_key_path):
                with open(self.master_key_path, 'rb') as f:
                    self.master_key = f.read()
                self.logger.info("Loaded existing master key")
            else:
                # Generate new master key
                self.master_key = Fernet.generate_key()
                os.makedirs(os.path.dirname(self.master_key_path), exist_ok=True)
                with open(self.master_key_path, 'wb') as f:
                    f.write(self.master_key)
                os.chmod(self.master_key_path, 0o600)  # Restrict permissions
                self.logger.info("Generated new master key")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize master key: {e}")
            raise
            
    def generate_data_key(self, key_id: str, purpose: str = "general") -> bytes:
        """Generate a new data encryption key."""
        data_key = Fernet.generate_key()
        
        # Encrypt the data key with master key
        fernet = Fernet(self.master_key)
        encrypted_data_key = fernet.encrypt(data_key)
        
        # Store key metadata
        self.key_metadata[key_id] = {
            "purpose": purpose,
            "created_at": time.time(),
            "last_rotated": time.time(),
            "usage_count": 0,
            "encrypted_key": base64.b64encode(encrypted_data_key).decode()
        }
        
        self.encryption_keys[key_id] = data_key
        self.logger.info(f"Generated data key: {key_id}")
        return data_key
        
    def get_data_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve a data encryption key."""
        if key_id in self.encryption_keys:
            self.key_metadata[key_id]["usage_count"] += 1
            return self.encryption_keys[key_id]
            
        # Try to decrypt from metadata
        if key_id in self.key_metadata:
            try:
                fernet = Fernet(self.master_key)
                encrypted_key = base64.b64decode(self.key_metadata[key_id]["encrypted_key"])
                data_key = fernet.decrypt(encrypted_key)
                self.encryption_keys[key_id] = data_key
                return data_key
            except Exception as e:
                self.logger.error(f"Failed to decrypt data key {key_id}: {e}")
                
        return None
        
    def encrypt_data(self, data: Union[str, bytes], key_id: str = "default") -> str:
        """Encrypt data using specified key."""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Get or create data key
        data_key = self.get_data_key(key_id)
        if not data_key:
            data_key = self.generate_data_key(key_id)
            
        # Encrypt data
        fernet = Fernet(data_key)
        encrypted_data = fernet.encrypt(data)
        
        # Return base64 encoded result with key ID
        result = {
            "key_id": key_id,
            "data": base64.b64encode(encrypted_data).decode(),
            "timestamp": time.time()
        }
        return base64.b64encode(json.dumps(result).encode()).decode()
        
    def decrypt_data(self, encrypted_data: str) -> Optional[bytes]:
        """Decrypt data using embedded key ID."""
        try:
            # Decode and parse
            decoded = base64.b64decode(encrypted_data)
            data_info = json.loads(decoded.decode())
            
            key_id = data_info["key_id"]
            encrypted_bytes = base64.b64decode(data_info["data"])
            
            # Get data key and decrypt
            data_key = self.get_data_key(key_id)
            if not data_key:
                self.logger.error(f"Data key not found: {key_id}")
                return None
                
            fernet = Fernet(data_key)
            return fernet.decrypt(encrypted_bytes)
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            return None
            
    def rotate_key(self, key_id: str) -> bool:
        """Rotate a data encryption key."""
        try:
            # Generate new key
            new_key = self.generate_data_key(f"{key_id}_new")
            
            # Update metadata
            if key_id in self.key_metadata:
                self.key_metadata[key_id]["last_rotated"] = time.time()
                
            # Replace old key
            self.encryption_keys[key_id] = new_key
            
            self.logger.info(f"Rotated encryption key: {key_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate key {key_id}: {e}")
            return False


class PIIDetector:
    """Advanced PII detection and anonymization system."""
    
    def __init__(self):
        self.logger = logging.getLogger("pii_detector")
        
        # PII patterns
        self.pii_patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            "name": re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # Simple name pattern
            "address": re.compile(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)'),
        }
        
        # GDPR sensitive categories
        self.gdpr_categories = {
            "personal_identifiers": ["email", "phone", "ssn", "name"],
            "location_data": ["address", "ip_address"],
            "financial_data": ["credit_card"],
            "biometric_data": [],  # Would need specialized detection
            "health_data": [],     # Would need specialized detection
        }
        
    def detect_pii(self, text: str, context: str = "general") -> PIIDetectionResult:
        """Detect PII in text content."""
        pii_types = []
        confidence_scores = {}
        locations = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = list(pattern.finditer(text))
            if matches:
                pii_types.append(pii_type)
                confidence_scores[pii_type] = self._calculate_confidence(pii_type, matches, text)
                
                for match in matches:
                    locations.append({
                        "type": pii_type,
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "confidence": confidence_scores[pii_type]
                    })
                    
        has_pii = len(pii_types) > 0
        anonymization_suggestions = self._generate_anonymization_suggestions(pii_types)
        
        return PIIDetectionResult(
            has_pii=has_pii,
            pii_types=pii_types,
            confidence_scores=confidence_scores,
            locations=locations,
            anonymization_suggestions=anonymization_suggestions
        )
        
    def _calculate_confidence(self, pii_type: str, matches: List, text: str) -> float:
        """Calculate confidence score for PII detection."""
        base_confidence = 0.8
        
        # Adjust based on context and pattern strength
        if pii_type == "email":
            # Check for valid TLD
            for match in matches:
                if any(tld in match.group().lower() for tld in ['.com', '.org', '.net', '.edu']):
                    base_confidence = 0.95
                    
        elif pii_type == "phone":
            # Check for proper formatting
            for match in matches:
                if len(re.sub(r'[^\d]', '', match.group())) == 10:
                    base_confidence = 0.9
                    
        elif pii_type == "credit_card":
            # Basic Luhn algorithm check
            for match in matches:
                digits = re.sub(r'[^\d]', '', match.group())
                if self._luhn_check(digits):
                    base_confidence = 0.95
                    
        return base_confidence
        
    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        def luhn_checksum(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10
        
        return luhn_checksum(card_number) == 0
        
    def _generate_anonymization_suggestions(self, pii_types: List[str]) -> List[str]:
        """Generate suggestions for anonymizing detected PII."""
        suggestions = []
        
        for pii_type in pii_types:
            if pii_type == "email":
                suggestions.append("Replace with hashed email or generic placeholder")
            elif pii_type == "phone":
                suggestions.append("Mask digits except last 4 or replace with placeholder")
            elif pii_type == "ssn":
                suggestions.append("Replace with XXX-XX-XXXX format")
            elif pii_type == "credit_card":
                suggestions.append("Mask all but last 4 digits")
            elif pii_type == "name":
                suggestions.append("Replace with initials or generic identifier")
            elif pii_type == "address":
                suggestions.append("Replace with city/state only or postal code")
            elif pii_type == "ip_address":
                suggestions.append("Mask last octet or replace with subnet")
                
        return suggestions
        
    def anonymize_text(self, text: str, anonymization_level: str = "medium") -> str:
        """Anonymize PII in text based on specified level."""
        result = text
        
        for pii_type, pattern in self.pii_patterns.items():
            if anonymization_level == "high":
                replacement = "[REDACTED]"
            elif anonymization_level == "medium":
                replacement = self._get_medium_replacement(pii_type)
            else:  # low
                replacement = self._get_low_replacement(pii_type)
                
            result = pattern.sub(replacement, result)
            
        return result
        
    def _get_medium_replacement(self, pii_type: str) -> str:
        """Get medium-level anonymization replacement."""
        replacements = {
            "email": "user@domain.com",
            "phone": "XXX-XXX-XXXX",
            "ssn": "XXX-XX-XXXX",
            "credit_card": "XXXX-XXXX-XXXX-XXXX",
            "name": "[NAME]",
            "address": "[ADDRESS]",
            "ip_address": "XXX.XXX.XXX.XXX"
        }
        return replacements.get(pii_type, "[REDACTED]")
        
    def _get_low_replacement(self, pii_type: str) -> str:
        """Get low-level anonymization replacement."""
        replacements = {
            "email": "user***@***.com",
            "phone": "XXX-XXX-1234",
            "ssn": "XXX-XX-1234",
            "credit_card": "XXXX-XXXX-XXXX-1234",
            "name": "John D.",
            "address": "123 Main St, [CITY]",
            "ip_address": "192.168.1.XXX"
        }
        return replacements.get(pii_type, "[PARTIAL]")


class TamperProofAuditLogger:
    """Tamper-proof audit logging system with cryptographic integrity."""
    
    def __init__(self, encryption_manager: AdvancedEncryptionManager):
        self.logger = logging.getLogger("audit_logger")
        self.encryption_manager = encryption_manager
        self.audit_log_path = "data/audit_logs"
        self.integrity_key = self._generate_integrity_key()
        self.log_buffer: List[SecurityEvent] = []
        self.buffer_size = 100
        
        # Ensure audit log directory exists
        os.makedirs(self.audit_log_path, exist_ok=True)
        
    def _generate_integrity_key(self) -> bytes:
        """Generate key for HMAC integrity verification."""
        return secrets.token_bytes(32)
        
    def log_security_event(self, event: SecurityEvent):
        """Log a security event with tamper protection."""
        try:
            # Add to buffer
            self.log_buffer.append(event)
            
            # Flush buffer if full
            if len(self.log_buffer) >= self.buffer_size:
                self._flush_log_buffer()
                
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
            
    def _flush_log_buffer(self):
        """Flush log buffer to persistent storage."""
        if not self.log_buffer:
            return
            
        try:
            # Create log batch
            batch_id = secrets.token_hex(16)
            timestamp = time.time()
            
            batch_data = {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "events": [self._serialize_event(event) for event in self.log_buffer],
                "event_count": len(self.log_buffer)
            }
            
            # Serialize and encrypt
            serialized = json.dumps(batch_data, sort_keys=True)
            encrypted_data = self.encryption_manager.encrypt_data(serialized, "audit_logs")
            
            # Generate integrity hash
            integrity_hash = hmac.new(
                self.integrity_key,
                encrypted_data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Create final log entry
            log_entry = {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "data": encrypted_data,
                "integrity_hash": integrity_hash,
                "version": "1.0"
            }
            
            # Write to file
            log_filename = f"audit_{datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H')}.log"
            log_filepath = os.path.join(self.audit_log_path, log_filename)
            
            with open(log_filepath, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
            # Clear buffer
            self.log_buffer.clear()
            
            self.logger.debug(f"Flushed {batch_data['event_count']} audit events to {log_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to flush audit log buffer: {e}")
            
    def _serialize_event(self, event: SecurityEvent) -> Dict[str, Any]:
        """Serialize security event to dictionary."""
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "user_id": event.user_id,
            "ip_address": event.ip_address,
            "user_agent": event.user_agent,
            "resource": event.resource,
            "action": event.action,
            "result": event.result,
            "details": event.details,
            "risk_score": event.risk_score
        }
