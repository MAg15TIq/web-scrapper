"""
Advanced data validation utilities for Phase 5 data processing enhancements.
"""
import re
import json
import hashlib
import asyncio
import httpx
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse
import logging

# External validation libraries
try:
    import phonenumbers
    from phonenumbers import geocoder, carrier
    PHONENUMBERS_AVAILABLE = True
except ImportError:
    PHONENUMBERS_AVAILABLE = False

try:
    import email_validator
    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATOR_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataValidator:
    """Advanced data validation utilities."""
    
    def __init__(self):
        """Initialize the data validator."""
        # Validation patterns
        self.patterns = {
            "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            "phone_us": re.compile(r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'),
            "phone_international": re.compile(r'^\+?[1-9]\d{1,14}$'),
            "url": re.compile(r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'),
            "ip_v4": re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
            "ip_v6": re.compile(r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'),
            "ssn": re.compile(r'^\d{3}-\d{2}-\d{4}$'),
            "credit_card": re.compile(r'^(?:\d{4}[-\s]?){3}\d{4}$'),
            "postal_code_us": re.compile(r'^\d{5}(?:-\d{4})?$'),
            "postal_code_ca": re.compile(r'^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$'),
            "postal_code_uk": re.compile(r'^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$'),
            "date_iso": re.compile(r'^\d{4}-\d{2}-\d{2}$'),
            "date_us": re.compile(r'^\d{1,2}\/\d{1,2}\/\d{4}$'),
            "time_24h": re.compile(r'^([01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?$'),
            "time_12h": re.compile(r'^(1[0-2]|0?[1-9]):[0-5][0-9](?::[0-5][0-9])?\s?(AM|PM|am|pm)$'),
            "currency": re.compile(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$'),
            "percentage": re.compile(r'^\d+(?:\.\d+)?%$'),
            "coordinates": re.compile(r'^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$')
        }
        
        # Country-specific validation patterns
        self.country_patterns = {
            "US": {
                "phone": self.patterns["phone_us"],
                "postal_code": self.patterns["postal_code_us"]
            },
            "CA": {
                "postal_code": self.patterns["postal_code_ca"]
            },
            "UK": {
                "postal_code": self.patterns["postal_code_uk"]
            }
        }
        
        # Data type validators
        self.type_validators = {
            "string": self._validate_string,
            "integer": self._validate_integer,
            "float": self._validate_float,
            "boolean": self._validate_boolean,
            "date": self._validate_date,
            "datetime": self._validate_datetime,
            "email": self._validate_email,
            "phone": self._validate_phone,
            "url": self._validate_url,
            "ip_address": self._validate_ip_address,
            "coordinates": self._validate_coordinates
        }
        
        # External validation cache
        self.validation_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 3600  # 1 hour
        
        # HTTP client for external validation
        self.http_client = None
        
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=10.0)
        return self.http_client
    
    def _get_cache_key(self, data_type: str, value: str, method: str = "basic") -> str:
        """Generate cache key for validation results."""
        content = f"{data_type}:{value}:{method}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get validation result from cache if available and not expired."""
        if cache_key in self.validation_cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                return self.validation_cache[cache_key]
            else:
                # Remove expired entry
                self.validation_cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
        return None
    
    def _store_in_cache(self, cache_key: str, result: Dict[str, Any]):
        """Store validation result in cache."""
        self.validation_cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now().timestamp()
    
    def validate_data_type(self, value: Any, data_type: str, **kwargs) -> Dict[str, Any]:
        """
        Validate data against specified type.
        
        Args:
            value: Value to validate
            data_type: Expected data type
            **kwargs: Additional validation parameters
            
        Returns:
            Validation result dictionary
        """
        try:
            if data_type in self.type_validators:
                return self.type_validators[data_type](value, **kwargs)
            else:
                return {
                    "is_valid": False,
                    "error": f"Unknown data type: {data_type}",
                    "confidence": 0.0
                }
        except Exception as e:
            logger.error(f"Error validating data type {data_type}: {e}")
            return {
                "is_valid": False,
                "error": str(e),
                "confidence": 0.0
            }
    
    def _validate_string(self, value: Any, min_length: int = 0, max_length: int = None, 
                        pattern: str = None, **kwargs) -> Dict[str, Any]:
        """Validate string value."""
        if not isinstance(value, str):
            return {"is_valid": False, "error": "Value is not a string", "confidence": 0.0}
        
        # Check length constraints
        if len(value) < min_length:
            return {"is_valid": False, "error": f"String too short (min: {min_length})", "confidence": 0.0}
        
        if max_length and len(value) > max_length:
            return {"is_valid": False, "error": f"String too long (max: {max_length})", "confidence": 0.0}
        
        # Check pattern if provided
        if pattern:
            if not re.match(pattern, value):
                return {"is_valid": False, "error": "String doesn't match required pattern", "confidence": 0.0}
        
        return {"is_valid": True, "confidence": 1.0, "length": len(value)}
    
    def _validate_integer(self, value: Any, min_value: int = None, max_value: int = None, **kwargs) -> Dict[str, Any]:
        """Validate integer value."""
        try:
            if isinstance(value, str):
                int_value = int(value)
            elif isinstance(value, (int, float)):
                int_value = int(value)
                if isinstance(value, float) and value != int_value:
                    return {"is_valid": False, "error": "Value is not an integer", "confidence": 0.0}
            else:
                return {"is_valid": False, "error": "Value cannot be converted to integer", "confidence": 0.0}
            
            # Check range constraints
            if min_value is not None and int_value < min_value:
                return {"is_valid": False, "error": f"Value too small (min: {min_value})", "confidence": 0.0}
            
            if max_value is not None and int_value > max_value:
                return {"is_valid": False, "error": f"Value too large (max: {max_value})", "confidence": 0.0}
            
            return {"is_valid": True, "confidence": 1.0, "value": int_value}
            
        except ValueError:
            return {"is_valid": False, "error": "Invalid integer format", "confidence": 0.0}
    
    def _validate_float(self, value: Any, min_value: float = None, max_value: float = None, **kwargs) -> Dict[str, Any]:
        """Validate float value."""
        try:
            if isinstance(value, str):
                float_value = float(value)
            elif isinstance(value, (int, float)):
                float_value = float(value)
            else:
                return {"is_valid": False, "error": "Value cannot be converted to float", "confidence": 0.0}
            
            # Check for special values
            if not (float_value == float_value):  # NaN check
                return {"is_valid": False, "error": "Value is NaN", "confidence": 0.0}
            
            if float_value == float('inf') or float_value == float('-inf'):
                return {"is_valid": False, "error": "Value is infinite", "confidence": 0.0}
            
            # Check range constraints
            if min_value is not None and float_value < min_value:
                return {"is_valid": False, "error": f"Value too small (min: {min_value})", "confidence": 0.0}
            
            if max_value is not None and float_value > max_value:
                return {"is_valid": False, "error": f"Value too large (max: {max_value})", "confidence": 0.0}
            
            return {"is_valid": True, "confidence": 1.0, "value": float_value}
            
        except ValueError:
            return {"is_valid": False, "error": "Invalid float format", "confidence": 0.0}
    
    def _validate_boolean(self, value: Any, **kwargs) -> Dict[str, Any]:
        """Validate boolean value."""
        if isinstance(value, bool):
            return {"is_valid": True, "confidence": 1.0, "value": value}
        
        if isinstance(value, str):
            lower_value = value.lower().strip()
            if lower_value in ["true", "yes", "1", "on", "enabled"]:
                return {"is_valid": True, "confidence": 0.9, "value": True}
            elif lower_value in ["false", "no", "0", "off", "disabled"]:
                return {"is_valid": True, "confidence": 0.9, "value": False}
        
        if isinstance(value, (int, float)):
            if value == 1:
                return {"is_valid": True, "confidence": 0.8, "value": True}
            elif value == 0:
                return {"is_valid": True, "confidence": 0.8, "value": False}
        
        return {"is_valid": False, "error": "Value cannot be converted to boolean", "confidence": 0.0}
    
    def _validate_date(self, value: Any, format: str = None, **kwargs) -> Dict[str, Any]:
        """Validate date value."""
        if isinstance(value, datetime):
            return {"is_valid": True, "confidence": 1.0, "value": value.date()}
        
        if not isinstance(value, str):
            return {"is_valid": False, "error": "Date value must be a string", "confidence": 0.0}
        
        # Try different date formats
        date_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%m-%d-%Y",
            "%d-%m-%Y"
        ]
        
        if format:
            date_formats.insert(0, format)
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(value, fmt).date()
                confidence = 1.0 if fmt == format else 0.8
                return {"is_valid": True, "confidence": confidence, "value": parsed_date, "format": fmt}
            except ValueError:
                continue
        
        return {"is_valid": False, "error": "Invalid date format", "confidence": 0.0}
    
    def _validate_datetime(self, value: Any, format: str = None, **kwargs) -> Dict[str, Any]:
        """Validate datetime value."""
        if isinstance(value, datetime):
            return {"is_valid": True, "confidence": 1.0, "value": value}
        
        if not isinstance(value, str):
            return {"is_valid": False, "error": "Datetime value must be a string", "confidence": 0.0}
        
        # Try different datetime formats
        datetime_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M"
        ]
        
        if format:
            datetime_formats.insert(0, format)
        
        for fmt in datetime_formats:
            try:
                parsed_datetime = datetime.strptime(value, fmt)
                confidence = 1.0 if fmt == format else 0.8
                return {"is_valid": True, "confidence": confidence, "value": parsed_datetime, "format": fmt}
            except ValueError:
                continue
        
        return {"is_valid": False, "error": "Invalid datetime format", "confidence": 0.0}
    
    def _validate_email(self, value: Any, check_mx: bool = False, **kwargs) -> Dict[str, Any]:
        """Validate email address."""
        if not isinstance(value, str):
            return {"is_valid": False, "error": "Email must be a string", "confidence": 0.0}

        # Basic regex validation
        if not self.patterns["email"].match(value):
            return {"is_valid": False, "error": "Invalid email format", "confidence": 0.0}

        # Advanced validation with email-validator library
        if EMAIL_VALIDATOR_AVAILABLE:
            try:
                validated_email = email_validator.validate_email(value, check_deliverability=check_mx)
                return {
                    "is_valid": True,
                    "confidence": 0.95,
                    "normalized": validated_email.email,
                    "local": validated_email.local,
                    "domain": validated_email.domain
                }
            except Exception as e:
                # Fall back to basic validation if email-validator fails
                pass

        # Basic validation passed
        parts = value.split('@')
        return {
            "is_valid": True,
            "confidence": 0.8,
            "local": parts[0],
            "domain": parts[1]
        }
    
    def _validate_phone(self, value: Any, country: str = None, **kwargs) -> Dict[str, Any]:
        """Validate phone number."""
        if not isinstance(value, str):
            return {"is_valid": False, "error": "Phone number must be a string", "confidence": 0.0}
        
        # Advanced validation with phonenumbers library
        if PHONENUMBERS_AVAILABLE:
            try:
                parsed_number = phonenumbers.parse(value, country)
                
                if phonenumbers.is_valid_number(parsed_number):
                    # Get additional information
                    formatted_number = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
                    country_code = phonenumbers.region_code_for_number(parsed_number)
                    location = geocoder.description_for_number(parsed_number, "en")
                    carrier_name = carrier.name_for_number(parsed_number, "en")
                    
                    return {
                        "is_valid": True,
                        "confidence": 0.95,
                        "formatted": formatted_number,
                        "country_code": country_code,
                        "location": location,
                        "carrier": carrier_name,
                        "type": "mobile" if phonenumbers.number_type(parsed_number) == phonenumbers.PhoneNumberType.MOBILE else "landline"
                    }
                else:
                    return {"is_valid": False, "error": "Invalid phone number", "confidence": 0.0}
                    
            except phonenumbers.NumberParseException as e:
                return {"is_valid": False, "error": str(e), "confidence": 0.0}
        
        # Basic regex validation
        if self.patterns["phone_us"].match(value) or self.patterns["phone_international"].match(value):
            return {"is_valid": True, "confidence": 0.7}
        
        return {"is_valid": False, "error": "Invalid phone number format", "confidence": 0.0}
    
    def _validate_url(self, value: Any, check_reachable: bool = False, **kwargs) -> Dict[str, Any]:
        """Validate URL."""
        if not isinstance(value, str):
            return {"is_valid": False, "error": "URL must be a string", "confidence": 0.0}
        
        # Basic regex validation
        if not self.patterns["url"].match(value):
            return {"is_valid": False, "error": "Invalid URL format", "confidence": 0.0}
        
        # Parse URL components
        try:
            parsed = urlparse(value)
            result = {
                "is_valid": True,
                "confidence": 0.8,
                "scheme": parsed.scheme,
                "domain": parsed.netloc,
                "path": parsed.path,
                "query": parsed.query,
                "fragment": parsed.fragment
            }
            
            # Check if URL is reachable (optional)
            if check_reachable and REQUESTS_AVAILABLE:
                try:
                    response = requests.head(value, timeout=5, allow_redirects=True)
                    result["reachable"] = response.status_code < 400
                    result["status_code"] = response.status_code
                    result["confidence"] = 0.95 if result["reachable"] else 0.6
                except requests.RequestException:
                    result["reachable"] = False
                    result["confidence"] = 0.6
            
            return result
            
        except Exception as e:
            return {"is_valid": False, "error": f"URL parsing error: {e}", "confidence": 0.0}
    
    def _validate_ip_address(self, value: Any, version: str = "any", **kwargs) -> Dict[str, Any]:
        """Validate IP address."""
        if not isinstance(value, str):
            return {"is_valid": False, "error": "IP address must be a string", "confidence": 0.0}
        
        is_ipv4 = self.patterns["ip_v4"].match(value)
        is_ipv6 = self.patterns["ip_v6"].match(value)
        
        if version == "4" and not is_ipv4:
            return {"is_valid": False, "error": "Invalid IPv4 address", "confidence": 0.0}
        elif version == "6" and not is_ipv6:
            return {"is_valid": False, "error": "Invalid IPv6 address", "confidence": 0.0}
        elif version == "any" and not (is_ipv4 or is_ipv6):
            return {"is_valid": False, "error": "Invalid IP address", "confidence": 0.0}
        
        ip_version = "4" if is_ipv4 else "6"
        
        # Additional validation for IPv4
        if is_ipv4:
            octets = value.split('.')
            for octet in octets:
                if int(octet) > 255:
                    return {"is_valid": False, "error": "Invalid IPv4 octet", "confidence": 0.0}
        
        return {
            "is_valid": True,
            "confidence": 1.0,
            "version": ip_version,
            "is_private": self._is_private_ip(value, ip_version)
        }
    
    def _validate_coordinates(self, value: Any, **kwargs) -> Dict[str, Any]:
        """Validate geographic coordinates."""
        if isinstance(value, str):
            if not self.patterns["coordinates"].match(value):
                return {"is_valid": False, "error": "Invalid coordinate format", "confidence": 0.0}
            
            try:
                lat_str, lng_str = value.split(',')
                lat = float(lat_str.strip())
                lng = float(lng_str.strip())
            except ValueError:
                return {"is_valid": False, "error": "Invalid coordinate values", "confidence": 0.0}
        
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                lat, lng = float(value[0]), float(value[1])
            except (ValueError, TypeError):
                return {"is_valid": False, "error": "Invalid coordinate values", "confidence": 0.0}
        
        else:
            return {"is_valid": False, "error": "Coordinates must be string or [lat, lng] array", "confidence": 0.0}
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90):
            return {"is_valid": False, "error": "Latitude out of range (-90 to 90)", "confidence": 0.0}
        
        if not (-180 <= lng <= 180):
            return {"is_valid": False, "error": "Longitude out of range (-180 to 180)", "confidence": 0.0}
        
        # Determine precision
        lat_decimals = len(str(lat).split('.')[-1]) if '.' in str(lat) else 0
        lng_decimals = len(str(lng).split('.')[-1]) if '.' in str(lng) else 0
        precision = min(lat_decimals, lng_decimals)
        
        precision_level = "low"
        if precision >= 6:
            precision_level = "high"  # ~1 meter
        elif precision >= 4:
            precision_level = "medium"  # ~10 meters
        elif precision >= 2:
            precision_level = "low"  # ~1 km
        
        return {
            "is_valid": True,
            "confidence": 1.0,
            "latitude": lat,
            "longitude": lng,
            "precision": precision_level,
            "decimal_places": precision
        }
    
    def _is_private_ip(self, ip: str, version: str) -> bool:
        """Check if IP address is private."""
        if version == "4":
            octets = [int(x) for x in ip.split('.')]
            # Private IPv4 ranges
            if octets[0] == 10:
                return True
            elif octets[0] == 172 and 16 <= octets[1] <= 31:
                return True
            elif octets[0] == 192 and octets[1] == 168:
                return True
            elif octets[0] == 127:  # Loopback
                return True
        
        # For IPv6, this would be more complex
        return False
    
    async def validate_external(self, value: str, validation_type: str, **kwargs) -> Dict[str, Any]:
        """
        Validate data against external services.
        
        Args:
            value: Value to validate
            validation_type: Type of external validation
            **kwargs: Additional parameters
            
        Returns:
            Validation result
        """
        cache_key = self._get_cache_key(validation_type, value, "external")
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            client = await self._get_http_client()
            
            if validation_type == "email_deliverability":
                # Use a hypothetical email validation service
                result = await self._validate_email_external(client, value)
            
            elif validation_type == "domain_exists":
                # Check if domain exists
                result = await self._validate_domain_external(client, value)
            
            elif validation_type == "url_reachable":
                # Check if URL is reachable
                result = await self._validate_url_external(client, value)
            
            else:
                result = {"is_valid": False, "error": f"Unknown external validation type: {validation_type}"}
            
            # Store in cache
            self._store_in_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in external validation: {e}")
            return {"is_valid": False, "error": str(e), "confidence": 0.0}
    
    async def _validate_email_external(self, client: httpx.AsyncClient, email: str) -> Dict[str, Any]:
        """Validate email using external service."""
        # This is a placeholder - in real implementation, you'd use a service like
        # Hunter.io, ZeroBounce, or similar
        try:
            # Simulate external API call
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Basic MX record check could be implemented here
            domain = email.split('@')[1]
            
            return {
                "is_valid": True,
                "confidence": 0.8,
                "deliverable": True,
                "domain": domain,
                "method": "external_api"
            }
            
        except Exception as e:
            return {"is_valid": False, "error": str(e), "confidence": 0.0}
    
    async def _validate_domain_external(self, client: httpx.AsyncClient, domain: str) -> Dict[str, Any]:
        """Validate domain existence."""
        try:
            response = await client.head(f"http://{domain}", timeout=5)
            return {
                "is_valid": True,
                "confidence": 0.9,
                "exists": True,
                "status_code": response.status_code
            }
        except Exception:
            return {
                "is_valid": False,
                "confidence": 0.7,
                "exists": False,
                "error": "Domain not reachable"
            }
    
    async def _validate_url_external(self, client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
        """Validate URL reachability."""
        try:
            response = await client.head(url, timeout=5)
            return {
                "is_valid": response.status_code < 400,
                "confidence": 0.9,
                "reachable": True,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else None
            }
        except Exception as e:
            return {
                "is_valid": False,
                "confidence": 0.7,
                "reachable": False,
                "error": str(e)
            }


# Global instance
data_validator = DataValidator()
