"""
Specialized Geocoding Agent for location data enhancement.
"""
import asyncio
import logging
import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import httpx

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.data_processing import GeolocationResult
from config.data_processing_config import data_processing_config

# Geocoding libraries
try:
    import googlemaps
    GOOGLEMAPS_AVAILABLE = True
except ImportError:
    GOOGLEMAPS_AVAILABLE = False
    logging.warning("Google Maps not available. Install with: pip install googlemaps")

try:
    from geopy.geocoders import Nominatim, GoogleV3, MapBox, OpenCage
    from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    logging.warning("geopy not available. Install with: pip install geopy")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class GeocodingAgent(Agent):
    """
    Specialized Geocoding Agent for location data enhancement.
    """
    
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize the Geocoding Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            coordinator_id: ID of the coordinator agent
        """
        super().__init__(agent_id=agent_id, agent_type="geocoding", coordinator_id=coordinator_id)
        
        # Configuration
        self.config = data_processing_config.data_enrichment
        
        # Initialize geocoding services
        self.geocoders = {}
        self.service_priorities = []
        
        # Rate limiting per service
        self.rate_limits = {}
        
        # Processing cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Performance metrics
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "service_usage": {},
            "average_processing_time": 0.0,
            "cache_hits": 0
        }
        
        # Address patterns for validation
        self.address_patterns = {
            "street_number": re.compile(r'\d+'),
            "postal_code": re.compile(r'\b\d{5}(?:-\d{4})?\b'),  # US ZIP codes
            "coordinates": re.compile(r'[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)')
        }
        
        # Initialize capabilities
        self.capabilities = {
            "forward_geocoding": True,  # Address to coordinates
            "reverse_geocoding": True,  # Coordinates to address
            "batch_geocoding": True,
            "address_validation": True,
            "coordinate_validation": True,
            "multiple_providers": True
        }
        
        # Register message handlers
        self.register_handler("geocode_forward", self._handle_geocode_forward)
        self.register_handler("geocode_reverse", self._handle_geocode_reverse)
        self.register_handler("geocode_batch", self._handle_geocode_batch)
        self.register_handler("validate_address", self._handle_validate_address)
        self.register_handler("validate_coordinates", self._handle_validate_coordinates)
        self.register_handler("get_location_info", self._handle_get_location_info)
        
        # Initialize geocoding services
        asyncio.create_task(self._initialize_geocoders())
        
        # Start periodic tasks
        asyncio.create_task(self._start_periodic_tasks())
    
    async def _initialize_geocoders(self):
        """Initialize available geocoding services."""
        try:
            self.logger.info("Initializing geocoding services...")
            
            # Initialize rate limiting
            self._initialize_rate_limits()
            
            if GEOPY_AVAILABLE:
                # Nominatim (OpenStreetMap) - free, no API key required
                try:
                    self.geocoders["nominatim"] = Nominatim(
                        user_agent="web-scraper-geocoding-agent",
                        timeout=self.config.geocoding_timeout
                    )
                    self.service_priorities.append("nominatim")
                    self.processing_stats["service_usage"]["nominatim"] = 0
                    self.logger.info("Initialized Nominatim geocoder")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Nominatim: {e}")
                
                # Google Maps
                if self.config.google_maps_api_key:
                    try:
                        self.geocoders["google"] = GoogleV3(
                            api_key=self.config.google_maps_api_key,
                            timeout=self.config.geocoding_timeout
                        )
                        self.service_priorities.insert(0, "google")  # Higher priority
                        self.processing_stats["service_usage"]["google"] = 0
                        self.logger.info("Initialized Google Maps geocoder")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize Google Maps: {e}")
                
                # OpenCage
                if self.config.opencage_api_key:
                    try:
                        self.geocoders["opencage"] = OpenCage(
                            api_key=self.config.opencage_api_key,
                            timeout=self.config.geocoding_timeout
                        )
                        self.service_priorities.insert(-1, "opencage")  # Medium priority
                        self.processing_stats["service_usage"]["opencage"] = 0
                        self.logger.info("Initialized OpenCage geocoder")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize OpenCage: {e}")
                
                # MapBox
                if self.config.mapbox_api_key:
                    try:
                        self.geocoders["mapbox"] = MapBox(
                            api_key=self.config.mapbox_api_key,
                            timeout=self.config.geocoding_timeout
                        )
                        self.service_priorities.insert(-1, "mapbox")  # Medium priority
                        self.processing_stats["service_usage"]["mapbox"] = 0
                        self.logger.info("Initialized MapBox geocoder")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize MapBox: {e}")
            
            # Initialize Google Maps client for additional features
            if GOOGLEMAPS_AVAILABLE and self.config.google_maps_api_key:
                try:
                    self.geocoders["googlemaps_client"] = googlemaps.Client(
                        key=self.config.google_maps_api_key
                    )
                    self.logger.info("Initialized Google Maps client")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Google Maps client: {e}")
            
            if not self.geocoders:
                self.logger.warning("No geocoding services available")
            else:
                self.logger.info(f"Initialized {len(self.geocoders)} geocoding services: {list(self.geocoders.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error initializing geocoding services: {e}")
    
    def _initialize_rate_limits(self):
        """Initialize rate limiting for different geocoding services."""
        self.rate_limits = {
            "google": {"limit": 50, "window": 60, "requests": []},
            "opencage": {"limit": 100, "window": 60, "requests": []},
            "mapbox": {"limit": 100, "window": 60, "requests": []},
            "nominatim": {"limit": 60, "window": 60, "requests": []},  # Be respectful to free service
            "default": {"limit": self.config.geocoding_rate_limit, "window": 60, "requests": []}
        }
    
    async def _start_periodic_tasks(self):
        """Start periodic maintenance tasks."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_cache()
                await self._cleanup_rate_limits()
                await self._update_performance_metrics()
            except Exception as e:
                self.logger.error(f"Error in periodic tasks: {e}")
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.config.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _cleanup_rate_limits(self):
        """Clean up old rate limit entries."""
        current_time = time.time()
        
        for service, limits in self.rate_limits.items():
            window = limits["window"]
            limits["requests"] = [req_time for req_time in limits["requests"] 
                                if current_time - req_time < window]
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        if self.processing_stats["total_requests"] > 0:
            success_rate = (self.processing_stats["successful_requests"] / 
                          self.processing_stats["total_requests"]) * 100
            cache_hit_rate = (self.processing_stats["cache_hits"] / 
                            self.processing_stats["total_requests"]) * 100
            
            service_usage = ", ".join([f"{service}: {count}" 
                                     for service, count in self.processing_stats["service_usage"].items()])
            
            self.logger.info(
                f"Geocoding Agent Performance - Success Rate: {success_rate:.1f}%, "
                f"Cache Hit Rate: {cache_hit_rate:.1f}%, "
                f"Service Usage: {service_usage}, "
                f"Avg Processing Time: {self.processing_stats['average_processing_time']:.2f}s"
            )
    
    def _get_cache_key(self, operation: str, query: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key for geocoding operations."""
        params_str = str(sorted(params.items())) if params else ""
        content = f"{operation}:{query.lower().strip()}:{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache if available and not expired."""
        if not self.config.enable_caching:
            return None
        
        if cache_key in self.cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.config.cache_ttl:
                self.processing_stats["cache_hits"] += 1
                return self.cache[cache_key]
            else:
                # Remove expired entry
                self.cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
        
        return None
    
    def _store_in_cache(self, cache_key: str, result: Any):
        """Store result in cache."""
        if self.config.enable_caching:
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()
    
    async def _check_rate_limit(self, service: str) -> bool:
        """Check if request is within rate limits for a service."""
        current_time = time.time()
        limits = self.rate_limits.get(service, self.rate_limits["default"])
        
        # Clean old requests
        window = limits["window"]
        limits["requests"] = [req_time for req_time in limits["requests"] 
                            if current_time - req_time < window]
        
        # Check if under limit
        if len(limits["requests"]) < limits["limit"]:
            limits["requests"].append(current_time)
            return True
        
        return False
    
    def _validate_address_format(self, address: str) -> Dict[str, Any]:
        """Validate address format and extract components."""
        validation = {
            "is_valid": False,
            "has_street_number": False,
            "has_postal_code": False,
            "is_coordinates": False,
            "confidence": 0.0,
            "components": {}
        }
        
        if not address or len(address.strip()) < 5:
            return validation
        
        address = address.strip()
        
        # Check if it's coordinates
        if self.address_patterns["coordinates"].match(address):
            validation["is_coordinates"] = True
            validation["is_valid"] = True
            validation["confidence"] = 0.9
            return validation
        
        # Check for street number
        if self.address_patterns["street_number"].search(address):
            validation["has_street_number"] = True
            validation["confidence"] += 0.3
        
        # Check for postal code
        postal_match = self.address_patterns["postal_code"].search(address)
        if postal_match:
            validation["has_postal_code"] = True
            validation["components"]["postal_code"] = postal_match.group()
            validation["confidence"] += 0.4
        
        # Basic validation - address should have at least some components
        parts = address.split(',')
        if len(parts) >= 2:
            validation["confidence"] += 0.3
        
        validation["is_valid"] = validation["confidence"] >= 0.5
        
        return validation
    
    def _validate_coordinates(self, lat: float, lng: float) -> Dict[str, Any]:
        """Validate coordinate values."""
        validation = {
            "is_valid": False,
            "latitude_valid": False,
            "longitude_valid": False,
            "precision": "unknown",
            "confidence": 0.0
        }
        
        # Check latitude range
        if -90 <= lat <= 90:
            validation["latitude_valid"] = True
            validation["confidence"] += 0.5
        
        # Check longitude range
        if -180 <= lng <= 180:
            validation["longitude_valid"] = True
            validation["confidence"] += 0.5
        
        validation["is_valid"] = validation["latitude_valid"] and validation["longitude_valid"]
        
        # Determine precision based on decimal places
        lat_decimals = len(str(lat).split('.')[-1]) if '.' in str(lat) else 0
        lng_decimals = len(str(lng).split('.')[-1]) if '.' in str(lng) else 0
        
        min_decimals = min(lat_decimals, lng_decimals)
        if min_decimals >= 6:
            validation["precision"] = "high"  # ~1 meter
        elif min_decimals >= 4:
            validation["precision"] = "medium"  # ~10 meters
        elif min_decimals >= 2:
            validation["precision"] = "low"  # ~1 km
        else:
            validation["precision"] = "very_low"  # >1 km
        
        return validation
    
    async def _handle_geocode_forward(self, message: Message):
        """Handle forward geocoding request (address to coordinates)."""
        try:
            address = message.data.get("address", "")
            preferred_service = message.data.get("service", None)
            include_components = message.data.get("include_components", True)
            
            result = await self.geocode_forward(address, preferred_service, include_components)
            
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                result=result
            )
            self.outbox.put(response)
            
        except Exception as e:
            self.logger.error(f"Error in forward geocoding: {e}")
            error_response = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                error=str(e)
            )
            self.outbox.put(error_response)
