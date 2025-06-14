"""
Data Enrichment Agent for enhancing scraped data with external sources.
"""
import asyncio
import logging
import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import httpx

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.data_processing import (
    ProcessingType, GeolocationResult, DataEnrichmentResult,
    ValidationResult, NormalizationResult
)
from config.data_processing_config import data_processing_config

# External API libraries
try:
    import googlemaps
    GOOGLEMAPS_AVAILABLE = True
except ImportError:
    GOOGLEMAPS_AVAILABLE = False
    logging.warning("Google Maps not available. Install with: pip install googlemaps")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests not available. Install with: pip install requests")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available. Install with: pip install yfinance")

try:
    from geopy.geocoders import Nominatim, GoogleV3, MapBox
    from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    logging.warning("geopy not available. Install with: pip install geopy")


class DataEnrichmentAgent(Agent):
    """
    Data Enrichment Agent for enhancing scraped data with external sources.
    """
    
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize the Data Enrichment Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            coordinator_id: ID of the coordinator agent
        """
        super().__init__(agent_id=agent_id, agent_type="data_enrichment", coordinator_id=coordinator_id)
        
        # Configuration
        self.config = data_processing_config.data_enrichment
        
        # Initialize external service clients
        self.clients = {}
        self.geocoders = {}
        
        # Rate limiting
        self.rate_limits = {}
        self.request_counts = {}
        
        # Processing cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Performance metrics
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "api_calls": 0
        }
        
        # Initialize capabilities based on available APIs
        self.capabilities = {
            "geocoding": GEOPY_AVAILABLE or GOOGLEMAPS_AVAILABLE,
            "company_enrichment": bool(self.config.clearbit_api_key or self.config.fullcontact_api_key),
            "financial_data": YFINANCE_AVAILABLE or bool(self.config.alpha_vantage_api_key),
            "weather_data": bool(self.config.openweather_api_key or self.config.weatherapi_key),
            "social_media": bool(self.config.twitter_bearer_token or self.config.linkedin_api_key),
            "data_validation": True,  # Always available with basic validation
            "data_normalization": True  # Always available
        }
        
        # Register message handlers
        self.register_handler("geocode_address", self._handle_geocode_address)
        self.register_handler("enrich_company_data", self._handle_enrich_company_data)
        self.register_handler("get_financial_data", self._handle_get_financial_data)
        self.register_handler("get_weather_data", self._handle_get_weather_data)
        self.register_handler("validate_data", self._handle_validate_data)
        self.register_handler("normalize_data", self._handle_normalize_data)
        self.register_handler("enrich_contact_info", self._handle_enrich_contact_info)
        self.register_handler("enrich_data_batch", self._handle_enrich_batch)
        
        # Initialize clients asynchronously
        asyncio.create_task(self._initialize_clients())
        
        # Start periodic tasks
        asyncio.create_task(self._start_periodic_tasks())
    
    async def _initialize_clients(self):
        """Initialize external service clients."""
        try:
            self.logger.info("Initializing data enrichment clients...")
            
            # Initialize HTTP client
            self.clients["http"] = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True
            )
            
            # Initialize geocoding services
            if GEOPY_AVAILABLE:
                # Nominatim (OpenStreetMap) - free
                self.geocoders["nominatim"] = Nominatim(user_agent="web-scraper-enrichment")
                
                # Google Maps
                if self.config.google_maps_api_key:
                    self.geocoders["google"] = GoogleV3(api_key=self.config.google_maps_api_key)
                
                # MapBox
                if self.config.mapbox_api_key:
                    self.geocoders["mapbox"] = MapBox(api_key=self.config.mapbox_api_key)
            
            # Initialize Google Maps client
            if GOOGLEMAPS_AVAILABLE and self.config.google_maps_api_key:
                self.clients["googlemaps"] = googlemaps.Client(key=self.config.google_maps_api_key)
            
            # Initialize rate limiting
            self._initialize_rate_limits()
            
            self.logger.info("Data enrichment clients initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error initializing data enrichment clients: {e}")
    
    def _initialize_rate_limits(self):
        """Initialize rate limiting for different services."""
        self.rate_limits = {
            "google_maps": {"limit": 50, "window": 60, "requests": []},
            "opencage": {"limit": 100, "window": 60, "requests": []},
            "clearbit": {"limit": 20, "window": 60, "requests": []},
            "openweather": {"limit": 60, "window": 60, "requests": []},
            "alpha_vantage": {"limit": 5, "window": 60, "requests": []},
            "default": {"limit": self.config.global_rate_limit, "window": 60, "requests": []}
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
            
            self.logger.info(
                f"Data Enrichment Agent Performance - Success Rate: {success_rate:.1f}%, "
                f"Cache Hit Rate: {cache_hit_rate:.1f}%, "
                f"API Calls: {self.processing_stats['api_calls']}, "
                f"Avg Processing Time: {self.processing_stats['average_processing_time']:.2f}s"
            )
    
    def _get_cache_key(self, operation: str, data: Any, params: Dict[str, Any] = None) -> str:
        """Generate cache key for enrichment operations."""
        params_str = str(sorted(params.items())) if params else ""
        data_str = str(data) if not isinstance(data, dict) else str(sorted(data.items()))
        content = f"{operation}:{data_str}:{params_str}"
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
        """Check if request is within rate limits."""
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
    
    async def _handle_geocode_address(self, message: Message):
        """Handle address geocoding request."""
        try:
            address = message.data.get("address", "")
            provider = message.data.get("provider", self.config.geocoding_provider)
            
            result = await self.geocode_address(address, provider)
            
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                result=result
            )
            self.outbox.put(response)
            
        except Exception as e:
            self.logger.error(f"Error in geocoding: {e}")
            error_response = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                error=str(e)
            )
            self.outbox.put(error_response)
    
    async def geocode_address(self, address: str, provider: str = None) -> Dict[str, Any]:
        """
        Geocode an address to get coordinates and location information.
        
        Args:
            address: Address to geocode
            provider: Geocoding provider to use
            
        Returns:
            Dictionary containing geocoding results
        """
        start_time = time.time()
        self.processing_stats["total_requests"] += 1
        
        try:
            provider = provider or self.config.geocoding_provider
            
            # Check cache
            cache_key = self._get_cache_key("geocode", address, {"provider": provider})
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Check rate limits
            if not await self._check_rate_limit(provider):
                raise Exception(f"Rate limit exceeded for {provider}")
            
            geocoding_result = None
            
            # Try different geocoding services
            if provider == "google" and "google" in self.geocoders:
                try:
                    location = self.geocoders["google"].geocode(address, timeout=self.config.geocoding_timeout)
                    if location:
                        geocoding_result = GeolocationResult(
                            address=address,
                            formatted_address=location.address,
                            latitude=location.latitude,
                            longitude=location.longitude,
                            confidence=0.9,  # Google typically has high confidence
                            service="google"
                        )
                        
                        # Extract additional components if available
                        if hasattr(location, 'raw') and 'address_components' in location.raw:
                            components = location.raw['address_components']
                            for component in components:
                                types = component.get('types', [])
                                if 'country' in types:
                                    geocoding_result.country = component['long_name']
                                    geocoding_result.country_code = component['short_name']
                                elif 'administrative_area_level_1' in types:
                                    geocoding_result.state = component['long_name']
                                elif 'locality' in types:
                                    geocoding_result.city = component['long_name']
                                elif 'postal_code' in types:
                                    geocoding_result.postal_code = component['long_name']
                        
                        self.processing_stats["api_calls"] += 1
                        
                except Exception as e:
                    self.logger.warning(f"Google geocoding failed: {e}")
            
            # Fallback to Nominatim
            if geocoding_result is None and "nominatim" in self.geocoders:
                try:
                    location = self.geocoders["nominatim"].geocode(address, timeout=self.config.geocoding_timeout)
                    if location:
                        geocoding_result = GeolocationResult(
                            address=address,
                            formatted_address=location.address,
                            latitude=location.latitude,
                            longitude=location.longitude,
                            confidence=0.7,  # Lower confidence for free service
                            service="nominatim"
                        )
                        
                        # Extract components from raw data
                        if hasattr(location, 'raw') and 'display_name' in location.raw:
                            raw_data = location.raw
                            geocoding_result.country = raw_data.get('country')
                            geocoding_result.country_code = raw_data.get('country_code', '').upper()
                            geocoding_result.state = raw_data.get('state')
                            geocoding_result.city = raw_data.get('city') or raw_data.get('town') or raw_data.get('village')
                            geocoding_result.postal_code = raw_data.get('postcode')
                        
                        self.processing_stats["api_calls"] += 1
                        
                except Exception as e:
                    self.logger.warning(f"Nominatim geocoding failed: {e}")
            
            if geocoding_result is None:
                raise Exception("No geocoding service available or address not found")
            
            result = {
                "geocoding": geocoding_result.dict(),
                "processing_time": time.time() - start_time,
                "provider": provider
            }
            
            # Store in cache
            self._store_in_cache(cache_key, result)
            
            self.processing_stats["successful_requests"] += 1
            return result
            
        except Exception as e:
            self.processing_stats["failed_requests"] += 1
            self.logger.error(f"Error geocoding address: {e}")
            raise
        
        finally:
            processing_time = time.time() - start_time
            self.processing_stats["average_processing_time"] = (
                (self.processing_stats["average_processing_time"] * 
                 (self.processing_stats["total_requests"] - 1) + processing_time) /
                self.processing_stats["total_requests"]
            )
