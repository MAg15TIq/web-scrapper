"""
Phase 4: Advanced Anti-Detection System with ML-based fingerprint generation.
"""
import asyncio
import logging
import time
import random
import json
import hashlib
import numpy as np
import pickle
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import httpx
from urllib.parse import urlparse

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# GeoIP imports
try:
    import geoip2.database
    import geoip2.errors
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False

from agents.base import Agent
from models.task import Task, TaskType
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage


@dataclass
class BehavioralPattern:
    """Represents a behavioral pattern for mimicking human behavior."""
    name: str
    description: str
    scroll_patterns: List[Dict[str, Any]]
    click_patterns: List[Dict[str, Any]]
    typing_patterns: List[Dict[str, Any]]
    pause_patterns: List[Dict[str, Any]]
    navigation_patterns: List[Dict[str, Any]]
    success_rate: float = 0.0
    usage_count: int = 0


@dataclass
class FingerprintProfile:
    """Enhanced fingerprint profile with ML-generated characteristics."""
    user_agent: str
    screen_resolution: str
    color_depth: int
    timezone: str
    language: str
    platform: str
    canvas_fingerprint: str
    webgl_fingerprint: str
    audio_fingerprint: str
    font_fingerprint: str
    hardware_concurrency: int
    device_memory: int
    connection_type: str
    geolocation: Optional[Dict[str, float]]
    behavioral_score: float = 0.0
    success_rate: float = 0.0
    last_used: float = 0.0


class MLFingerprintGenerator:
    """ML-based fingerprint generator that learns from successful patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger("ml_fingerprint_generator")
        self.model_path = "data/fingerprint_model.pkl"
        self.scaler_path = "data/fingerprint_scaler.pkl"
        self.patterns_path = "data/behavioral_patterns.json"
        
        # Initialize ML components
        self.classifier = None
        self.scaler = None
        self.behavioral_patterns = []
        
        # Feature vectors for successful fingerprints
        self.successful_fingerprints = []
        self.failed_fingerprints = []
        
        # Load existing models and patterns
        self._load_models()
        self._load_behavioral_patterns()
        
        # Real user data patterns (anonymized)
        self.real_user_patterns = self._load_real_user_patterns()
        
    def _load_models(self):
        """Load pre-trained ML models."""
        try:
            if os.path.exists(self.model_path) and ML_AVAILABLE:
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                self.logger.info("Loaded fingerprint classification model")
            
            if os.path.exists(self.scaler_path) and ML_AVAILABLE:
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info("Loaded fingerprint feature scaler")
                
        except Exception as e:
            self.logger.warning(f"Failed to load ML models: {e}")
            
    def _load_behavioral_patterns(self):
        """Load behavioral patterns from storage."""
        try:
            if os.path.exists(self.patterns_path):
                with open(self.patterns_path, 'r') as f:
                    patterns_data = json.load(f)
                    self.behavioral_patterns = [
                        BehavioralPattern(**pattern) for pattern in patterns_data
                    ]
                self.logger.info(f"Loaded {len(self.behavioral_patterns)} behavioral patterns")
        except Exception as e:
            self.logger.warning(f"Failed to load behavioral patterns: {e}")
            
    def _load_real_user_patterns(self) -> List[Dict[str, Any]]:
        """Load anonymized real user patterns for mimicking."""
        # This would typically load from a database of anonymized user behavior
        return [
            {
                "session_duration": {"mean": 180, "std": 60},
                "pages_per_session": {"mean": 5.2, "std": 2.1},
                "scroll_speed": {"mean": 1200, "std": 300},
                "click_frequency": {"mean": 0.8, "std": 0.3},
                "typing_speed": {"mean": 45, "std": 15},
                "pause_duration": {"mean": 2.5, "std": 1.2}
            }
        ]
        
    def generate_ml_fingerprint(self, 
                               target_domain: str = None,
                               success_history: List[Dict] = None) -> FingerprintProfile:
        """Generate an ML-optimized fingerprint based on success patterns."""
        
        # If we have a trained model, use it to predict optimal features
        if self.classifier and self.scaler and ML_AVAILABLE:
            return self._generate_predicted_fingerprint(target_domain, success_history)
        else:
            return self._generate_heuristic_fingerprint(target_domain)
            
    def _generate_predicted_fingerprint(self, 
                                      target_domain: str,
                                      success_history: List[Dict]) -> FingerprintProfile:
        """Generate fingerprint using ML predictions."""
        try:
            # Extract features from successful fingerprints for this domain
            domain_features = self._extract_domain_features(target_domain, success_history)
            
            # Predict optimal fingerprint characteristics
            if len(domain_features) > 0:
                features_array = np.array(domain_features).reshape(1, -1)
                scaled_features = self.scaler.transform(features_array)
                prediction = self.classifier.predict_proba(scaled_features)[0]
                
                # Use prediction to guide fingerprint generation
                return self._create_fingerprint_from_prediction(prediction)
            else:
                return self._generate_heuristic_fingerprint(target_domain)
                
        except Exception as e:
            self.logger.error(f"ML fingerprint generation failed: {e}")
            return self._generate_heuristic_fingerprint(target_domain)
            
    def _generate_heuristic_fingerprint(self, target_domain: str) -> FingerprintProfile:
        """Generate fingerprint using heuristic rules."""
        # Base fingerprint generation with enhanced randomization
        base_profiles = [
            # Windows Chrome profiles
            {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "platform": "Win32",
                "screen_resolution": random.choice(["1920x1080", "1366x768", "2560x1440"]),
                "hardware_concurrency": random.choice([4, 8, 12, 16]),
                "device_memory": random.choice([4, 8, 16, 32])
            },
            # macOS Safari profiles
            {
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
                "platform": "MacIntel",
                "screen_resolution": random.choice(["2560x1600", "1920x1080", "3840x2160"]),
                "hardware_concurrency": random.choice([8, 10, 12]),
                "device_memory": random.choice([8, 16, 32, 64])
            }
        ]
        
        profile_data = random.choice(base_profiles)
        
        return FingerprintProfile(
            user_agent=profile_data["user_agent"],
            screen_resolution=profile_data["screen_resolution"],
            color_depth=random.choice([24, 32]),
            timezone=random.choice([
                "America/New_York", "America/Los_Angeles", "Europe/London",
                "Europe/Paris", "Asia/Tokyo", "Australia/Sydney"
            ]),
            language=random.choice([
                "en-US,en;q=0.9", "en-GB,en;q=0.9", "fr-FR,fr;q=0.9",
                "de-DE,de;q=0.9", "es-ES,es;q=0.9"
            ]),
            platform=profile_data["platform"],
            canvas_fingerprint=self._generate_canvas_fingerprint(),
            webgl_fingerprint=self._generate_webgl_fingerprint(),
            audio_fingerprint=self._generate_audio_fingerprint(),
            font_fingerprint=self._generate_font_fingerprint(),
            hardware_concurrency=profile_data["hardware_concurrency"],
            device_memory=profile_data["device_memory"],
            connection_type=random.choice(["4g", "wifi", "ethernet"]),
            geolocation=self._generate_geolocation(),
            behavioral_score=random.uniform(0.7, 0.95),
            last_used=time.time()
        )
        
    def _generate_canvas_fingerprint(self) -> str:
        """Generate a realistic canvas fingerprint."""
        # Simulate canvas rendering variations
        base_data = f"canvas_{random.randint(1000, 9999)}_{time.time()}"
        return hashlib.sha256(base_data.encode()).hexdigest()[:16]
        
    def _generate_webgl_fingerprint(self) -> str:
        """Generate a realistic WebGL fingerprint."""
        gpu_vendors = ["NVIDIA", "AMD", "Intel", "Apple"]
        gpu_models = ["RTX 4080", "RX 7800 XT", "UHD Graphics 630", "M2 Pro"]
        
        vendor = random.choice(gpu_vendors)
        model = random.choice(gpu_models)
        base_data = f"webgl_{vendor}_{model}_{random.randint(1000, 9999)}"
        return hashlib.sha256(base_data.encode()).hexdigest()[:16]
        
    def _generate_audio_fingerprint(self) -> str:
        """Generate a realistic audio fingerprint."""
        base_data = f"audio_{random.uniform(0.1, 0.9):.6f}_{time.time()}"
        return hashlib.sha256(base_data.encode()).hexdigest()[:16]
        
    def _generate_font_fingerprint(self) -> str:
        """Generate a realistic font fingerprint."""
        font_count = random.randint(45, 120)
        base_data = f"fonts_{font_count}_{random.randint(1000, 9999)}"
        return hashlib.sha256(base_data.encode()).hexdigest()[:16]
        
    def _generate_geolocation(self) -> Optional[Dict[str, float]]:
        """Generate realistic geolocation data."""
        if random.random() < 0.3:  # 30% chance of having geolocation
            # Major city coordinates with some noise
            cities = [
                {"lat": 40.7128, "lng": -74.0060},  # New York
                {"lat": 51.5074, "lng": -0.1278},   # London
                {"lat": 48.8566, "lng": 2.3522},    # Paris
                {"lat": 35.6762, "lng": 139.6503},  # Tokyo
                {"lat": -33.8688, "lng": 151.2093}, # Sydney
            ]
            city = random.choice(cities)
            return {
                "latitude": city["lat"] + random.uniform(-0.1, 0.1),
                "longitude": city["lng"] + random.uniform(-0.1, 0.1),
                "accuracy": random.uniform(10, 100)
            }
        return None
        
    def update_fingerprint_success(self, fingerprint: FingerprintProfile, success: bool):
        """Update fingerprint success rate based on usage results."""
        fingerprint.usage_count += 1
        if success:
            fingerprint.success_rate = (
                (fingerprint.success_rate * (fingerprint.usage_count - 1) + 1.0) / 
                fingerprint.usage_count
            )
        else:
            fingerprint.success_rate = (
                (fingerprint.success_rate * (fingerprint.usage_count - 1)) / 
                fingerprint.usage_count
            )
            
        # Store for ML training
        features = self._extract_fingerprint_features(fingerprint)
        if success:
            self.successful_fingerprints.append(features)
        else:
            self.failed_fingerprints.append(features)
            
        # Retrain model periodically
        if len(self.successful_fingerprints) + len(self.failed_fingerprints) > 100:
            self._retrain_model()
            
    def _extract_fingerprint_features(self, fingerprint: FingerprintProfile) -> List[float]:
        """Extract numerical features from fingerprint for ML."""
        features = [
            len(fingerprint.user_agent),
            fingerprint.color_depth,
            fingerprint.hardware_concurrency,
            fingerprint.device_memory,
            len(fingerprint.language),
            1.0 if fingerprint.geolocation else 0.0,
            fingerprint.behavioral_score
        ]
        return features
        
    def _retrain_model(self):
        """Retrain the ML model with new data."""
        if not ML_AVAILABLE:
            return
            
        try:
            # Prepare training data
            X = np.array(self.successful_fingerprints + self.failed_fingerprints)
            y = ([1] * len(self.successful_fingerprints) + 
                 [0] * len(self.failed_fingerprints))
            
            if len(X) < 10:  # Need minimum data
                return
                
            # Train scaler and model
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X_scaled, y)
            
            # Save models
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            self.logger.info("Retrained fingerprint ML model")
            
            # Clear training data to prevent memory issues
            self.successful_fingerprints = self.successful_fingerprints[-50:]
            self.failed_fingerprints = self.failed_fingerprints[-50:]
            
        except Exception as e:
            self.logger.error(f"Failed to retrain ML model: {e}")


class AdvancedProxyManager:
    """Advanced proxy management with geolocation and health monitoring."""

    def __init__(self):
        self.logger = logging.getLogger("advanced_proxy_manager")
        self.proxies: Dict[str, Dict[str, Any]] = {}
        self.proxy_pools: Dict[str, List[str]] = {
            "residential": [],
            "datacenter": [],
            "mobile": []
        }
        self.geolocation_db = None
        self.health_check_interval = 60
        self.rotation_strategy = "round_robin"  # round_robin, random, performance_based

        # Load GeoIP database if available
        self._load_geolocation_db()

    def _load_geolocation_db(self):
        """Load GeoIP database for proxy geolocation."""
        try:
            if GEOIP_AVAILABLE:
                # Try to load GeoLite2 database
                db_paths = [
                    "data/GeoLite2-City.mmdb",
                    "/usr/share/GeoIP/GeoLite2-City.mmdb",
                    "GeoLite2-City.mmdb"
                ]

                for path in db_paths:
                    if os.path.exists(path):
                        self.geolocation_db = geoip2.database.Reader(path)
                        self.logger.info(f"Loaded GeoIP database from {path}")
                        break
        except Exception as e:
            self.logger.warning(f"Failed to load GeoIP database: {e}")

    def add_proxy(self, proxy_url: str, proxy_type: str = "datacenter",
                  country: str = None, city: str = None, **kwargs):
        """Add a proxy with enhanced metadata."""
        proxy_id = hashlib.md5(proxy_url.encode()).hexdigest()

        proxy_info = {
            "url": proxy_url,
            "type": proxy_type,
            "country": country,
            "city": city,
            "added_at": time.time(),
            "last_used": 0,
            "success_count": 0,
            "failure_count": 0,
            "avg_response_time": 0,
            "health_score": 1.0,
            "is_active": True,
            "consecutive_failures": 0,
            "last_health_check": 0,
            "geolocation": None,
            **kwargs
        }

        # Detect geolocation if not provided
        if not country and self.geolocation_db:
            proxy_info["geolocation"] = self._detect_proxy_location(proxy_url)

        self.proxies[proxy_id] = proxy_info
        self.proxy_pools[proxy_type].append(proxy_id)

        self.logger.info(f"Added {proxy_type} proxy: {proxy_url}")

    def _detect_proxy_location(self, proxy_url: str) -> Optional[Dict[str, Any]]:
        """Detect proxy geolocation using GeoIP."""
        try:
            # Extract IP from proxy URL
            from urllib.parse import urlparse
            parsed = urlparse(proxy_url)
            host = parsed.hostname

            if host and self.geolocation_db:
                response = self.geolocation_db.city(host)
                return {
                    "country": response.country.name,
                    "country_code": response.country.iso_code,
                    "city": response.city.name,
                    "latitude": float(response.location.latitude) if response.location.latitude else None,
                    "longitude": float(response.location.longitude) if response.location.longitude else None,
                    "timezone": str(response.location.time_zone) if response.location.time_zone else None
                }
        except Exception as e:
            self.logger.debug(f"Failed to detect proxy location: {e}")
        return None

    def get_proxy_by_location(self, country: str = None, city: str = None) -> Optional[Dict[str, Any]]:
        """Get a proxy from a specific location."""
        candidates = []

        for proxy_id, proxy_info in self.proxies.items():
            if not proxy_info["is_active"]:
                continue

            # Check country match
            if country:
                proxy_country = (proxy_info.get("country") or
                               (proxy_info.get("geolocation", {}) or {}).get("country"))
                if proxy_country and proxy_country.lower() != country.lower():
                    continue

            # Check city match
            if city:
                proxy_city = (proxy_info.get("city") or
                             (proxy_info.get("geolocation", {}) or {}).get("city"))
                if proxy_city and proxy_city.lower() != city.lower():
                    continue

            candidates.append((proxy_id, proxy_info))

        if not candidates:
            return None

        # Select best candidate based on health score
        best_proxy = max(candidates, key=lambda x: x[1]["health_score"])
        return best_proxy[1]

    def get_optimal_proxy(self, target_domain: str = None) -> Optional[Dict[str, Any]]:
        """Get the optimal proxy based on performance and strategy."""
        active_proxies = [
            (pid, pinfo) for pid, pinfo in self.proxies.items()
            if pinfo["is_active"] and pinfo["consecutive_failures"] < 3
        ]

        if not active_proxies:
            return None

        if self.rotation_strategy == "performance_based":
            # Select based on health score and response time
            scored_proxies = []
            for proxy_id, proxy_info in active_proxies:
                score = (proxy_info["health_score"] * 0.7 +
                        (1.0 / max(proxy_info["avg_response_time"], 0.1)) * 0.3)
                scored_proxies.append((score, proxy_info))

            return max(scored_proxies, key=lambda x: x[0])[1]

        elif self.rotation_strategy == "random":
            return random.choice(active_proxies)[1]

        else:  # round_robin
            # Simple round-robin selection
            if hasattr(self, '_last_proxy_index'):
                self._last_proxy_index = (self._last_proxy_index + 1) % len(active_proxies)
            else:
                self._last_proxy_index = 0
            return active_proxies[self._last_proxy_index][1]

    async def health_check_proxy(self, proxy_info: Dict[str, Any]) -> bool:
        """Perform health check on a proxy."""
        try:
            start_time = time.time()

            async with httpx.AsyncClient(
                proxies={"all://": proxy_info["url"]},
                timeout=10.0
            ) as client:
                response = await client.get("https://httpbin.org/ip")
                response_time = time.time() - start_time

                if response.status_code == 200:
                    # Update proxy stats
                    proxy_info["success_count"] += 1
                    proxy_info["consecutive_failures"] = 0
                    proxy_info["last_health_check"] = time.time()

                    # Update average response time
                    if proxy_info["avg_response_time"] == 0:
                        proxy_info["avg_response_time"] = response_time
                    else:
                        proxy_info["avg_response_time"] = (
                            proxy_info["avg_response_time"] * 0.8 + response_time * 0.2
                        )

                    # Update health score
                    total_requests = proxy_info["success_count"] + proxy_info["failure_count"]
                    if total_requests > 0:
                        success_rate = proxy_info["success_count"] / total_requests
                        response_score = min(1.0, 5.0 / max(proxy_info["avg_response_time"], 0.1))
                        proxy_info["health_score"] = success_rate * 0.7 + response_score * 0.3

                    return True

        except Exception as e:
            self.logger.debug(f"Proxy health check failed: {e}")

        # Health check failed
        proxy_info["failure_count"] += 1
        proxy_info["consecutive_failures"] += 1
        proxy_info["last_health_check"] = time.time()

        # Deactivate proxy if too many consecutive failures
        if proxy_info["consecutive_failures"] >= 5:
            proxy_info["is_active"] = False
            self.logger.warning(f"Deactivated proxy due to failures: {proxy_info['url']}")

        return False

    async def periodic_health_check(self):
        """Periodically check proxy health."""
        while True:
            try:
                for proxy_info in self.proxies.values():
                    if (time.time() - proxy_info["last_health_check"] > self.health_check_interval):
                        await self.health_check_proxy(proxy_info)
                        await asyncio.sleep(1)  # Rate limit health checks

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                self.logger.error(f"Error in proxy health check: {e}")
                await asyncio.sleep(60)

    def get_proxy_stats(self) -> Dict[str, Any]:
        """Get comprehensive proxy statistics."""
        total_proxies = len(self.proxies)
        active_proxies = sum(1 for p in self.proxies.values() if p["is_active"])

        stats_by_type = {}
        stats_by_country = {}

        for proxy_info in self.proxies.values():
            # Stats by type
            proxy_type = proxy_info["type"]
            if proxy_type not in stats_by_type:
                stats_by_type[proxy_type] = {"total": 0, "active": 0, "avg_health": 0}
            stats_by_type[proxy_type]["total"] += 1
            if proxy_info["is_active"]:
                stats_by_type[proxy_type]["active"] += 1
            stats_by_type[proxy_type]["avg_health"] += proxy_info["health_score"]

            # Stats by country
            country = (proxy_info.get("country") or
                      (proxy_info.get("geolocation", {}) or {}).get("country", "Unknown"))
            if country not in stats_by_country:
                stats_by_country[country] = {"total": 0, "active": 0}
            stats_by_country[country]["total"] += 1
            if proxy_info["is_active"]:
                stats_by_country[country]["active"] += 1

        # Calculate averages
        for type_stats in stats_by_type.values():
            if type_stats["total"] > 0:
                type_stats["avg_health"] /= type_stats["total"]

        return {
            "total_proxies": total_proxies,
            "active_proxies": active_proxies,
            "stats_by_type": stats_by_type,
            "stats_by_country": stats_by_country,
            "rotation_strategy": self.rotation_strategy
        }
