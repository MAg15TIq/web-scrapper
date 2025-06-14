"""
Configuration for advanced data processing capabilities.
"""
import os
from typing import Dict, List, Optional, Any

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    try:
        from pydantic import BaseSettings, Field
    except ImportError:
        # Fallback for older pydantic versions
        class BaseSettings:
            pass
        def Field(default=None, **kwargs):
            return default


class NLPConfig(BaseSettings):
    """Configuration for Natural Language Processing."""
    
    # Model configurations
    spacy_model: str = Field("en_core_web_sm", env="SPACY_MODEL")
    transformers_model: str = Field("distilbert-base-uncased", env="TRANSFORMERS_MODEL")
    sentiment_model: str = Field("cardiffnlp/twitter-roberta-base-sentiment-latest", env="SENTIMENT_MODEL")
    summarization_model: str = Field("facebook/bart-large-cnn", env="SUMMARIZATION_MODEL")
    translation_model: str = Field("Helsinki-NLP/opus-mt-en-de", env="TRANSLATION_MODEL")
    
    # Processing parameters
    max_text_length: int = Field(10000, env="MAX_TEXT_LENGTH")
    summary_max_length: int = Field(150, env="SUMMARY_MAX_LENGTH")
    summary_min_length: int = Field(30, env="SUMMARY_MIN_LENGTH")
    entity_confidence_threshold: float = Field(0.7, env="ENTITY_CONFIDENCE_THRESHOLD")
    sentiment_confidence_threshold: float = Field(0.6, env="SENTIMENT_CONFIDENCE_THRESHOLD")
    
    # Language detection
    language_detection_threshold: float = Field(0.8, env="LANGUAGE_DETECTION_THRESHOLD")
    supported_languages: List[str] = Field(
        ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar"],
        env="SUPPORTED_LANGUAGES"
    )
    
    # Cache settings
    enable_caching: bool = Field(True, env="NLP_ENABLE_CACHING")
    cache_ttl: int = Field(3600, env="NLP_CACHE_TTL")  # 1 hour
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ComputerVisionConfig(BaseSettings):
    """Configuration for Computer Vision processing."""
    
    # OCR configurations
    tesseract_path: Optional[str] = Field(None, env="TESSERACT_PATH")
    tesseract_config: str = Field("--oem 3 --psm 6", env="TESSERACT_CONFIG")
    easyocr_languages: List[str] = Field(["en"], env="EASYOCR_LANGUAGES")
    paddleocr_language: str = Field("en", env="PADDLEOCR_LANGUAGE")
    
    # Image processing
    max_image_size: int = Field(2048, env="MAX_IMAGE_SIZE")  # pixels
    image_quality: int = Field(95, env="IMAGE_QUALITY")  # JPEG quality
    supported_formats: List[str] = Field(
        ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        env="SUPPORTED_IMAGE_FORMATS"
    )
    
    # Object detection
    object_detection_model: str = Field("yolov5s", env="OBJECT_DETECTION_MODEL")
    object_confidence_threshold: float = Field(0.5, env="OBJECT_CONFIDENCE_THRESHOLD")
    nms_threshold: float = Field(0.4, env="NMS_THRESHOLD")
    
    # Image classification
    classification_model: str = Field("resnet50", env="CLASSIFICATION_MODEL")
    classification_threshold: float = Field(0.3, env="CLASSIFICATION_THRESHOLD")
    top_k_predictions: int = Field(5, env="TOP_K_PREDICTIONS")
    
    # Visual element detection
    ui_detection_model: str = Field("custom_ui_detector", env="UI_DETECTION_MODEL")
    ui_confidence_threshold: float = Field(0.6, env="UI_CONFIDENCE_THRESHOLD")
    
    # Screenshot comparison
    similarity_threshold: float = Field(0.95, env="SIMILARITY_THRESHOLD")
    change_detection_sensitivity: float = Field(0.1, env="CHANGE_DETECTION_SENSITIVITY")
    
    # Cache settings
    enable_caching: bool = Field(True, env="CV_ENABLE_CACHING")
    cache_ttl: int = Field(1800, env="CV_CACHE_TTL")  # 30 minutes
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DataEnrichmentConfig(BaseSettings):
    """Configuration for Data Enrichment services."""
    
    # External API configurations
    google_maps_api_key: Optional[str] = Field(None, env="GOOGLE_MAPS_API_KEY")
    opencage_api_key: Optional[str] = Field(None, env="OPENCAGE_API_KEY")
    mapbox_api_key: Optional[str] = Field(None, env="MAPBOX_API_KEY")
    
    # Geocoding settings
    geocoding_provider: str = Field("google", env="GEOCODING_PROVIDER")  # google, opencage, mapbox
    geocoding_timeout: int = Field(10, env="GEOCODING_TIMEOUT")
    geocoding_rate_limit: int = Field(50, env="GEOCODING_RATE_LIMIT")  # requests per minute
    
    # Data validation APIs
    email_validation_api: Optional[str] = Field(None, env="EMAIL_VALIDATION_API")
    phone_validation_api: Optional[str] = Field(None, env="PHONE_VALIDATION_API")
    address_validation_api: Optional[str] = Field(None, env="ADDRESS_VALIDATION_API")
    
    # Company/organization enrichment
    clearbit_api_key: Optional[str] = Field(None, env="CLEARBIT_API_KEY")
    fullcontact_api_key: Optional[str] = Field(None, env="FULLCONTACT_API_KEY")
    
    # Social media enrichment
    twitter_bearer_token: Optional[str] = Field(None, env="TWITTER_BEARER_TOKEN")
    linkedin_api_key: Optional[str] = Field(None, env="LINKEDIN_API_KEY")
    
    # Financial data
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    yahoo_finance_enabled: bool = Field(True, env="YAHOO_FINANCE_ENABLED")
    
    # Weather data
    openweather_api_key: Optional[str] = Field(None, env="OPENWEATHER_API_KEY")
    weatherapi_key: Optional[str] = Field(None, env="WEATHERAPI_KEY")
    
    # Cache settings
    enable_caching: bool = Field(True, env="ENRICHMENT_ENABLE_CACHING")
    cache_ttl: int = Field(7200, env="ENRICHMENT_CACHE_TTL")  # 2 hours
    
    # Rate limiting
    global_rate_limit: int = Field(100, env="GLOBAL_RATE_LIMIT")  # requests per minute
    per_service_rate_limit: int = Field(30, env="PER_SERVICE_RATE_LIMIT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DataValidationConfig(BaseSettings):
    """Configuration for Data Validation."""
    
    # Validation rules
    email_regex: str = Field(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        env="EMAIL_REGEX"
    )
    phone_regex: str = Field(
        r'^\+?1?-?\.?\s?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$',
        env="PHONE_REGEX"
    )
    url_regex: str = Field(
        r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$',
        env="URL_REGEX"
    )
    
    # Validation thresholds
    confidence_threshold: float = Field(0.8, env="VALIDATION_CONFIDENCE_THRESHOLD")
    external_validation_timeout: int = Field(5, env="EXTERNAL_VALIDATION_TIMEOUT")
    
    # Data quality scoring
    completeness_weight: float = Field(0.3, env="COMPLETENESS_WEIGHT")
    accuracy_weight: float = Field(0.4, env="ACCURACY_WEIGHT")
    consistency_weight: float = Field(0.3, env="CONSISTENCY_WEIGHT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DataNormalizationConfig(BaseSettings):
    """Configuration for Data Normalization."""
    
    # Text normalization
    normalize_case: bool = Field(True, env="NORMALIZE_CASE")
    remove_extra_whitespace: bool = Field(True, env="REMOVE_EXTRA_WHITESPACE")
    normalize_unicode: bool = Field(True, env="NORMALIZE_UNICODE")
    
    # Date normalization
    default_date_format: str = Field("%Y-%m-%d", env="DEFAULT_DATE_FORMAT")
    date_formats: List[str] = Field(
        ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"],
        env="DATE_FORMATS"
    )
    
    # Number normalization
    decimal_places: int = Field(2, env="DECIMAL_PLACES")
    thousand_separator: str = Field(",", env="THOUSAND_SEPARATOR")
    decimal_separator: str = Field(".", env="DECIMAL_SEPARATOR")
    
    # Address normalization
    normalize_addresses: bool = Field(True, env="NORMALIZE_ADDRESSES")
    address_components: List[str] = Field(
        ["street", "city", "state", "country", "postal_code"],
        env="ADDRESS_COMPONENTS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DataProcessingConfig:
    """Main configuration class for data processing."""
    
    def __init__(self):
        """Initialize all configuration sections."""
        self.nlp = NLPConfig()
        self.computer_vision = ComputerVisionConfig()
        self.data_enrichment = DataEnrichmentConfig()
        self.data_validation = DataValidationConfig()
        self.data_normalization = DataNormalizationConfig()
    
    def get_processing_config(self, processing_type: str) -> Dict[str, Any]:
        """Get configuration for a specific processing type."""
        config_map = {
            "nlp": self.nlp.dict(),
            "computer_vision": self.computer_vision.dict(),
            "data_enrichment": self.data_enrichment.dict(),
            "data_validation": self.data_validation.dict(),
            "data_normalization": self.data_normalization.dict()
        }
        return config_map.get(processing_type, {})
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are configured."""
        validations = {}
        
        # Check geocoding APIs
        validations["google_maps"] = bool(self.data_enrichment.google_maps_api_key)
        validations["opencage"] = bool(self.data_enrichment.opencage_api_key)
        validations["mapbox"] = bool(self.data_enrichment.mapbox_api_key)
        
        # Check enrichment APIs
        validations["clearbit"] = bool(self.data_enrichment.clearbit_api_key)
        validations["fullcontact"] = bool(self.data_enrichment.fullcontact_api_key)
        validations["twitter"] = bool(self.data_enrichment.twitter_bearer_token)
        validations["alpha_vantage"] = bool(self.data_enrichment.alpha_vantage_api_key)
        validations["openweather"] = bool(self.data_enrichment.openweather_api_key)
        
        return validations
    
    def get_available_services(self) -> Dict[str, List[str]]:
        """Get list of available services based on configured API keys."""
        api_validations = self.validate_api_keys()
        
        services = {
            "geocoding": [],
            "enrichment": [],
            "validation": [],
            "weather": [],
            "financial": []
        }
        
        # Geocoding services
        if api_validations["google_maps"]:
            services["geocoding"].append("google_maps")
        if api_validations["opencage"]:
            services["geocoding"].append("opencage")
        if api_validations["mapbox"]:
            services["geocoding"].append("mapbox")
        
        # Enrichment services
        if api_validations["clearbit"]:
            services["enrichment"].append("clearbit")
        if api_validations["fullcontact"]:
            services["enrichment"].append("fullcontact")
        if api_validations["twitter"]:
            services["enrichment"].append("twitter")
        
        # Weather services
        if api_validations["openweather"]:
            services["weather"].append("openweather")
        
        # Financial services
        if api_validations["alpha_vantage"]:
            services["financial"].append("alpha_vantage")
        if self.data_enrichment.yahoo_finance_enabled:
            services["financial"].append("yahoo_finance")
        
        return services


# Global configuration instance
data_processing_config = DataProcessingConfig()
