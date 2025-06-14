"""
Data models for advanced data processing capabilities.
"""
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class ProcessingType(str, Enum):
    """Types of data processing operations."""
    NLP_ENTITY_EXTRACTION = "nlp_entity_extraction"
    NLP_SENTIMENT_ANALYSIS = "nlp_sentiment_analysis"
    NLP_TEXT_SUMMARIZATION = "nlp_text_summarization"
    NLP_LANGUAGE_DETECTION = "nlp_language_detection"
    NLP_TRANSLATION = "nlp_translation"
    CV_OCR_EXTRACTION = "cv_ocr_extraction"
    CV_IMAGE_CLASSIFICATION = "cv_image_classification"
    CV_OBJECT_DETECTION = "cv_object_detection"
    CV_VISUAL_ELEMENT_DETECTION = "cv_visual_element_detection"
    CV_SCREENSHOT_COMPARISON = "cv_screenshot_comparison"
    DATA_GEOCODING = "data_geocoding"
    DATA_ENRICHMENT = "data_enrichment"
    DATA_VALIDATION = "data_validation"
    DATA_NORMALIZATION = "data_normalization"


class EntityType(str, Enum):
    """Types of named entities."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"


class SentimentLabel(str, Enum):
    """Sentiment analysis labels."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class LanguageCode(str, Enum):
    """Supported language codes."""
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    IT = "it"
    PT = "pt"
    RU = "ru"
    ZH = "zh"
    JA = "ja"
    KO = "ko"
    AR = "ar"


class OCREngine(str, Enum):
    """Available OCR engines."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    AZURE_COGNITIVE = "azure_cognitive"
    GOOGLE_VISION = "google_vision"


class ImageClassificationModel(str, Enum):
    """Available image classification models."""
    RESNET = "resnet"
    VGG = "vgg"
    INCEPTION = "inception"
    MOBILENET = "mobilenet"
    EFFICIENTNET = "efficientnet"


class ExtractedEntity(BaseModel):
    """Represents an extracted named entity."""
    text: str = Field(..., description="The entity text")
    label: EntityType = Field(..., description="The entity type")
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    description: Optional[str] = Field(None, description="Entity description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SentimentResult(BaseModel):
    """Represents sentiment analysis results."""
    label: SentimentLabel = Field(..., description="Sentiment label")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    positive_score: float = Field(0.0, ge=0.0, le=1.0, description="Positive sentiment score")
    negative_score: float = Field(0.0, ge=0.0, le=1.0, description="Negative sentiment score")
    neutral_score: float = Field(0.0, ge=0.0, le=1.0, description="Neutral sentiment score")
    emotions: Dict[str, float] = Field(default_factory=dict, description="Emotion scores")


class TextSummary(BaseModel):
    """Represents text summarization results."""
    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original text length")
    summary_length: int = Field(..., description="Summary text length")
    compression_ratio: float = Field(..., description="Compression ratio")
    key_sentences: List[str] = Field(default_factory=list, description="Key sentences")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")


class LanguageDetectionResult(BaseModel):
    """Represents language detection results."""
    language: LanguageCode = Field(..., description="Detected language")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    alternatives: List[Dict[str, float]] = Field(default_factory=list, description="Alternative languages")


class TranslationResult(BaseModel):
    """Represents translation results."""
    translated_text: str = Field(..., description="Translated text")
    source_language: LanguageCode = Field(..., description="Source language")
    target_language: LanguageCode = Field(..., description="Target language")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Translation confidence")
    service: str = Field(..., description="Translation service used")


class OCRResult(BaseModel):
    """Represents OCR extraction results."""
    text: str = Field(..., description="Extracted text")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence")
    engine: OCREngine = Field(..., description="OCR engine used")
    bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list, description="Text bounding boxes")
    words: List[Dict[str, Any]] = Field(default_factory=list, description="Individual words")
    lines: List[Dict[str, Any]] = Field(default_factory=list, description="Text lines")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ImageClassificationResult(BaseModel):
    """Represents image classification results."""
    predictions: List[Dict[str, float]] = Field(..., description="Classification predictions")
    top_prediction: str = Field(..., description="Top prediction label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top prediction confidence")
    model: ImageClassificationModel = Field(..., description="Model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DetectedObject(BaseModel):
    """Represents a detected object in an image."""
    label: str = Field(..., description="Object label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bounding_box: Dict[str, float] = Field(..., description="Bounding box coordinates")
    area: float = Field(0.0, description="Object area")


class ObjectDetectionResult(BaseModel):
    """Represents object detection results."""
    objects: List[DetectedObject] = Field(..., description="Detected objects")
    total_objects: int = Field(..., description="Total number of objects")
    image_dimensions: Dict[str, int] = Field(..., description="Image dimensions")
    model: str = Field(..., description="Detection model used")
    processing_time: float = Field(0.0, description="Processing time in seconds")


class VisualElement(BaseModel):
    """Represents a detected visual element."""
    element_type: str = Field(..., description="Type of visual element")
    text: Optional[str] = Field(None, description="Text content if applicable")
    coordinates: Dict[str, float] = Field(..., description="Element coordinates")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Element attributes")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Detection confidence")


class VisualElementDetectionResult(BaseModel):
    """Represents visual element detection results."""
    elements: List[VisualElement] = Field(..., description="Detected visual elements")
    buttons: List[VisualElement] = Field(default_factory=list, description="Detected buttons")
    forms: List[VisualElement] = Field(default_factory=list, description="Detected forms")
    links: List[VisualElement] = Field(default_factory=list, description="Detected links")
    inputs: List[VisualElement] = Field(default_factory=list, description="Detected input fields")
    total_elements: int = Field(..., description="Total number of elements")


class ScreenshotComparison(BaseModel):
    """Represents screenshot comparison results."""
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    differences: List[Dict[str, Any]] = Field(default_factory=list, description="Detected differences")
    changed_regions: List[Dict[str, float]] = Field(default_factory=list, description="Changed regions")
    total_changes: int = Field(..., description="Total number of changes")
    change_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of change")


class GeolocationResult(BaseModel):
    """Represents geocoding results."""
    address: str = Field(..., description="Original address")
    formatted_address: str = Field(..., description="Formatted address")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    country: Optional[str] = Field(None, description="Country name")
    country_code: Optional[str] = Field(None, description="Country code")
    state: Optional[str] = Field(None, description="State/province")
    city: Optional[str] = Field(None, description="City name")
    postal_code: Optional[str] = Field(None, description="Postal code")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Geocoding confidence")
    service: str = Field(..., description="Geocoding service used")


class DataEnrichmentResult(BaseModel):
    """Represents data enrichment results."""
    original_data: Dict[str, Any] = Field(..., description="Original data")
    enriched_data: Dict[str, Any] = Field(..., description="Enriched data")
    enrichment_sources: List[str] = Field(..., description="Sources used for enrichment")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores")
    processing_time: float = Field(0.0, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ValidationResult(BaseModel):
    """Represents data validation results."""
    is_valid: bool = Field(..., description="Overall validation result")
    field_validations: Dict[str, bool] = Field(..., description="Per-field validation results")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Validation confidence")
    external_sources: List[str] = Field(default_factory=list, description="External validation sources")


class NormalizationResult(BaseModel):
    """Represents data normalization results."""
    original_value: Any = Field(..., description="Original value")
    normalized_value: Any = Field(..., description="Normalized value")
    normalization_type: str = Field(..., description="Type of normalization applied")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Normalization confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Normalization metadata")


class ProcessingRequest(BaseModel):
    """Represents a data processing request."""
    processing_type: ProcessingType = Field(..., description="Type of processing")
    data: Any = Field(..., description="Data to process")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Processing parameters")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")
    priority: int = Field(1, ge=1, le=10, description="Processing priority")
    timeout: int = Field(300, description="Processing timeout in seconds")


class ProcessingResponse(BaseModel):
    """Represents a data processing response."""
    request_id: str = Field(..., description="Request identifier")
    processing_type: ProcessingType = Field(..., description="Type of processing")
    status: str = Field(..., description="Processing status")
    result: Optional[Any] = Field(None, description="Processing result")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(0.0, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
