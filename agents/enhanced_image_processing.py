"""
Enhanced Image Processing Agent with advanced computer vision capabilities.
"""
import asyncio
import logging
import time
import base64
import io
import hashlib
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import numpy as np

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.data_processing import (
    ProcessingType, OCREngine, ImageClassificationModel,
    OCRResult, ImageClassificationResult, ObjectDetectionResult,
    VisualElementDetectionResult, ScreenshotComparison,
    DetectedObject, VisualElement
)
from config.data_processing_config import data_processing_config

# Image processing libraries
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Install with: pip install Pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Install with: pip install opencv-python")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available. Install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available. Install with: pip install easyocr")

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. Install with: pip install paddleocr")

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

try:
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available. Install with: pip install ultralytics")

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image not available. Install with: pip install scikit-image")


class EnhancedImageProcessingAgent(Agent):
    """
    Enhanced Image Processing Agent with advanced computer vision capabilities.
    """
    
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize the Enhanced Image Processing Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            coordinator_id: ID of the coordinator agent
        """
        super().__init__(agent_id=agent_id, agent_type="enhanced_image_processing", coordinator_id=coordinator_id)
        
        # Configuration
        self.config = data_processing_config.computer_vision
        
        # Initialize models and engines
        self.ocr_engines = {}
        self.classification_models = {}
        self.detection_models = {}
        self.pipelines = {}
        
        # Processing cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Performance metrics
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0
        }
        
        # Initialize capabilities
        self.capabilities = {
            "basic_processing": PIL_AVAILABLE,
            "advanced_processing": CV2_AVAILABLE,
            "tesseract_ocr": TESSERACT_AVAILABLE,
            "easyocr": EASYOCR_AVAILABLE,
            "paddleocr": PADDLEOCR_AVAILABLE,
            "object_detection": YOLO_AVAILABLE or TRANSFORMERS_AVAILABLE,
            "image_classification": TRANSFORMERS_AVAILABLE,
            "screenshot_comparison": SKIMAGE_AVAILABLE and CV2_AVAILABLE,
            "visual_element_detection": CV2_AVAILABLE
        }
        
        # Register message handlers
        self.register_handler("extract_text_ocr", self._handle_extract_text_ocr)
        self.register_handler("classify_image", self._handle_classify_image)
        self.register_handler("detect_objects", self._handle_detect_objects)
        self.register_handler("detect_visual_elements", self._handle_detect_visual_elements)
        self.register_handler("compare_screenshots", self._handle_compare_screenshots)
        self.register_handler("enhance_image", self._handle_enhance_image)
        self.register_handler("analyze_image_content", self._handle_analyze_image_content)
        self.register_handler("process_image_batch", self._handle_process_batch)
        
        # Create output directory
        self.output_dir = "output/enhanced_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models asynchronously
        asyncio.create_task(self._initialize_models())
        
        # Start periodic tasks
        asyncio.create_task(self._start_periodic_tasks())
    
    async def _initialize_models(self):
        """Initialize computer vision models and engines."""
        try:
            self.logger.info("Initializing computer vision models...")
            
            # Initialize OCR engines
            if TESSERACT_AVAILABLE:
                if self.config.tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_path
                self.ocr_engines["tesseract"] = pytesseract
                self.logger.info("Initialized Tesseract OCR")
            
            if EASYOCR_AVAILABLE:
                try:
                    self.ocr_engines["easyocr"] = easyocr.Reader(self.config.easyocr_languages)
                    self.logger.info("Initialized EasyOCR")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize EasyOCR: {e}")
            
            if PADDLEOCR_AVAILABLE:
                try:
                    self.ocr_engines["paddleocr"] = paddleocr.PaddleOCR(
                        use_angle_cls=True,
                        lang=self.config.paddleocr_language
                    )
                    self.logger.info("Initialized PaddleOCR")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize PaddleOCR: {e}")
            
            # Initialize object detection models
            if YOLO_AVAILABLE:
                try:
                    self.detection_models["yolo"] = YOLO(self.config.object_detection_model)
                    self.logger.info(f"Initialized YOLO model: {self.config.object_detection_model}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize YOLO: {e}")
            
            # Initialize transformer models
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Image classification
                    self.pipelines["image_classification"] = pipeline(
                        "image-classification",
                        model="google/vit-base-patch16-224"
                    )
                    
                    # Object detection
                    self.pipelines["object_detection"] = pipeline(
                        "object-detection",
                        model="facebook/detr-resnet-50"
                    )
                    
                    self.logger.info("Initialized transformer pipelines")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to initialize transformer models: {e}")
            
            self.logger.info("Computer vision models initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error initializing computer vision models: {e}")
    
    async def _start_periodic_tasks(self):
        """Start periodic maintenance tasks."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_cache()
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
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        if self.processing_stats["total_requests"] > 0:
            success_rate = (self.processing_stats["successful_requests"] / 
                          self.processing_stats["total_requests"]) * 100
            cache_hit_rate = (self.processing_stats["cache_hits"] / 
                            self.processing_stats["total_requests"]) * 100
            
            self.logger.info(
                f"Image Processing Agent Performance - Success Rate: {success_rate:.1f}%, "
                f"Cache Hit Rate: {cache_hit_rate:.1f}%, "
                f"Avg Processing Time: {self.processing_stats['average_processing_time']:.2f}s"
            )
    
    def _get_cache_key(self, image_data: str, operation: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key for image processing."""
        params_str = str(sorted(params.items())) if params else ""
        # Use hash of image data for cache key
        image_hash = hashlib.md5(image_data.encode()).hexdigest()[:16]
        content = f"{operation}:{image_hash}:{params_str}"
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
    
    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image data to PIL Image."""
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            if max(image.size) > self.config.max_image_size:
                image.thumbnail((self.config.max_image_size, self.config.max_image_size), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error decoding image: {e}")
            raise
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.config.image_quality)
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image: {e}")
            raise
