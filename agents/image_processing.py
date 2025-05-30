"""
Image Processing agent for the web scraping system.
"""
import logging
import asyncio
import base64
import io
import re
import os
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from urllib.parse import urljoin
from datetime import datetime, timedelta

from agents.base import Agent
from models.task import Task, TaskStatus, TaskType
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority

# Try to import image processing libraries, but provide fallbacks if not available
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ImageProcessingAgent(Agent):
    """
    Agent for processing and analyzing images from web pages.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new image processing agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        super().__init__(agent_id=agent_id, agent_type="image_processing", coordinator_id=coordinator_id)

        # Check available libraries
        self.capabilities = {
            "basic_processing": PIL_AVAILABLE,
            "advanced_processing": CV2_AVAILABLE,
            "ocr": TESSERACT_AVAILABLE,
            "transformers": TRANSFORMERS_AVAILABLE
        }

        # Log available capabilities
        self.logger.info(f"Image processing capabilities: {self.capabilities}")

        # Output directory for processed images
        self.output_dir = "output/images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize advanced models if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.object_detector = pipeline("object-detection")
                self.image_classifier = pipeline("image-classification")
                self.ocr_processor = pipeline("image-to-text")
                self.logger.info("Transformer models initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize transformer models: {str(e)}")
                self.capabilities["transformers"] = False

        # Cache for processed results
        self.processed_cache: Dict[str, Dict[str, Any]] = {}

        # Register additional message handlers
        self.register_handler("analyze_image", self._handle_analyze_image)
        self.register_handler("extract_text", self._handle_extract_text)
        self.register_handler("detect_objects", self._handle_detect_objects)
        self.register_handler("classify_image", self._handle_classify_image)
        self.register_handler("process_image", self._handle_process_image)

        # Start periodic tasks
        self._start_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the image processor agent."""
        asyncio.create_task(self._periodic_cache_cleanup())

    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up the processed cache."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                if not self.running:
                    break

                self._cleanup_cache()
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}", exc_info=True)

    def _cleanup_cache(self) -> None:
        """Clean up the processed cache."""
        current_time = time.time()
        self.processed_cache = {
            key: value
            for key, value in self.processed_cache.items()
            if current_time - value.get("timestamp", 0) < 86400  # 24 hours
        }

    async def _handle_analyze_image(self, message: Message) -> None:
        """
        Handle an image analysis request.

        Args:
            message: The message containing image analysis parameters.
        """
        if not hasattr(message, "image_data"):
            self.logger.warning("Received analyze_image message without image data")
            return

        try:
            result = await self.analyze_image(
                message.image_data,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="image_analysis_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_extract_text(self, message: Message) -> None:
        """
        Handle a text extraction request.

        Args:
            message: The message containing text extraction parameters.
        """
        if not hasattr(message, "image_data"):
            self.logger.warning("Received extract_text message without image data")
            return

        try:
            result = await self.extract_text(
                message.image_data,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="text_extraction_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_detect_objects(self, message: Message) -> None:
        """
        Handle an object detection request.

        Args:
            message: The message containing object detection parameters.
        """
        if not hasattr(message, "image_data"):
            self.logger.warning("Received detect_objects message without image data")
            return

        try:
            result = await self.detect_objects(
                message.image_data,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error detecting objects: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="object_detection_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_classify_image(self, message: Message) -> None:
        """
        Handle an image classification request.

        Args:
            message: The message containing image classification parameters.
        """
        if not hasattr(message, "image_data"):
            self.logger.warning("Received classify_image message without image data")
            return

        try:
            result = await self.classify_image(
                message.image_data,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error classifying image: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="image_classification_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_process_image(self, message: Message) -> None:
        """
        Handle an image processing request.

        Args:
            message: The message containing image processing parameters.
        """
        if not hasattr(message, "image_data"):
            self.logger.warning("Received process_image message without image data")
            return

        try:
            result = await self.process_image(
                message.image_data,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="image_processing_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def analyze_image(self, image_data: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Analyze an image using various techniques.

        Args:
            image_data: Base64-encoded image data.
            options: Analysis options.

        Returns:
            A dictionary containing analysis results.
        """
        # Check cache
        cache_key = hashlib.md5(image_data.encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Decode image
        image = self._decode_image(image_data)

        # Get image properties
        properties = self._get_image_properties(image)

        # Perform basic analysis
        analysis = self._analyze_image_content(image)

        # Store result
        result = {
            "properties": properties,
            "analysis": analysis
        }

        self.processed_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        return result

    async def extract_text(self, image_data: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Extract text from an image.

        Args:
            image_data: Base64-encoded image data.
            options: Text extraction options.

        Returns:
            A dictionary containing extracted text.
        """
        # Check cache
        cache_key = hashlib.md5(image_data.encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Decode image
        image = self._decode_image(image_data)

        # Preprocess image
        processed_image = self._preprocess_image(image, options)

        # Extract text using OCR
        result = {}

        # Use transformer OCR if available
        if TRANSFORMERS_AVAILABLE and hasattr(self, 'ocr_processor'):
            try:
                text = self.ocr_processor(processed_image)[0]["generated_text"]
                result["text"] = text
            except Exception as e:
                self.logger.warning(f"Error using transformer OCR: {str(e)}")

        # Use Tesseract OCR if available
        if TESSERACT_AVAILABLE:
            try:
                tesseract_text = pytesseract.image_to_string(processed_image)
                result["tesseract_text"] = tesseract_text

                # If transformer OCR failed, use Tesseract as primary
                if "text" not in result:
                    result["text"] = tesseract_text
            except Exception as e:
                self.logger.warning(f"Error using Tesseract OCR: {str(e)}")

        # If no OCR method worked, return error
        if "text" not in result:
            return {
                "success": False,
                "error": "No OCR method available or all methods failed"
            }

        # Calculate confidence if both methods available
        if "tesseract_text" in result and "text" in result:
            result["confidence"] = self._calculate_text_similarity(result["text"], result["tesseract_text"])

        # Add success flag
        result["success"] = True

        # Store in cache
        self.processed_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        return result

    async def detect_objects(self, image_data: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Detect objects in an image.

        Args:
            image_data: Base64-encoded image data.
            options: Object detection options.

        Returns:
            A dictionary containing detected objects.
        """
        # Check cache
        cache_key = hashlib.md5(image_data.encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Decode image
        image = self._decode_image(image_data)

        # Check if transformer object detection is available
        if not (TRANSFORMERS_AVAILABLE and hasattr(self, 'object_detector')):
            return {
                "success": False,
                "error": "Object detection requires transformer models"
            }

        try:
            # Detect objects
            detections = self.object_detector(image)

            # Process detections
            objects = []
            for detection in detections:
                objects.append({
                    "label": detection["label"],
                    "score": detection["score"],
                    "box": detection["box"]
                })

            # Group objects by label
            grouped_objects = {}
            for obj in objects:
                label = obj["label"]
                if label not in grouped_objects:
                    grouped_objects[label] = []
                grouped_objects[label].append(obj)

            # Store result
            result = {
                "success": True,
                "objects": objects,
                "grouped_objects": grouped_objects,
                "object_types": list(grouped_objects.keys())
            }

            self.processed_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }

            return result
        except Exception as e:
            self.logger.error(f"Error detecting objects: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def classify_image(self, image_data: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Classify an image.

        Args:
            image_data: Base64-encoded image data.
            options: Classification options.

        Returns:
            A dictionary containing classification results.
        """
        # Check cache
        cache_key = hashlib.md5(image_data.encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Decode image
        image = self._decode_image(image_data)

        # Try transformer-based classification first
        if TRANSFORMERS_AVAILABLE and hasattr(self, 'image_classifier'):
            try:
                # Classify image
                classifications = self.image_classifier(image)

                # Process classifications
                results = []
                for classification in classifications:
                    results.append({
                        "label": classification["label"],
                        "score": classification["score"]
                    })

                # Store result
                result = {
                    "success": True,
                    "classifications": results,
                    "top_classification": results[0] if results else None,
                    "method": "transformer"
                }

                self.processed_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }

                return result
            except Exception as e:
                self.logger.warning(f"Transformer classification failed: {str(e)}")

        # Fall back to OpenCV-based classification
        if CV2_AVAILABLE:
            try:
                # Convert PIL image to OpenCV format if needed
                if isinstance(image, Image.Image):
                    cv_image = np.array(image)
                    if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                else:
                    cv_image = image

                # Basic image analysis
                height, width = cv_image.shape[:2]
                channels = cv_image.shape[2] if len(cv_image.shape) > 2 else 1

                # Calculate color histogram
                color_hist = {}
                if channels == 3:
                    for i, color in enumerate(["blue", "green", "red"]):
                        hist = cv2.calcHist([cv_image], [i], None, [256], [0, 256])
                        color_hist[color] = hist.flatten().tolist()

                # Calculate average color
                avg_color = cv_image.mean(axis=0).mean(axis=0).tolist() if channels == 3 else [cv_image.mean()]

                # Detect if image is mostly text
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if channels == 3 else cv_image
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                text_ratio = np.count_nonzero(binary) / (height * width)
                is_text = text_ratio > 0.05

                # Detect if image is a photo or graphic
                edges = cv2.Canny(gray, 100, 200)
                edge_ratio = np.count_nonzero(edges) / (height * width)
                is_photo = edge_ratio < 0.1

                # Simple classification based on analysis
                image_type = "photo" if is_photo else "graphic"
                if is_text:
                    image_type = "text_" + image_type

                return {
                    "success": True,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "avg_color": avg_color,
                    "text_ratio": float(text_ratio),
                    "edge_ratio": float(edge_ratio),
                    "classification": {
                        "type": image_type,
                        "is_text": bool(is_text),
                        "is_photo": bool(is_photo)
                    },
                    "method": "opencv"
                }
            except Exception as e:
                self.logger.error(f"Error classifying image with OpenCV: {str(e)}")

        return {
            "success": False,
            "error": "No classification method available"
        }

    async def process_image(self, image_data: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Process an image according to options.

        Args:
            image_data: Base64-encoded image data.
            options: Processing options.

        Returns:
            A dictionary containing processing results.
        """
        # Check cache
        cache_key = hashlib.md5((image_data + json.dumps(options, sort_keys=True)).encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Decode image
        image = self._decode_image(image_data)

        # Apply processing options
        processed_image = image.copy()

        if "resize" in options:
            processed_image = self._resize_image(processed_image, options["resize"])

        if "rotate" in options:
            processed_image = self._rotate_image(processed_image, options["rotate"])

        if "filter" in options:
            processed_image = self._apply_filter(processed_image, options["filter"])

        if "enhance" in options:
            processed_image = self._enhance_image(processed_image, options["enhance"])

        # Encode processed image
        processed_data = self._encode_image(processed_image)

        # Store result
        result = {
            "success": True,
            "processed_image": processed_data,
            "original_size": image.size if hasattr(image, 'size') else None,
            "processed_size": processed_image.size if hasattr(processed_image, 'size') else None
        }

        self.processed_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        return result

    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image data to PIL Image."""
        if image_data.startswith('data:image'):
            # Extract base64 data from data URL
            image_data = image_data.split(',', 1)[1]

        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes))

    def _encode_image(self, image: Image.Image, format: str = "JPEG") -> str:
        """Encode PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _get_image_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Get basic properties of an image."""
        return {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "aspect_ratio": image.width / image.height if image.height > 0 else 0
        }

    def _analyze_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """Perform basic analysis of image content."""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get color histogram
        histogram = image.histogram()

        # Calculate average color
        avg_color = [0, 0, 0]
        pixels = list(image.getdata())
        for pixel in pixels:
            avg_color[0] += pixel[0]
            avg_color[1] += pixel[1]
            avg_color[2] += pixel[2]

        num_pixels = len(pixels)
        if num_pixels > 0:
            avg_color = [c / num_pixels for c in avg_color]

        # Determine if image is mostly dark or light
        brightness = sum(avg_color) / 3
        is_dark = brightness < 128

        return {
            "histogram": histogram,
            "avg_color": avg_color,
            "brightness": brightness,
            "is_dark": is_dark
        }

    def _preprocess_image(self, image: Image.Image, options: Dict[str, Any]) -> Image.Image:
        """Preprocess image for OCR or other analysis."""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply preprocessing options
        processed = image

        # Resize if needed
        if "resize" in options:
            processed = self._resize_image(processed, options["resize"])

        # Convert to grayscale for OCR
        if options.get("grayscale", True):
            processed = processed.convert("L")

        # Apply OpenCV preprocessing if available
        if CV2_AVAILABLE and options.get("enhance_ocr", True):
            # Convert PIL to OpenCV
            cv_image = np.array(processed)

            # Apply thresholding
            _, binary = cv2.threshold(cv_image, 150, 255, cv2.THRESH_BINARY)

            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

            # Convert back to PIL
            processed = Image.fromarray(denoised)

        return processed

    def _resize_image(self, image: Image.Image, resize_options: Dict[str, Any]) -> Image.Image:
        """Resize an image according to options."""
        if isinstance(resize_options, dict):
            width = resize_options.get("width")
            height = resize_options.get("height")

            if width and height:
                return image.resize((width, height), Image.LANCZOS)
            elif width:
                ratio = width / image.width
                return image.resize((width, int(image.height * ratio)), Image.LANCZOS)
            elif height:
                ratio = height / image.height
                return image.resize((int(image.width * ratio), height), Image.LANCZOS)

        return image

    def _rotate_image(self, image: Image.Image, angle: float) -> Image.Image:
        """Rotate an image by the specified angle."""
        return image.rotate(angle, Image.BICUBIC, expand=True)

    def _apply_filter(self, image: Image.Image, filter_name: str) -> Image.Image:
        """Apply a filter to an image."""
        if filter_name == "grayscale":
            return image.convert("L")
        elif filter_name == "sepia":
            # Convert to grayscale first
            gray = image.convert("L")
            # Apply sepia tone
            return Image.merge("RGB", [
                gray.point(lambda x: min(255, x * 1.1)),
                gray.point(lambda x: min(255, x * 0.9)),
                gray.point(lambda x: min(255, x * 0.7))
            ])
        elif filter_name == "negative":
            return Image.eval(image, lambda x: 255 - x)

        return image

    def _enhance_image(self, image: Image.Image, enhance_options: Dict[str, Any]) -> Image.Image:
        """Enhance an image according to options."""
        from PIL import ImageEnhance

        enhanced = image

        if "contrast" in enhance_options:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(enhance_options["contrast"])

        if "brightness" in enhance_options:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(enhance_options["brightness"])

        if "sharpness" in enhance_options:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(enhance_options["sharpness"])

        return enhanced

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Simple similarity based on character overlap
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        # Calculate character overlap
        common_chars = set(text1) & set(text2)
        all_chars = set(text1) | set(text2)

        if not all_chars:
            return 0.0

        return len(common_chars) / len(all_chars)

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        self.logger.info(f"Executing task: {task.type}")

        if task.type == TaskType.IMAGE_DOWNLOAD:
            return await self._execute_image_download(task)
        elif task.type == TaskType.IMAGE_OCR:
            return await self._execute_image_ocr(task)
        elif task.type == TaskType.IMAGE_CLASSIFICATION:
            return await self._execute_image_classification(task)
        elif task.type == TaskType.IMAGE_EXTRACTION:
            return await self._execute_image_extraction(task)
        elif task.type == TaskType.IMAGE_COMPARISON:
            return await self._execute_image_comparison(task)
        else:
            raise ValueError(f"Unsupported task type: {task.type}")

    async def _execute_image_download(self, task: Task) -> Dict[str, Any]:
        """
        Execute an image download task.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the downloaded image information.
        """
        params = task.parameters

        # Get required parameters
        image_url = params.get("url")
        base_url = params.get("base_url", "")
        save_path = params.get("save_path")

        if not image_url:
            raise ValueError("Missing required parameter: url")

        # Resolve relative URLs
        if base_url and not image_url.startswith(("http://", "https://")):
            image_url = urljoin(base_url, image_url)

        # Generate save path if not provided
        if not save_path:
            image_name = os.path.basename(image_url).split("?")[0]
            save_path = os.path.join(self.output_dir, image_name)

        try:
            # Download the image
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()

                # Save the image
                with open(save_path, "wb") as f:
                    f.write(response.content)

                # Get image metadata if PIL is available
                metadata = {}
                if PIL_AVAILABLE:
                    try:
                        with Image.open(io.BytesIO(response.content)) as img:
                            metadata = {
                                "format": img.format,
                                "mode": img.mode,
                                "width": img.width,
                                "height": img.height,
                                "size_kb": len(response.content) // 1024
                            }
                    except Exception as e:
                        self.logger.warning(f"Error getting image metadata: {str(e)}")

                return {
                    "success": True,
                    "url": image_url,
                    "save_path": save_path,
                    "content_type": response.headers.get("content-type", ""),
                    "size_bytes": len(response.content),
                    "metadata": metadata
                }

        except Exception as e:
            self.logger.error(f"Error downloading image: {str(e)}")
            return {
                "success": False,
                "url": image_url,
                "error": str(e)
            }

    async def _execute_image_ocr(self, task: Task) -> Dict[str, Any]:
        """
        Execute an OCR task on an image.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the OCR results.
        """
        params = task.parameters

        # Get required parameters
        image_path = params.get("image_path")
        language = params.get("language", "eng")

        if not image_path:
            raise ValueError("Missing required parameter: image_path")

        if not TESSERACT_AVAILABLE:
            return {
                "success": False,
                "error": "Tesseract OCR is not available"
            }

        try:
            # Open the image
            if PIL_AVAILABLE:
                image = Image.open(image_path)

                # Preprocess the image if CV2 is available
                if CV2_AVAILABLE and params.get("preprocess", True):
                    # Convert PIL Image to OpenCV format
                    cv_image = np.array(image)

                    # Convert to grayscale
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

                    # Apply thresholding
                    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

                    # Noise removal
                    denoised = cv2.medianBlur(binary, 3)

                    # Convert back to PIL Image
                    image = Image.fromarray(denoised)

                # Perform OCR
                text = pytesseract.image_to_string(image, lang=language)

                # Get bounding boxes for text regions
                boxes = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)

                # Extract structured data
                structured_data = []
                for i in range(len(boxes["text"])):
                    if int(boxes["conf"][i]) > 60 and boxes["text"][i].strip():
                        structured_data.append({
                            "text": boxes["text"][i],
                            "confidence": int(boxes["conf"][i]),
                            "x": boxes["left"][i],
                            "y": boxes["top"][i],
                            "width": boxes["width"][i],
                            "height": boxes["height"][i]
                        })

                return {
                    "success": True,
                    "text": text,
                    "structured_data": structured_data,
                    "language": language
                }
            else:
                # Fallback to direct pytesseract call
                text = pytesseract.image_to_string(image_path, lang=language)

                return {
                    "success": True,
                    "text": text,
                    "language": language
                }

        except Exception as e:
            self.logger.error(f"Error performing OCR: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_image_classification(self, task: Task) -> Dict[str, Any]:
        """
        Execute an image classification task.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the classification results.
        """
        params = task.parameters

        # Get required parameters
        image_path = params.get("image_path")

        if not image_path:
            raise ValueError("Missing required parameter: image_path")

        if not CV2_AVAILABLE:
            return {
                "success": False,
                "error": "OpenCV is not available for image classification"
            }

        try:
            # Load the image
            image = cv2.imread(image_path)

            # Basic image analysis
            height, width, channels = image.shape

            # Calculate color histogram
            color_hist = {}
            for i, color in enumerate(["blue", "green", "red"]):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                color_hist[color] = hist.flatten().tolist()

            # Calculate average color
            avg_color = image.mean(axis=0).mean(axis=0).tolist()

            # Detect if image is mostly text
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            text_ratio = np.count_nonzero(binary) / (height * width)
            is_text = text_ratio > 0.05

            # Detect if image is a photo or graphic
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.count_nonzero(edges) / (height * width)
            is_photo = edge_ratio < 0.1

            # Simple classification based on analysis
            image_type = "photo" if is_photo else "graphic"
            if is_text:
                image_type = "text_" + image_type

            return {
                "success": True,
                "width": width,
                "height": height,
                "channels": channels,
                "avg_color": avg_color,
                "text_ratio": float(text_ratio),
                "edge_ratio": float(edge_ratio),
                "classification": {
                    "type": image_type,
                    "is_text": bool(is_text),
                    "is_photo": bool(is_photo)
                }
            }

        except Exception as e:
            self.logger.error(f"Error classifying image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_image_extraction(self, task: Task) -> Dict[str, Any]:
        """
        Execute an image extraction task from HTML content.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the extracted images.
        """
        params = task.parameters

        # Get required parameters
        html_content = params.get("html_content")
        base_url = params.get("base_url", "")
        download = params.get("download", False)

        if not html_content:
            raise ValueError("Missing required parameter: html_content")

        try:
            # Extract image URLs from HTML
            img_pattern = r'<img[^>]+src=["\'](.*?)["\']'
            img_urls = re.findall(img_pattern, html_content)

            # Extract background images
            bg_pattern = r'background-image:\s*url\(["\']?(.*?)["\']?\)'
            bg_urls = re.findall(bg_pattern, html_content)

            # Combine all image URLs
            all_urls = img_urls + bg_urls

            # Resolve relative URLs
            resolved_urls = []
            for url in all_urls:
                if not url.startswith(("http://", "https://", "data:")):
                    resolved_url = urljoin(base_url, url)
                else:
                    resolved_url = url
                resolved_urls.append(resolved_url)

            # Filter out data URLs
            filtered_urls = [url for url in resolved_urls if not url.startswith("data:")]

            # Download images if requested
            downloaded_images = []
            if download:
                for url in filtered_urls:
                    download_task = Task(
                        type=TaskType.IMAGE_DOWNLOAD,
                        parameters={"url": url, "base_url": base_url}
                    )
                    result = await self._execute_image_download(download_task)
                    if result.get("success", False):
                        downloaded_images.append(result)

            return {
                "success": True,
                "image_urls": filtered_urls,
                "count": len(filtered_urls),
                "downloaded_images": downloaded_images if download else []
            }

        except Exception as e:
            self.logger.error(f"Error extracting images: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_image_comparison(self, task: Task) -> Dict[str, Any]:
        """
        Execute an image comparison task.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the comparison results.
        """
        params = task.parameters

        # Get required parameters
        image_path1 = params.get("image_path1")
        image_path2 = params.get("image_path2")

        if not image_path1 or not image_path2:
            raise ValueError("Missing required parameters: image_path1 and image_path2")

        if not CV2_AVAILABLE:
            return {
                "success": False,
                "error": "OpenCV is not available for image comparison"
            }

        try:
            # Load the images
            img1 = cv2.imread(image_path1)
            img2 = cv2.imread(image_path2)

            # Resize images to the same dimensions
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])
            img1_resized = cv2.resize(img1, (width, height))
            img2_resized = cv2.resize(img2, (width, height))

            # Convert to grayscale
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

            # Calculate histogram similarity
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # Calculate structural similarity
            from skimage.metrics import structural_similarity as ssim
            similarity_index, _ = ssim(gray1, gray2, full=True)

            # Calculate mean squared error
            mse = np.mean((img1_resized - img2_resized) ** 2)

            # Calculate feature matching similarity
            # Initialize ORB detector
            orb = cv2.ORB_create()

            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)

            # Create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)

                # Sort them in order of distance
                matches = sorted(matches, key=lambda x: x.distance)

                # Calculate feature similarity
                feature_similarity = len(matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
            else:
                feature_similarity = 0

            return {
                "success": True,
                "histogram_similarity": float(hist_similarity),
                "structural_similarity": float(similarity_index),
                "mean_squared_error": float(mse),
                "feature_similarity": float(feature_similarity),
                "overall_similarity": float((hist_similarity + similarity_index + feature_similarity) / 3)
            }

        except Exception as e:
            self.logger.error(f"Error comparing images: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
