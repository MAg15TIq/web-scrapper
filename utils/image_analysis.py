"""
Enhanced image analysis utilities for Phase 5 computer vision enhancements.
"""
import cv2
import numpy as np
import base64
import io
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging

# Image processing libraries
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from skimage import measure, morphology, segmentation
    from skimage.metrics import structural_similarity as ssim
    from skimage.feature import local_binary_pattern, hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Advanced image analysis utilities."""
    
    def __init__(self):
        """Initialize the image analyzer."""
        self.supported_formats = ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']
        
        # UI element detection templates (basic patterns)
        self.ui_templates = {
            "button": self._create_button_template(),
            "input_field": self._create_input_template(),
            "checkbox": self._create_checkbox_template()
        }
    
    def _create_button_template(self) -> np.ndarray:
        """Create a basic button template for template matching."""
        if not CV2_AVAILABLE:
            return None
        
        # Create a simple rectangular button template
        template = np.zeros((30, 100, 3), dtype=np.uint8)
        cv2.rectangle(template, (2, 2), (98, 28), (200, 200, 200), -1)
        cv2.rectangle(template, (0, 0), (100, 30), (100, 100, 100), 2)
        return template
    
    def _create_input_template(self) -> np.ndarray:
        """Create a basic input field template."""
        if not CV2_AVAILABLE:
            return None
        
        # Create a simple input field template
        template = np.zeros((25, 150, 3), dtype=np.uint8)
        cv2.rectangle(template, (0, 0), (150, 25), (255, 255, 255), -1)
        cv2.rectangle(template, (0, 0), (150, 25), (100, 100, 100), 1)
        return template
    
    def _create_checkbox_template(self) -> np.ndarray:
        """Create a basic checkbox template."""
        if not CV2_AVAILABLE:
            return None
        
        # Create a simple checkbox template
        template = np.zeros((20, 20, 3), dtype=np.uint8)
        cv2.rectangle(template, (0, 0), (20, 20), (255, 255, 255), -1)
        cv2.rectangle(template, (0, 0), (20, 20), (100, 100, 100), 2)
        return template
    
    def decode_image(self, image_data: str) -> np.ndarray:
        """
        Decode base64 image data to OpenCV format.
        
        Args:
            image_data: Base64 encoded image
            
        Returns:
            OpenCV image array
        """
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            return image
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise
    
    def encode_image(self, image: np.ndarray, format: str = 'JPEG', quality: int = 95) -> str:
        """
        Encode OpenCV image to base64 string.
        
        Args:
            image: OpenCV image array
            format: Image format
            quality: JPEG quality (0-100)
            
        Returns:
            Base64 encoded image string
        """
        try:
            if format.upper() == 'JPEG':
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, buffer = cv2.imencode('.jpg', image, encode_param)
            elif format.upper() == 'PNG':
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                _, buffer = cv2.imencode('.png', image, encode_param)
            else:
                _, buffer = cv2.imencode('.jpg', image)
            
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def enhance_image(self, image: np.ndarray, enhancement_type: str = "auto") -> np.ndarray:
        """
        Enhance image quality for better processing.
        
        Args:
            image: Input image
            enhancement_type: Type of enhancement
            
        Returns:
            Enhanced image
        """
        if not CV2_AVAILABLE:
            return image
        
        try:
            enhanced = image.copy()
            
            if enhancement_type == "auto":
                # Auto enhancement pipeline
                enhanced = self._auto_enhance(enhanced)
                
            elif enhancement_type == "contrast":
                # Enhance contrast
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
                
            elif enhancement_type == "brightness":
                # Enhance brightness
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.0, beta=30)
                
            elif enhancement_type == "sharpening":
                # Apply sharpening filter
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                
            elif enhancement_type == "denoising":
                # Apply denoising
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
                
            elif enhancement_type == "histogram_equalization":
                # Apply histogram equalization
                if len(enhanced.shape) == 3:
                    # Convert to YUV and equalize Y channel
                    yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                else:
                    enhanced = cv2.equalizeHist(enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    def _auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply automatic image enhancement."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in auto enhancement: {e}")
            return image
    
    def detect_visual_elements(self, image: np.ndarray, element_types: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Detect visual UI elements in the image.
        
        Args:
            image: Input image
            element_types: Types of elements to detect
            
        Returns:
            Dictionary of detected elements by type
        """
        if not CV2_AVAILABLE:
            return {}
        
        element_types = element_types or ["button", "input_field", "text", "link"]
        detected_elements = {}
        
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect buttons using edge detection and contours
            if "button" in element_types:
                detected_elements["button"] = self._detect_buttons(image, gray)
            
            # Detect input fields
            if "input_field" in element_types:
                detected_elements["input_field"] = self._detect_input_fields(image, gray)
            
            # Detect text regions
            if "text" in element_types:
                detected_elements["text"] = self._detect_text_regions(image, gray)
            
            # Detect potential links (blue text)
            if "link" in element_types:
                detected_elements["link"] = self._detect_links(image)
            
            return detected_elements
            
        except Exception as e:
            logger.error(f"Error detecting visual elements: {e}")
            return {}
    
    def _detect_buttons(self, image: np.ndarray, gray: np.ndarray) -> List[Dict]:
        """Detect button-like elements."""
        buttons = []
        
        try:
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                if area < 500 or area > 50000:  # Filter by size
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (buttons are usually wider than tall)
                aspect_ratio = w / h
                if 0.5 <= aspect_ratio <= 5.0:
                    # Calculate rectangularity
                    rect_area = w * h
                    extent = area / rect_area
                    
                    if extent > 0.6:  # Reasonably rectangular
                        buttons.append({
                            "type": "button",
                            "coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                            "area": int(area),
                            "confidence": min(0.9, extent),
                            "attributes": {
                                "aspect_ratio": aspect_ratio,
                                "rectangularity": extent
                            }
                        })
            
            return buttons
            
        except Exception as e:
            logger.error(f"Error detecting buttons: {e}")
            return []
    
    def _detect_input_fields(self, image: np.ndarray, gray: np.ndarray) -> List[Dict]:
        """Detect input field elements."""
        input_fields = []
        
        try:
            # Apply morphological operations to find rectangular regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Apply edge detection
            edges = cv2.Canny(morph, 30, 100)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200 or area > 20000:  # Filter by size
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Input fields are usually wider than tall
                aspect_ratio = w / h
                if aspect_ratio >= 2.0:  # Wide rectangles
                    # Check if it's likely a text input (white/light background)
                    roi = gray[y:y+h, x:x+w]
                    mean_intensity = np.mean(roi)
                    
                    if mean_intensity > 200:  # Light background
                        input_fields.append({
                            "type": "input_field",
                            "coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                            "area": int(area),
                            "confidence": 0.7,
                            "attributes": {
                                "aspect_ratio": aspect_ratio,
                                "mean_intensity": float(mean_intensity)
                            }
                        })
            
            return input_fields
            
        except Exception as e:
            logger.error(f"Error detecting input fields: {e}")
            return []
    
    def _detect_text_regions(self, image: np.ndarray, gray: np.ndarray) -> List[Dict]:
        """Detect text regions in the image."""
        text_regions = []
        
        try:
            # Use MSER (Maximally Stable Extremal Regions) for text detection
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            for region in regions:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(region)
                
                # Filter by size and aspect ratio
                if w < 10 or h < 10 or w > 500 or h > 100:
                    continue
                
                aspect_ratio = w / h
                if aspect_ratio < 0.1 or aspect_ratio > 20:
                    continue
                
                text_regions.append({
                    "type": "text",
                    "coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "confidence": 0.6,
                    "attributes": {
                        "aspect_ratio": aspect_ratio,
                        "detection_method": "mser"
                    }
                })
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []
    
    def _detect_links(self, image: np.ndarray) -> List[Dict]:
        """Detect potential link elements (blue text)."""
        links = []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define blue color range for links
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Create mask for blue regions
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50 or area > 5000:  # Filter by size
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Links are usually text-like (wider than tall)
                aspect_ratio = w / h
                if aspect_ratio >= 1.5:
                    links.append({
                        "type": "link",
                        "coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "confidence": 0.5,
                        "attributes": {
                            "color": "blue",
                            "aspect_ratio": aspect_ratio,
                            "detection_method": "color_based"
                        }
                    })
            
            return links
            
        except Exception as e:
            logger.error(f"Error detecting links: {e}")
            return []
    
    def compare_screenshots(self, image1: np.ndarray, image2: np.ndarray, 
                          threshold: float = 0.95) -> Dict[str, Any]:
        """
        Compare two screenshots and detect differences.
        
        Args:
            image1: First image
            image2: Second image
            threshold: Similarity threshold
            
        Returns:
            Comparison results
        """
        if not SKIMAGE_AVAILABLE or not CV2_AVAILABLE:
            return {"error": "Required libraries not available"}
        
        try:
            # Resize images to same size if different
            if image1.shape != image2.shape:
                h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
                image1 = cv2.resize(image1, (w, h))
                image2 = cv2.resize(image2, (w, h))
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Calculate structural similarity
            similarity_score, diff_image = ssim(gray1, gray2, full=True)
            
            # Find differences
            diff_image = (diff_image * 255).astype(np.uint8)
            diff_binary = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Find contours of differences
            contours, _ = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            differences = []
            total_diff_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small differences
                    x, y, w, h = cv2.boundingRect(contour)
                    differences.append({
                        "coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "area": int(area)
                    })
                    total_diff_area += area
            
            # Calculate change percentage
            total_area = image1.shape[0] * image1.shape[1]
            change_percentage = (total_diff_area / total_area) * 100
            
            return {
                "similarity_score": float(similarity_score),
                "differences": differences,
                "total_changes": len(differences),
                "change_percentage": float(change_percentage),
                "changed_regions": differences,
                "is_similar": similarity_score >= threshold
            }
            
        except Exception as e:
            logger.error(f"Error comparing screenshots: {e}")
            return {"error": str(e)}
    
    def extract_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract various features from an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Basic properties
            features["dimensions"] = {"height": image.shape[0], "width": image.shape[1]}
            features["channels"] = image.shape[2] if len(image.shape) == 3 else 1
            
            # Color analysis
            if len(image.shape) == 3:
                features["color_analysis"] = self._analyze_colors(image)
            
            # Texture analysis
            if SKIMAGE_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                features["texture_analysis"] = self._analyze_texture(gray)
            
            # Edge analysis
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                features["edge_analysis"] = self._analyze_edges(gray)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return {}
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in the image."""
        try:
            # Calculate color histograms
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # Calculate dominant colors
            pixels = image.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # Get top 5 colors
            top_indices = np.argsort(counts)[-5:][::-1]
            dominant_colors = [
                {"color": unique_colors[i].tolist(), "count": int(counts[i])}
                for i in top_indices
            ]
            
            # Calculate average color
            avg_color = np.mean(pixels, axis=0).tolist()
            
            return {
                "dominant_colors": dominant_colors,
                "average_color": avg_color,
                "histogram_peaks": {
                    "blue": int(np.argmax(hist_b)),
                    "green": int(np.argmax(hist_g)),
                    "red": int(np.argmax(hist_r))
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing colors: {e}")
            return {}
    
    def _analyze_texture(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture features using Local Binary Patterns."""
        try:
            # Calculate Local Binary Pattern
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            
            # Calculate LBP histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Normalize
            
            # Calculate texture measures
            contrast = np.var(gray_image)
            homogeneity = np.sum(hist ** 2)
            
            return {
                "lbp_histogram": hist.tolist(),
                "contrast": float(contrast),
                "homogeneity": float(homogeneity),
                "texture_uniformity": float(np.sum(hist[:n_points]))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing texture: {e}")
            return {}
    
    def _analyze_edges(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Analyze edge features in the image."""
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_density = edge_pixels / total_pixels
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return {
                "edge_density": float(edge_density),
                "total_edge_pixels": int(edge_pixels),
                "contour_count": len(contours),
                "average_contour_area": float(np.mean([cv2.contourArea(c) for c in contours])) if contours else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing edges: {e}")
            return {}


# Global instance
image_analyzer = ImageAnalyzer()
