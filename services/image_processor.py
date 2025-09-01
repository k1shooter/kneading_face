"""
Image Processing Service for Facial Expression Transformation

This module provides comprehensive image preprocessing and postprocessing capabilities
for AI-powered facial expression modification, including validation, format conversion,
face detection, image optimization, and quality enhancement.
"""

import os
import io
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
import cv2
import numpy as np
from werkzeug.datastructures import FileStorage
import hashlib
import tempfile
import shutil
from flask import current_app

# Configure logging
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Comprehensive image processing service for facial expression transformation.
    
    Handles image validation, preprocessing, face detection, format conversion,
    optimization, and postprocessing operations required for AI model inference.
    """
    
    def __init__(self, config=None):
        """
        Initialize ImageProcessor with configuration settings.
        
        Args:
            config: Configuration object with image processing parameters
        """
        self.config = config
        self.supported_formats = {'JPEG', 'JPG', 'PNG', 'WEBP', 'BMP', 'TIFF'}
        self.output_formats = {'JPEG', 'PNG', 'WEBP'}
        
        # Default processing parameters
        self.max_file_size = getattr(config, 'MAX_CONTENT_LENGTH', 16 * 1024 * 1024)  # 16MB
        self.max_image_size = getattr(config, 'MAX_IMAGE_SIZE', (2048, 2048))
        self.min_image_size = getattr(config, 'MIN_IMAGE_SIZE', (128, 128))
        self.default_quality = getattr(config, 'IMAGE_QUALITY', 85)
        #self.temp_dir = getattr(config, 'TEMP_UPLOAD_DIR', tempfile.gettempdir())
        self.results_folder = current_app.config.get('RESULTS_FOLDER', 'static/uploads/results')
        self.output_dir = getattr(config, 'UPLOAD_FOLDER', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'uploads'))

        # Face detection setup
        self.face_cascade = None
        self._init_face_detection()
        
        logger.info("ImageProcessor initialized with config parameters")
    
    def _init_face_detection(self):
        """Initialize OpenCV face detection cascade."""
        try:
            # Try to load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Face detection cascade loaded successfully")
            else:
                logger.warning("Face detection cascade not found, face detection disabled")
        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
            self.face_cascade = None
    
    def validate_upload(self, file: FileStorage) -> Dict[str, Any]:
        """
        Validate uploaded image file for processing.
        
        Args:
            file: Uploaded file from Flask request
            
        Returns:
            Dictionary with validation results and metadata
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Check if file exists
            if not file or not file.filename:
                result['errors'].append("No file provided")
                return result
            
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > self.max_file_size:
                result['errors'].append(f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)")
                return result
            
            # Check file extension
            filename = file.filename.lower()
            file_ext = filename.split('.')[-1].upper() if '.' in filename else ''
            
            if file_ext not in self.supported_formats:
                result['errors'].append(f"Unsupported file format: {file_ext}")
                return result
            
            # Try to open and validate image
            try:
                image = Image.open(file.stream)
                image.verify()  # Verify image integrity
                file.seek(0)  # Reset stream position
                
                # Re-open for metadata extraction
                image = Image.open(file.stream)
                width, height = image.size
                
                # Check image dimensions
                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    result['errors'].append(f"Image too small: {width}x{height}, minimum: {self.min_image_size}")
                    return result
                
                if width > self.max_image_size[0] or height > self.max_image_size[1]:
                    result['warnings'].append(f"Large image will be resized: {width}x{height}")
                
                # Extract metadata
                result['metadata'] = {
                    'filename': file.filename,
                    'size_bytes': file_size,
                    'format': image.format,
                    'mode': image.mode,
                    'width': width,
                    'height': height,
                    'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
                }
                
                # Extract EXIF data if available
                if hasattr(image, '_getexif') and image._getexif():
                    exif_data = {}
                    for tag_id, value in image._getexif().items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
                    result['metadata']['exif'] = exif_data
                
                result['valid'] = True
                logger.info(f"Image validation successful: {filename} ({width}x{height})")
                
            except Exception as e:
                result['errors'].append(f"Invalid image file: {str(e)}")
                return result
            
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Image validation failed: {e}")
        
        finally:
            file.seek(0)  # Reset file position
        
        return result
    
    def preprocess_image(self, file: FileStorage, target_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Preprocess image for AI model inference.
        
        Args:
            file: Input image file
            target_size: Target dimensions for resizing (width, height)
            
        Returns:
            Dictionary with processed image data and metadata
        """
        result = {
            'success': False,
            'image_data': None,
            'image_array': None,
            'metadata': {},
            'temp_path': None,
            'errors': []
        }
        
        try:
            # Validate image first
            validation = self.validate_upload(file)
            if not validation['valid']:
                result['errors'] = validation['errors']
                return result
            
            # Load image
            image = Image.open(file.stream)
            original_size = image.size
            
            # Handle EXIF orientation
            image = self._fix_image_orientation(image)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Create white background for transparent images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Resize if target size specified or image is too large
            if target_size:
                image = self._smart_resize(image, target_size)
            elif image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image = self._smart_resize(image, self.max_image_size)
            
            # Enhance image quality
            image = self._enhance_image(image)
            
            # Convert to numpy array for model processing
            image_array = np.array(image)
            
            # Save temporary processed image
            temp_path = self._save_temp_image(image, file.filename)
            
            # Generate image hash for caching
            image_hash = self._generate_image_hash(image_array)
            
            result.update({
                'success': True,
                'image_data': image,
                'image_array': image_array,
                'temp_path': temp_path,
                'metadata': {
                    'original_size': original_size,
                    'processed_size': image.size,
                    'format': 'RGB',
                    'hash': image_hash,
                    'preprocessing_applied': True
                }
            })
            
            logger.info(f"Image preprocessing successful: {original_size} -> {image.size}")
            
        except Exception as e:
            result['errors'].append(f"Preprocessing error: {str(e)}")
            logger.error(f"Image preprocessing failed: {e}")
        
        return result
    
    def detect_faces(self, image: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect faces in the image using OpenCV.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            List of detected face regions with coordinates and confidence
        """
        faces = []
        
        if self.face_cascade is None:
            logger.warning("Face detection not available")
            return faces
        
        try:
            # Convert to numpy array if PIL Image
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            detected_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to list of dictionaries
            for (x, y, w, h) in detected_faces:
                faces.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'center_x': int(x + w // 2),
                    'center_y': int(y + h // 2),
                    'area': int(w * h)
                })
            
            logger.info(f"Detected {len(faces)} faces in image")
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
        
        return faces
    
    def postprocess_result(self, result_image: Union[Image.Image, np.ndarray], 
                          original_metadata: Dict[str, Any],
                          output_format: str = 'JPEG',
                          quality: Optional[int] = None) -> Dict[str, Any]:
        """
        Postprocess AI model result for final output.
        
        Args:
            result_image: Processed image from AI model
            original_metadata: Metadata from original image
            output_format: Desired output format
            quality: Output quality (1-100)
            
        Returns:
            Dictionary with final processed image and metadata
        """
        result = {
            'success': False,
            'image_data': None,
            'output_path': None,
            'metadata': {},
            'errors': []
        }
        
        try:
            # Convert numpy array to PIL Image if necessary
            if isinstance(result_image, np.ndarray):
                if result_image.dtype != np.uint8:
                    result_image = (result_image * 255).astype(np.uint8)
                image = Image.fromarray(result_image)
            else:
                image = result_image
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply final enhancements
            image = self._apply_final_enhancements(image)
            
            # Resize back to original dimensions if needed
            if 'original_size' in original_metadata:
                original_size = original_metadata['original_size']
                if image.size != original_size:
                    image = self._smart_resize(image, original_size)
            
            # Set output quality
            if quality is None:
                quality = self.default_quality
            
            # Validate output format
            if output_format.upper() not in self.output_formats:
                output_format = 'JPEG'
            
            # Save final image
            output_path = self._save_final_image(image, output_format, quality)
            
            result.update({
                'success': True,
                'image_data': image,
                'output_path': output_path,
                'metadata': {
                    'final_size': image.size,
                    'output_format': output_format,
                    'quality': quality,
                    'postprocessing_applied': True,
                    'file_size': os.path.getsize(output_path) if output_path else None
                }
            })
            
            logger.info(f"Image postprocessing successful: {image.size}, format: {output_format}")
            
        except Exception as e:
            result['errors'].append(f"Postprocessing error: {str(e)}")
            logger.error(f"Image postprocessing failed: {e}")
        
        return result
    
    def _fix_image_orientation(self, image: Image.Image) -> Image.Image:
        """Fix image orientation based on EXIF data."""
        try:
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                orientation = exif.get(274)  # Orientation tag
                
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except Exception as e:
            logger.warning(f"Could not fix image orientation: {e}")
        
        return image
    
    def _smart_resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calculate aspect ratios
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        
        if original_ratio > target_ratio:
            # Image is wider, fit to width
            new_width = target_width
            new_height = int(target_width / original_ratio)
        else:
            # Image is taller, fit to height
            new_height = target_height
            new_width = int(target_height * original_ratio)
        
        # Resize with high-quality resampling
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # If exact target size needed, pad with white background
        if (new_width, new_height) != target_size:
            padded = Image.new('RGB', target_size, (255, 255, 255))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            padded.paste(resized, (paste_x, paste_y))
            return padded
        
        return resized
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply basic image enhancements."""
        try:
            # Slight sharpening
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Slight contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
        
        return image
    
    def _apply_final_enhancements(self, image: Image.Image) -> Image.Image:
        """Apply final enhancements to processed image."""
        try:
            # Noise reduction
            image = image.filter(ImageFilter.SMOOTH_MORE)
            
            # Slight color enhancement
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)
            
        except Exception as e:
            logger.warning(f"Final enhancement failed: {e}")
        
        return image
    
    def _generate_image_hash(self, image_array: np.ndarray) -> str:
        """Generate hash for image caching."""
        return hashlib.md5(image_array.tobytes()).hexdigest()
    
    def _save_temp_image(self, image: Image.Image, original_filename: str) -> str:
        """Save temporary processed image."""
        try:
            # Generate unique filename
            base_name = os.path.splitext(original_filename)[0]
            temp_filename = f"{base_name}_processed_{hash(str(image.size))}.jpg"
            temp_path = os.path.join(self.temp_dir, temp_filename)
            
            # Ensure temp directory exists
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Save image
            image.save(temp_path, 'JPEG', quality=self.default_quality)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save temp image: {e}")
            return None
    
    def _save_final_image(self, image: Image.Image, output_format: str, quality: int) -> str:
        """Save final processed image."""
        try:
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            filename = f"result_{timestamp}.{output_format.lower()}"
            #output_path = os.path.join(self.output_dir, filename)
            output_path = os.path.join(self.results_folder, filename)
            
            # Ensure temp directory exists
            os.makedirs(self.results_folder, exist_ok=True)
            
            # Save with appropriate parameters
            if output_format.upper() == 'JPEG':
                image.save(output_path, 'JPEG', quality=quality, optimize=True)
            elif output_format.upper() == 'PNG':
                image.save(output_path, 'PNG', optimize=True)
            elif output_format.upper() == 'WEBP':
                image.save(output_path, 'WEBP', quality=quality, optimize=True)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save final image: {e}")
            return None
    
    def cleanup_temp_files(self, file_paths: List[str]) -> int:
        """
        Clean up temporary files.
        
        Args:
            file_paths: List of file paths to remove
            
        Returns:
            Number of files successfully removed
        """
        removed_count = 0
        
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    removed_count += 1
                    logger.debug(f"Removed temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")
        
        return removed_count
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get image processing statistics."""
        return {
            'supported_formats': list(self.supported_formats),
            'output_formats': list(self.output_formats),
            'max_file_size': self.max_file_size,
            'max_image_size': self.max_image_size,
            'min_image_size': self.min_image_size,
            'default_quality': self.default_quality,
            'face_detection_available': self.face_cascade is not None,
            'temp_directory': self.temp_dir
        }

# Import time for timestamp generation
import time