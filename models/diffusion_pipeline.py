"""
Diffusion Pipeline Implementation for Facial Expression Transformation

This module provides the core diffusion model pipeline for AI-powered facial expression
modification. It implements a custom diffusion pipeline optimized for facial expression
transformation with support for multiple model architectures and inference optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from PIL import Image
import logging
from dataclasses import dataclass
from enum import Enum
import os
import gc
import time
from contextlib import contextmanager

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    import cv2
except ImportError as e:
    logging.error(f"Required packages not installed: {e}")
    raise ImportError("Please install required packages: diffusers, transformers, opencv-python")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineType(Enum):
    """Supported diffusion pipeline types"""
    TEXT_TO_IMAGE = "text2img"
    IMAGE_TO_IMAGE = "img2img"
    INPAINTING = "inpaint"
    CONTROLNET = "controlnet"


class SchedulerType(Enum):
    """Supported scheduler types for diffusion sampling"""
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    EULER = "euler"
    LMS = "lms"
    PNDM = "pndm"


@dataclass
class PipelineConfig:
    """Configuration for diffusion pipeline"""
    model_id: str = "runwayml/stable-diffusion-v1-5"
    pipeline_type: PipelineType = PipelineType.IMAGE_TO_IMAGE
    scheduler_type: SchedulerType = SchedulerType.DDIM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    enable_memory_efficient_attention: bool = True
    enable_cpu_offload: bool = False
    enable_model_cpu_offload: bool = False
    safety_checker: bool = False
    requires_safety_checker: bool = False
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    use_auth_token: Optional[str] = None


class FacialExpressionPipeline:
    """
    Custom diffusion pipeline for facial expression transformation
    
    This class implements a specialized diffusion pipeline optimized for
    facial expression modification with advanced features like face detection,
    expression conditioning, and memory optimization.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the facial expression diffusion pipeline
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config or PipelineConfig()
        self.pipeline = None
        self.scheduler = None
        self.face_detector = None
        self.is_loaded = False
        self.device = self.config.device
        self.dtype = self.config.dtype
        
        # Performance tracking
        self.inference_times = []
        self.memory_usage = []
        
        # Expression prompts mapping
        self.expression_prompts = {
            "happy": "a person with a bright, joyful smile, happy expression, cheerful face",
            "sad": "a person with a sad, melancholic expression, downturned mouth, sorrowful face",
            "angry": "a person with an angry, fierce expression, furrowed brow, intense gaze",
            "surprised": "a person with a surprised, shocked expression, wide eyes, open mouth",
            "fearful": "a person with a fearful, scared expression, wide eyes, tense face",
            "disgusted": "a person with a disgusted, repulsed expression, wrinkled nose, grimace",
            "neutral": "a person with a neutral, calm expression, relaxed face, natural look",
            "excited": "a person with an excited, enthusiastic expression, bright eyes, animated face",
            "confused": "a person with a confused, puzzled expression, raised eyebrows, questioning look",
            "contempt": "a person with a contemptuous, disdainful expression, slight smirk, superior look"
        }
        
        # Negative prompts for better quality
        self.negative_prompts = [
            "blurry", "low quality", "distorted", "deformed", "ugly", "bad anatomy",
            "bad proportions", "extra limbs", "cloned face", "malformed limbs",
            "missing arms", "missing legs", "extra arms", "extra legs", "fused fingers",
            "too many fingers", "long neck", "cross-eyed", "mutated hands", "poorly drawn hands",
            "poorly drawn face", "mutation", "deformed", "blurry", "bad anatomy",
            "bad proportions", "extra limbs", "disfigured", "out of frame", "ugly",
            "extra limbs", "bad anatomy", "gross proportions", "malformed limbs"
        ]
        
        logger.info(f"Initialized FacialExpressionPipeline with device: {self.device}")
    
    def load_pipeline(self, force_reload: bool = False) -> bool:
        """
        Load the diffusion pipeline and associated models
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            bool: Success status
        """
        if self.is_loaded and not force_reload:
            logger.info("Pipeline already loaded")
            return True
        
        try:
            logger.info(f"Loading diffusion pipeline: {self.config.model_id}")
            start_time = time.time()
            
            # Load appropriate pipeline based on type
            if self.config.pipeline_type == PipelineType.IMAGE_TO_IMAGE:
                self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.config.model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None if not self.config.safety_checker else "default",
                    requires_safety_checker=self.config.requires_safety_checker,
                    cache_dir=self.config.cache_dir,
                    local_files_only=self.config.local_files_only,
                    use_auth_token=self.config.use_auth_token
                )
            elif self.config.pipeline_type == PipelineType.INPAINTING:
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.config.model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None if not self.config.safety_checker else "default",
                    requires_safety_checker=self.config.requires_safety_checker,
                    cache_dir=self.config.cache_dir,
                    local_files_only=self.config.local_files_only,
                    use_auth_token=self.config.use_auth_token
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.config.model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None if not self.config.safety_checker else "default",
                    requires_safety_checker=self.config.requires_safety_checker,
                    cache_dir=self.config.cache_dir,
                    local_files_only=self.config.local_files_only,
                    use_auth_token=self.config.use_auth_token
                )
            
            # Configure scheduler
            self._setup_scheduler()
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory optimizations
            self._enable_optimizations()
            
            # Load face detector
            self._load_face_detector()
            
            self.is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Pipeline loaded successfully in {load_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return False
    
    def _setup_scheduler(self):
        """Setup the diffusion scheduler"""
        try:
            if self.config.scheduler_type == SchedulerType.DDIM:
                self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            elif self.config.scheduler_type == SchedulerType.DPM_SOLVER:
                self.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            elif self.config.scheduler_type == SchedulerType.EULER:
                self.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
            elif self.config.scheduler_type == SchedulerType.LMS:
                self.scheduler = LMSDiscreteScheduler.from_config(self.pipeline.scheduler.config)
            elif self.config.scheduler_type == SchedulerType.PNDM:
                self.scheduler = PNDMScheduler.from_config(self.pipeline.scheduler.config)
            else:
                logger.warning(f"Unknown scheduler type: {self.config.scheduler_type}, using DDIM")
                self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            
            self.pipeline.scheduler = self.scheduler
            logger.info(f"Scheduler configured: {self.config.scheduler_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to setup scheduler: {e}")
            # Fallback to default scheduler
            pass
    
    def _enable_optimizations(self):
        """Enable memory and performance optimizations"""
        try:
            # Enable memory efficient attention
            if self.config.enable_memory_efficient_attention:
                if hasattr(self.pipeline, "enable_memory_efficient_attention"):
                    self.pipeline.enable_memory_efficient_attention()
                    logger.info("Memory efficient attention enabled")
            
            # Enable CPU offload
            if self.config.enable_cpu_offload:
                if hasattr(self.pipeline, "enable_sequential_cpu_offload"):
                    self.pipeline.enable_sequential_cpu_offload()
                    logger.info("Sequential CPU offload enabled")
            
            # Enable model CPU offload
            if self.config.enable_model_cpu_offload:
                if hasattr(self.pipeline, "enable_model_cpu_offload"):
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("Model CPU offload enabled")
            
            # Enable attention slicing for lower memory usage
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
                logger.info("Attention slicing enabled")
            
        except Exception as e:
            logger.warning(f"Some optimizations failed to enable: {e}")
    
    def _load_face_detector(self):
        """Load OpenCV face detector for face-aware processing"""
        try:
            # Load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            
            if self.face_detector.empty():
                logger.warning("Failed to load face detector")
                self.face_detector = None
            else:
                logger.info("Face detector loaded successfully")
                
        except Exception as e:
            logger.warning(f"Failed to load face detector: {e}")
            self.face_detector = None
    
    def detect_faces(self, image: Union[Image.Image, np.ndarray]) -> List[Dict]:
        """
        Detect faces in the input image
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            List of face detection results
        """
        if self.face_detector is None:
            return []
        
        try:
            # Convert to numpy array if PIL Image
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Convert to grayscale for face detection
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to list of dictionaries
            face_results = []
            for (x, y, w, h) in faces:
                face_results.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': 1.0  # Haar cascades don't provide confidence scores
                })
            
            logger.info(f"Detected {len(face_results)} faces")
            return face_results
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    @contextmanager
    def _memory_management(self):
        """Context manager for memory management during inference"""
        try:
            # Clear cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            yield
            
        finally:
            # Clean up after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def generate_expression(
        self,
        image: Union[Image.Image, np.ndarray],
        target_expression: str,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
        face_focus: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate facial expression transformation
        
        Args:
            image: Input image
            target_expression: Target expression type
            strength: Transformation strength (0.0 to 1.0)
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            face_focus: Whether to focus on detected faces
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generated image and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        try:
            start_time = time.time()
            
            # Convert input image to PIL if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get expression prompt
            prompt = self.expression_prompts.get(
                target_expression.lower(),
                f"a person with {target_expression} expression"
            )
            
            # Combine negative prompts
            negative_prompt = ", ".join(self.negative_prompts)
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Face detection for focus
            faces = []
            if face_focus:
                faces = self.detect_faces(image)
            
            # Generate with memory management
            with self._memory_management():
                if self.config.pipeline_type == PipelineType.IMAGE_TO_IMAGE:
                    result = self.pipeline(
                        prompt=prompt,
                        image=image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        negative_prompt=negative_prompt,
                        **kwargs
                    )
                else:
                    # Fallback to text-to-image
                    result = self.pipeline(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        negative_prompt=negative_prompt,
                        **kwargs
                    )
            
            # Extract generated image
            generated_image = result.images[0]
            
            # Calculate inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Prepare metadata
            metadata = {
                'target_expression': target_expression,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'strength': strength,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_inference_steps,
                'seed': seed,
                'inference_time': inference_time,
                'faces_detected': len(faces),
                'face_coordinates': faces,
                'pipeline_type': self.config.pipeline_type.value,
                'scheduler_type': self.config.scheduler_type.value,
                'device': self.device,
                'model_id': self.config.model_id
            }
            
            logger.info(f"Expression generation completed in {inference_time:.2f}s")
            
            return {
                'success': True,
                'image': generated_image,
                'metadata': metadata,
                'faces': faces
            }
            
        except Exception as e:
            logger.error(f"Expression generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'target_expression': target_expression,
                    'error_type': type(e).__name__
                }
            }
    
    def batch_generate(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        expressions: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate expressions for multiple images
        
        Args:
            images: List of input images
            expressions: List of target expressions
            **kwargs: Additional parameters for generation
            
        Returns:
            List of generation results
        """
        if len(images) != len(expressions):
            raise ValueError("Number of images must match number of expressions")
        
        results = []
        total_images = len(images)
        
        logger.info(f"Starting batch generation for {total_images} images")
        
        for i, (image, expression) in enumerate(zip(images, expressions)):
            logger.info(f"Processing image {i+1}/{total_images} - Expression: {expression}")
            
            result = self.generate_expression(
                image=image,
                target_expression=expression,
                **kwargs
            )
            
            result['batch_index'] = i
            results.append(result)
        
        logger.info(f"Batch generation completed: {len(results)} results")
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {'message': 'No inference data available'}
        
        return {
            'total_inferences': len(self.inference_times),
            'average_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_inference_time': np.sum(self.inference_times),
            'device': self.device,
            'dtype': str(self.dtype),
            'pipeline_type': self.config.pipeline_type.value,
            'scheduler_type': self.config.scheduler_type.value,
            'model_id': self.config.model_id,
            'memory_optimizations': {
                'memory_efficient_attention': self.config.enable_memory_efficient_attention,
                'cpu_offload': self.config.enable_cpu_offload,
                'model_cpu_offload': self.config.enable_model_cpu_offload
            }
        }
    
    def get_supported_expressions(self) -> List[str]:
        """
        Get list of supported expressions
        
        Returns:
            List of supported expression names
        """
        return list(self.expression_prompts.keys())
    
    def update_expression_prompt(self, expression: str, prompt: str):
        """
        Update or add custom expression prompt
        
        Args:
            expression: Expression name
            prompt: Custom prompt for the expression
        """
        self.expression_prompts[expression.lower()] = prompt
        logger.info(f"Updated prompt for expression '{expression}'")
    
    def clear_cache(self):
        """Clear inference cache and free memory"""
        self.inference_times.clear()
        self.memory_usage.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("Cache cleared")
    
    def unload_pipeline(self):
        """Unload pipeline and free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if self.scheduler is not None:
            del self.scheduler
            self.scheduler = None
        
        self.face_detector = None
        self.is_loaded = False
        
        self.clear_cache()
        logger.info("Pipeline unloaded")


def create_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    pipeline_type: str = "img2img",
    device: str = "auto",
    **kwargs
) -> FacialExpressionPipeline:
    """
    Factory function to create a facial expression pipeline
    
    Args:
        model_id: Hugging Face model ID
        pipeline_type: Type of pipeline (img2img, inpaint, text2img)
        device: Device to use (auto, cuda, cpu)
        **kwargs: Additional configuration parameters
        
    Returns:
        FacialExpressionPipeline instance
    """
    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert pipeline type string to enum
    pipeline_type_map = {
        "img2img": PipelineType.IMAGE_TO_IMAGE,
        "image_to_image": PipelineType.IMAGE_TO_IMAGE,
        "inpaint": PipelineType.INPAINTING,
        "inpainting": PipelineType.INPAINTING,
        "text2img": PipelineType.TEXT_TO_IMAGE,
        "text_to_image": PipelineType.TEXT_TO_IMAGE,
        "controlnet": PipelineType.CONTROLNET
    }
    
    pipeline_type_enum = pipeline_type_map.get(
        pipeline_type.lower(),
        PipelineType.IMAGE_TO_IMAGE
    )
    
    # Create configuration
    config = PipelineConfig(
        model_id=model_id,
        pipeline_type=pipeline_type_enum,
        device=device,
        **kwargs
    )
    
    # Create and return pipeline
    pipeline = FacialExpressionPipeline(config)
    logger.info(f"Created facial expression pipeline: {model_id} on {device}")
    
    return pipeline


# Utility functions for pipeline management
def get_available_models() -> List[str]:
    """Get list of available pre-trained models"""
    return [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-2-1-base",
        "CompVis/stable-diffusion-v1-4",
        "hakurei/waifu-diffusion",
        "nitrosocke/Arcane-Diffusion",
        "dreamlike-art/dreamlike-diffusion-1.0"
    ]


def get_recommended_settings(device: str = "auto") -> Dict[str, Any]:
    """Get recommended settings based on device capabilities"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        return {
            "dtype": torch.float16,
            "enable_memory_efficient_attention": True,
            "enable_cpu_offload": False,
            "guidance_scale": 7.5,
            "num_inference_steps": 20,
            "strength": 0.75
        }
    else:
        return {
            "dtype": torch.float32,
            "enable_memory_efficient_attention": False,
            "enable_cpu_offload": True,
            "guidance_scale": 5.0,
            "num_inference_steps": 15,
            "strength": 0.6
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Facial Expression Diffusion Pipeline")
    print("====================================")
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Load pipeline
    if pipeline.load_pipeline():
        print("Pipeline loaded successfully!")
        
        # Get supported expressions
        expressions = pipeline.get_supported_expressions()
        print(f"Supported expressions: {expressions}")
        
        # Get performance stats
        stats = pipeline.get_performance_stats()
        print(f"Performance stats: {stats}")
        
    else:
        print("Failed to load pipeline")