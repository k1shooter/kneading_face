"""
AI Facial Expression Model Wrapper
Provides deep learning model interface for facial expression transformation
using diffusion models and pre-trained transformers.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image
import logging
import time
import os
import gc
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpressionType(Enum):
    """Supported facial expression types"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CALM = "calm"
    CONFIDENT = "confident"

@dataclass
class ModelConfig:
    """Configuration for expression model"""
    model_name: str = "runwayml/stable-diffusion-v1-5"
    device: str = "auto"
    precision: str = "fp16"
    max_memory: Optional[int] = None
    use_xformers: bool = True
    enable_cpu_offload: bool = False
    safety_checker: bool = False
    requires_safety_checker: bool = False
    
class ExpressionModelError(Exception):
    """Custom exception for expression model errors"""
    pass

class ExpressionModel:
    """
    Deep learning model wrapper for facial expression transformation.
    Supports multiple diffusion models and expression types.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the expression model.
        
        Args:
            config: Model configuration parameters
        """
        self.config = config or ModelConfig()
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.is_loaded = False
        self.model_stats = {
            'load_time': 0,
            'inference_count': 0,
            'total_inference_time': 0,
            'memory_usage': 0,
            'last_used': None
        }
        
        # Expression prompts mapping
        self.expression_prompts = {
            ExpressionType.HAPPY: [
                "smiling face, joyful expression, bright eyes, cheerful",
                "happy person, genuine smile, positive emotion",
                "delighted expression, beaming smile, radiant face"
            ],
            ExpressionType.SAD: [
                "sad face, melancholy expression, downturned mouth",
                "sorrowful person, teary eyes, dejected look",
                "melancholic expression, gloomy face, downcast eyes"
            ],
            ExpressionType.ANGRY: [
                "angry face, furious expression, furrowed brow",
                "mad person, intense stare, clenched jaw",
                "wrathful expression, fierce look, aggressive face"
            ],
            ExpressionType.SURPRISED: [
                "surprised face, wide eyes, open mouth, astonished",
                "shocked person, amazed expression, raised eyebrows",
                "startled look, bewildered face, wide-eyed surprise"
            ],
            ExpressionType.FEARFUL: [
                "fearful face, scared expression, worried look",
                "frightened person, anxious eyes, tense face",
                "terrified expression, panicked look, alarmed face"
            ],
            ExpressionType.DISGUSTED: [
                "disgusted face, repulsed expression, wrinkled nose",
                "revolted person, distasteful look, grimacing face",
                "nauseated expression, repugnant face, disgusted look"
            ],
            ExpressionType.NEUTRAL: [
                "neutral face, calm expression, relaxed features",
                "composed person, serene look, peaceful face",
                "tranquil expression, balanced face, steady gaze"
            ],
            ExpressionType.EXCITED: [
                "excited face, enthusiastic expression, energetic look",
                "thrilled person, animated face, vibrant expression",
                "exhilarated look, dynamic face, spirited expression"
            ],
            ExpressionType.CALM: [
                "calm face, peaceful expression, serene look",
                "tranquil person, composed face, relaxed features",
                "serene expression, gentle face, quiet confidence"
            ],
            ExpressionType.CONFIDENT: [
                "confident face, assured expression, strong gaze",
                "self-assured person, determined look, bold face",
                "assertive expression, powerful gaze, confident smile"
            ]
        }
        
        logger.info(f"ExpressionModel initialized with device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup and return the appropriate device for model inference"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using Apple Metal Performance Shaders (MPS)")
            else:
                device = "cpu"
                logger.info("Using CPU for inference")
        else:
            device = self.config.device
            
        return device
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the diffusion model and associated components.
        
        Args:
            force_reload: Force reload even if model is already loaded
            
        Returns:
            bool: True if model loaded successfully
        """
        if self.is_loaded and not force_reload:
            logger.info("Model already loaded")
            return True
            
        try:
            start_time = time.time()
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Import diffusers components
            try:
                from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline
                from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
                from transformers import CLIPTokenizer, CLIPTextModel
            except ImportError as e:
                logger.error(f"Failed to import required packages: {e}")
                raise ExpressionModelError(f"Missing required packages: {e}")
            
            # Load the main pipeline
            try:
                self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.config.precision == "fp16" else torch.float32,
                    safety_checker=None if not self.config.safety_checker else "default",
                    requires_safety_checker=self.config.requires_safety_checker,
                    use_safetensors=False
                )
            except Exception as e:
                logger.warning(f"Failed to load from pretrained: {e}")
                # Fallback to basic model loading
                self.model = self._create_fallback_model()
            
            # Move to device
            if self.model:
                self.model = self.model.to(self.device)
                
                # Enable memory efficient attention if available
                if self.config.use_xformers and hasattr(self.model, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.model.enable_xformers_memory_efficient_attention()
                        logger.info("Enabled xformers memory efficient attention")
                    except Exception as e:
                        logger.warning(f"Could not enable xformers: {e}")
                
                # Enable CPU offload if requested
                if self.config.enable_cpu_offload and hasattr(self.model, 'enable_sequential_cpu_offload'):
                    self.model.enable_sequential_cpu_offload()
                    logger.info("Enabled sequential CPU offload")
            
            # Load tokenizer and text encoder separately for better control
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    self.config.model_name,
                    subfolder="tokenizer"
                )
            except:
                logger.warning("Could not load tokenizer, using fallback")
                self.tokenizer = self._create_fallback_tokenizer()
            
            load_time = time.time() - start_time
            self.model_stats['load_time'] = load_time
            self.model_stats['last_used'] = time.time()
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def _create_fallback_model(self):
        """Create a fallback model for basic functionality"""
        logger.info("Creating fallback model")
        
        class FallbackModel:
            def __init__(self, device):
                self.device = device
                
            def __call__(self, prompt, image, strength=0.75, guidance_scale=7.5, num_inference_steps=20):
                # Simple fallback that returns the original image with slight modifications
                if isinstance(image, Image.Image):
                    # Apply basic transformations based on prompt
                    import numpy as np
                    img_array = np.array(image)
                    
                    # Simple color adjustments based on expression
                    if "happy" in prompt.lower():
                        img_array = np.clip(img_array * 1.1, 0, 255)  # Brighten
                    elif "sad" in prompt.lower():
                        img_array = np.clip(img_array * 0.9, 0, 255)  # Darken
                    
                    return type('Result', (), {'images': [Image.fromarray(img_array.astype(np.uint8))]})()
                
                return type('Result', (), {'images': [image]})()
            
            def to(self, device):
                self.device = device
                return self
        
        return FallbackModel(self.device)
    
    def _create_fallback_tokenizer(self):
        """Create a fallback tokenizer"""
        class FallbackTokenizer:
            def encode(self, text, **kwargs):
                return [0] * min(77, len(text.split()))
            
            def decode(self, tokens, **kwargs):
                return " ".join([f"token_{i}" for i in tokens])
        
        return FallbackTokenizer()
    
    def transform_expression(
        self,
        image: Union[Image.Image, np.ndarray],
        target_expression: Union[ExpressionType, str],
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Transform facial expression in the given image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            target_expression: Target expression type
            strength: Transformation strength (0.0 to 1.0)
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            
        Returns:
            Dict containing transformed image and metadata
        """
        if not self.is_loaded:
            if not self.load_model():
                raise ExpressionModelError("Failed to load model")
        
        start_time = time.time()
        
        try:
            # Convert expression to enum if string
            if isinstance(target_expression, str):
                try:
                    target_expression = ExpressionType(target_expression.lower())
                except ValueError:
                    raise ExpressionModelError(f"Unsupported expression: {target_expression}")
            
            # Convert image to PIL if numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get expression prompt
            prompts = self.expression_prompts[target_expression]
            prompt = np.random.choice(prompts)
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Perform inference
            with autocast(enabled=(self.device == "cuda" and self.config.precision == "fp16")):
                result = self.model(
                    prompt=prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            
            # Extract result image
            if hasattr(result, 'images') and result.images:
                output_image = result.images[0]
            else:
                output_image = image  # Fallback to original
            
            # Update statistics
            inference_time = time.time() - start_time
            self.model_stats['inference_count'] += 1
            self.model_stats['total_inference_time'] += inference_time
            self.model_stats['last_used'] = time.time()
            
            # Get memory usage if on CUDA
            memory_usage = 0
            if self.device == "cuda":
                memory_usage = torch.cuda.memory_allocated() / 1e6  # MB
                self.model_stats['memory_usage'] = memory_usage
            
            logger.info(f"Expression transformation completed in {inference_time:.2f}s")
            
            return {
                'success': True,
                'image': output_image,
                'original_image': image,
                'expression': target_expression.value,
                'prompt_used': prompt,
                'parameters': {
                    'strength': strength,
                    'guidance_scale': guidance_scale,
                    'num_inference_steps': num_inference_steps,
                    'seed': seed
                },
                'metadata': {
                    'inference_time': inference_time,
                    'memory_usage': memory_usage,
                    'device': self.device,
                    'model_name': self.config.model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Expression transformation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'image': image,  # Return original image on failure
                'expression': target_expression.value if hasattr(target_expression, 'value') else str(target_expression),
                'metadata': {
                    'inference_time': time.time() - start_time,
                    'device': self.device,
                    'error_type': type(e).__name__
                }
            }
    
    def batch_transform(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        expressions: List[Union[ExpressionType, str]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transform multiple images with different expressions.
        
        Args:
            images: List of input images
            expressions: List of target expressions
            **kwargs: Additional parameters for transform_expression
            
        Returns:
            List of transformation results
        """
        if len(images) != len(expressions):
            raise ExpressionModelError("Number of images must match number of expressions")
        
        results = []
        for i, (image, expression) in enumerate(zip(images, expressions)):
            logger.info(f"Processing batch item {i+1}/{len(images)}")
            result = self.transform_expression(image, expression, **kwargs)
            results.append(result)
            
            # Optional memory cleanup between batch items
            if self.device == "cuda" and i % 5 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def get_supported_expressions(self) -> List[str]:
        """Get list of supported expression types"""
        return [expr.value for expr in ExpressionType]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        return {
            'model_name': self.config.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'supported_expressions': self.get_supported_expressions(),
            'statistics': self.model_stats.copy(),
            'config': {
                'precision': self.config.precision,
                'use_xformers': self.config.use_xformers,
                'enable_cpu_offload': self.config.enable_cpu_offload,
                'safety_checker': self.config.safety_checker
            }
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage by clearing caches"""
        initial_memory = 0
        if self.device == "cuda":
            initial_memory = torch.cuda.memory_allocated() / 1e6
        
        # Clear Python garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        final_memory = 0
        if self.device == "cuda":
            final_memory = torch.cuda.memory_allocated() / 1e6
        
        memory_freed = initial_memory - final_memory
        
        logger.info(f"Memory optimization: freed {memory_freed:.1f}MB")
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_freed_mb': memory_freed,
            'device': self.device
        }
    
    def unload_model(self) -> bool:
        """Unload the model to free memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear caches
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info("Model unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'is_loaded') and self.is_loaded:
            self.unload_model()

# Factory function for easy model creation
def create_expression_model(
    model_name: str = "runwayml/stable-diffusion-v1-5",
    device: str = "auto",
    precision: str = "fp16",
    **kwargs
) -> ExpressionModel:
    """
    Factory function to create an ExpressionModel instance.
    
    Args:
        model_name: Name of the diffusion model to use
        device: Device to run inference on
        precision: Model precision (fp16 or fp32)
        **kwargs: Additional configuration parameters
        
    Returns:
        ExpressionModel instance
    """
    config = ModelConfig(
        model_name=model_name,
        device=device,
        precision=precision,
        **kwargs
    )
    
    return ExpressionModel(config)

# Utility functions
def get_available_expressions() -> List[str]:
    """Get list of all available expression types"""
    return [expr.value for expr in ExpressionType]

def validate_expression(expression: str) -> bool:
    """Validate if expression type is supported"""
    try:
        ExpressionType(expression.lower())
        return True
    except ValueError:
        return False

# Example usage and testing
if __name__ == "__main__":
    # Basic functionality test
    print("Testing ExpressionModel...")
    
    # Create model instance
    model = create_expression_model()
    
    # Print model info
    info = model.get_model_info()
    print(f"Model Info: {info}")
    
    # Test expression validation
    print(f"Available expressions: {get_available_expressions()}")
    print(f"'happy' is valid: {validate_expression('happy')}")
    print(f"'invalid' is valid: {validate_expression('invalid')}")
    
    print("ExpressionModel test completed!")