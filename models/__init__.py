"""
AI Facial Expression Transformer - Models Module

This module provides the core AI models and pipelines for facial expression transformation.
It exposes the main classes and functions needed for expression modification using diffusion models.

Main Components:
- ExpressionModel: High-level wrapper for facial expression transformation
- FacialExpressionPipeline: Low-level diffusion pipeline implementation
- ExpressionType: Enum defining supported expression types
- Factory functions for model creation and configuration

Usage:
    from models import ExpressionModel, ExpressionType
    
    # Create model instance
    model = ExpressionModel()
    
    # Transform expression
    result = model.transform_expression(
        image=input_image,
        target_expression=ExpressionType.HAPPY
    )
"""

# Core model classes
from .expression_model import (
    ExpressionModel,
    ExpressionType,
    ModelConfig,
    create_expression_model
)

# Diffusion pipeline components
from .diffusion_pipeline import (
    FacialExpressionPipeline,
    PipelineConfig,
    create_pipeline,
    get_available_models,
    get_recommended_settings
)

# Version information
__version__ = "1.0.0"
__author__ = "AI Facial Expression Transformer Team"

# Public API exports
__all__ = [
    # Core model classes
    "ExpressionModel",
    "ExpressionType",
    "ModelConfig",
    
    # Pipeline classes
    "FacialExpressionPipeline", 
    "PipelineConfig",
    
    # Factory functions
    "create_expression_model",
    "create_pipeline",
    
    # Utility functions
    "get_available_models",
    "get_recommended_settings",
    
    # Constants
    "SUPPORTED_EXPRESSIONS",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_PIPELINE_CONFIG"
]

# Constants for easy access
SUPPORTED_EXPRESSIONS = [
    "happy", "sad", "angry", "surprised", "fearful", 
    "disgusted", "neutral", "excited", "confused", "contempt"
]

# Default configurations
DEFAULT_MODEL_CONFIG = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "device": "auto",
    "precision": "fp16",
    "enable_memory_efficient_attention": True,
    "enable_cpu_offload": False,
    "cache_dir": "./model_cache"
}

DEFAULT_PIPELINE_CONFIG = {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "pipeline_type": "img2img",
    "device": "auto",
    "torch_dtype": "float16",
    "enable_memory_efficient_attention": True,
    "enable_cpu_offload": False,
    "enable_sequential_cpu_offload": False,
    "enable_model_cpu_offload": False,
    "enable_vae_slicing": True,
    "enable_vae_tiling": False,
    "cache_dir": "./model_cache"
}

# Convenience functions
def get_default_model(**kwargs):
    """
    Create a default ExpressionModel instance with recommended settings.
    
    Args:
        **kwargs: Additional configuration parameters to override defaults
        
    Returns:
        ExpressionModel: Configured model instance
        
    Example:
        model = get_default_model(device="cuda", precision="fp32")
    """
    config = DEFAULT_MODEL_CONFIG.copy()
    config.update(kwargs)
    return create_expression_model(**config)

def get_default_pipeline(**kwargs):
    """
    Create a default FacialExpressionPipeline instance with recommended settings.
    
    Args:
        **kwargs: Additional configuration parameters to override defaults
        
    Returns:
        FacialExpressionPipeline: Configured pipeline instance
        
    Example:
        pipeline = get_default_pipeline(device="cuda", model_id="custom-model")
    """
    config = DEFAULT_PIPELINE_CONFIG.copy()
    config.update(kwargs)
    return create_pipeline(**config)

def list_supported_expressions():
    """
    Get a list of all supported facial expressions.
    
    Returns:
        List[str]: List of supported expression names
    """
    return SUPPORTED_EXPRESSIONS.copy()

def get_expression_info():
    """
    Get detailed information about supported expressions and their descriptions.
    
    Returns:
        Dict[str, str]: Mapping of expression names to descriptions
    """
    return {
        "happy": "Joyful, smiling expression with raised cheeks",
        "sad": "Melancholic expression with downturned mouth",
        "angry": "Fierce expression with furrowed brows and tense features",
        "surprised": "Wide-eyed expression with raised eyebrows and open mouth",
        "fearful": "Anxious expression showing concern or worry",
        "disgusted": "Expression of revulsion with wrinkled nose",
        "neutral": "Calm, emotionless baseline expression",
        "excited": "Enthusiastic, energetic expression",
        "confused": "Puzzled expression with slightly furrowed brows",
        "contempt": "Disdainful expression with slight smirk"
    }

def validate_expression(expression):
    """
    Validate if an expression is supported.
    
    Args:
        expression (str): Expression name to validate
        
    Returns:
        bool: True if expression is supported, False otherwise
        
    Raises:
        ValueError: If expression is not a string
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string")
    
    return expression.lower() in [expr.lower() for expr in SUPPORTED_EXPRESSIONS]

def get_model_requirements():
    """
    Get system requirements and recommendations for running the models.
    
    Returns:
        Dict[str, Any]: System requirements and recommendations
    """
    return {
        "minimum_ram": "8GB",
        "recommended_ram": "16GB",
        "minimum_vram": "4GB",
        "recommended_vram": "8GB",
        "supported_devices": ["cuda", "cpu", "mps"],
        "python_version": ">=3.8",
        "pytorch_version": ">=2.0.0",
        "recommended_os": ["Linux", "Windows", "macOS"],
        "disk_space": "10GB for model cache"
    }

# Module initialization
def _initialize_module():
    """Initialize the models module with any required setup."""
    import logging
    import os
    
    # Set up logging for the models module
    logger = logging.getLogger(__name__)
    logger.info("AI Facial Expression Transformer Models Module initialized")
    
    # Create model cache directory if it doesn't exist
    cache_dir = DEFAULT_MODEL_CONFIG.get("cache_dir", "./model_cache")
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Created model cache directory: {cache_dir}")
        except Exception as e:
            logger.warning(f"Could not create cache directory {cache_dir}: {e}")

# Initialize module on import
_initialize_module()