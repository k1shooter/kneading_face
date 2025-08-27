"""
API Module for Facial Expression Transformer

This module provides REST API endpoints for the facial expression transformation service.
It includes authentication, image processing, batch operations, and conversion history management.

Components:
- routes: REST API endpoints with authentication
- auth: Authentication utilities and decorators

Usage:
    from api import init_api_routes
    init_api_routes(app)
"""

from .routes import init_api_routes

__all__ = ['init_api_routes']

# API version information
API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

# Supported features
SUPPORTED_FEATURES = {
    "authentication": {
        "api_key": True,
        "jwt_token": True,
        "session_based": True
    },
    "image_processing": {
        "single_transform": True,
        "batch_transform": True,
        "supported_formats": ["jpg", "jpeg", "png", "webp"],
        "max_file_size": "10MB",
        "max_batch_size": 10
    },
    "expressions": {
        "supported_types": [
            "happy", "sad", "angry", "surprised", 
            "neutral", "fearful", "disgusted"
        ],
        "custom_expressions": False
    },
    "history": {
        "conversion_tracking": True,
        "pagination": True,
        "metadata_storage": True
    },
    "monitoring": {
        "health_check": True,
        "performance_metrics": True,
        "error_tracking": True
    }
}

def get_api_info():
    """
    Get API information and capabilities.
    
    Returns:
        dict: API information including version, features, and endpoints
    """
    return {
        "name": "Facial Expression Transformer API",
        "version": API_VERSION,
        "prefix": API_PREFIX,
        "description": "REST API for AI-powered facial expression transformation",
        "features": SUPPORTED_FEATURES,
        "endpoints": {
            "authentication": {
                "POST /api/v1/auth/token": "Generate JWT token",
                "GET /api/v1/health": "Service health check"
            },
            "image_processing": {
                "POST /api/v1/transform": "Single image transformation",
                "POST /api/v1/batch": "Batch image processing",
                "GET /api/v1/status/{conversion_id}": "Check conversion status"
            },
            "history": {
                "GET /api/v1/history": "Get conversion history",
                "GET /api/v1/history/{conversion_id}": "Get specific conversion"
            }
        },
        "authentication_methods": [
            "API Key (X-API-Key header)",
            "JWT Token (Bearer token)",
            "Session-based (for web interface)"
        ]
    }

def validate_api_requirements():
    """
    Validate that all required dependencies are available for the API.
    
    Returns:
        tuple: (bool, list) - (is_valid, missing_requirements)
    """
    missing_requirements = []
    
    try:
        import flask
        import jwt
        import werkzeug
        from datetime import datetime
        import uuid
        import os
        import logging
    except ImportError as e:
        missing_requirements.append(f"Missing package: {str(e).split()[-1]}")
    
    # Check for required services
    try:
        from ..services.model_service import ModelService
        from ..services.image_processor import ImageProcessor
        from ..services.storage_service import StorageService
    except ImportError as e:
        missing_requirements.append(f"Missing service: {str(e)}")
    
    # Check for database models
    try:
        from ..database.models import Conversion, UserSession
    except ImportError as e:
        missing_requirements.append(f"Missing database model: {str(e)}")
    
    return len(missing_requirements) == 0, missing_requirements

# Module initialization
def init_api_module(app=None):
    """
    Initialize the API module with optional Flask app.
    
    Args:
        app: Flask application instance (optional)
        
    Returns:
        dict: Initialization status and information
    """
    is_valid, missing_reqs = validate_api_requirements()
    
    if not is_valid:
        return {
            "status": "error",
            "message": "API module initialization failed",
            "missing_requirements": missing_reqs
        }
    
    if app:
        # Register API routes if app is provided
        init_api_routes(app)
        
        # Add API info endpoint
        @app.route(f"{API_PREFIX}/info", methods=['GET'])
        def api_info():
            return get_api_info()
    
    return {
        "status": "success",
        "message": "API module initialized successfully",
        "version": API_VERSION,
        "features": list(SUPPORTED_FEATURES.keys())
    }

# Error handling utilities
class APIError(Exception):
    """Base exception for API-related errors."""
    
    def __init__(self, message, status_code=500, payload=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload
    
    def to_dict(self):
        """Convert error to dictionary format."""
        result = {
            "error": True,
            "message": self.message,
            "status_code": self.status_code
        }
        if self.payload:
            result.update(self.payload)
        return result

class AuthenticationError(APIError):
    """Exception for authentication-related errors."""
    
    def __init__(self, message="Authentication failed"):
        super().__init__(message, status_code=401)

class ValidationError(APIError):
    """Exception for input validation errors."""
    
    def __init__(self, message="Validation failed", details=None):
        payload = {"details": details} if details else None
        super().__init__(message, status_code=400, payload=payload)

class ProcessingError(APIError):
    """Exception for image processing errors."""
    
    def __init__(self, message="Processing failed"):
        super().__init__(message, status_code=422)

class RateLimitError(APIError):
    """Exception for rate limiting errors."""
    
    def __init__(self, message="Rate limit exceeded"):
        super().__init__(message, status_code=429)

# Export error classes
__all__.extend([
    'APIError', 'AuthenticationError', 'ValidationError', 
    'ProcessingError', 'RateLimitError', 'get_api_info', 
    'init_api_module', 'API_VERSION', 'API_PREFIX'
])