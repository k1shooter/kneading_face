"""
REST API Routes for Facial Expression Transformer
Provides authenticated API endpoints for external integration and batch processing
"""

from flask import Blueprint, request, jsonify, current_app, session
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from functools import wraps
import jwt
import uuid
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import our services
from ..services.model_service import ModelService
from ..services.image_processor import ImageProcessor
from ..database.models import ConversionHistory, UserSession, db

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Configure logging
logger = logging.getLogger(__name__)

# Global model service instance
model_service = None

def get_model_service():
    """Get or create model service instance"""
    global model_service
    if model_service is None:
        model_service = ModelService()
    return model_service

def get_image_processor():
    """Get image processor instance"""
    return ImageProcessor()

# Authentication decorator
def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({
                'error': 'API key required',
                'message': 'Please provide X-API-Key header'
            }), 401
        
        # In production, validate against database
        # For now, check against config
        valid_api_keys = current_app.config.get('API_KEYS', ['demo-api-key'])
        if api_key not in valid_api_keys:
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is not valid'
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function

def require_jwt(f):
    """Decorator to require JWT token authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({
                'error': 'Token required',
                'message': 'Please provide Authorization header with Bearer token'
            }), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            
            payload = jwt.decode(
                token, 
                current_app.config['SECRET_KEY'], 
                algorithms=['HS256']
            )
            request.user_id = payload.get('user_id')
            request.session_id = payload.get('session_id')
            
        except jwt.ExpiredSignatureError:
            return jsonify({
                'error': 'Token expired',
                'message': 'The provided token has expired'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'error': 'Invalid token',
                'message': 'The provided token is invalid'
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function

def validate_file(file: FileStorage) -> Dict[str, Any]:
    """Validate uploaded file"""
    if not file or file.filename == '':
        return {'valid': False, 'error': 'No file provided'}
    
    # Check file extension
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'png', 'jpg', 'jpeg', 'gif', 'webp'})
    if not ('.' in file.filename and 
            file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return {
            'valid': False, 
            'error': f'File type not allowed. Supported: {", ".join(allowed_extensions)}'
        }
    
    # Check file size
    max_size = current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)  # 16MB
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > max_size:
        return {
            'valid': False, 
            'error': f'File too large. Maximum size: {max_size // (1024*1024)}MB'
        }
    
    return {'valid': True, 'size': size}

# API Routes

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        service = get_model_service()
        stats = service.get_processing_stats()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'service_stats': stats
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500

@api_bp.route('/auth/token', methods=['POST'])
def generate_token():
    """Generate JWT token for API access"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        # In production, validate credentials against database
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'error': 'Username and password required'
            }), 400
        
        # For demo purposes, accept any non-empty credentials
        # In production, validate against user database
        if len(username) > 0 and len(password) > 0:
            session_id = str(uuid.uuid4())
            
            # Create JWT token
            payload = {
                'user_id': username,
                'session_id': session_id,
                'exp': datetime.utcnow() + timedelta(hours=24),
                'iat': datetime.utcnow()
            }
            
            token = jwt.encode(
                payload, 
                current_app.config['SECRET_KEY'], 
                algorithm='HS256'
            )
            
            return jsonify({
                'token': token,
                'expires_in': 86400,  # 24 hours
                'session_id': session_id
            })
        else:
            return jsonify({
                'error': 'Invalid credentials'
            }), 401
            
    except Exception as e:
        logger.error(f"Token generation failed: {str(e)}")
        return jsonify({
            'error': 'Token generation failed',
            'message': str(e)
        }), 500

@api_bp.route('/expressions', methods=['GET'])
@require_api_key
def get_expressions():
    """Get list of supported expressions"""
    try:
        model_name = request.args.get('model', 'default')
        service = get_model_service()
        expressions = service.get_expression_list(model_name)
        
        return jsonify({
            'expressions': expressions,
            'model': model_name,
            'count': len(expressions)
        })
        
    except Exception as e:
        logger.error(f"Failed to get expressions: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve expressions',
            'message': str(e)
        }), 500

@api_bp.route('/transform', methods=['POST'])
@require_api_key
def transform_image():
    """Transform single image expression"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'message': 'Please upload an image file with key "image"'
            }), 400
        
        file = request.files['image']
        target_expression = request.form.get('expression')
        
        if not target_expression:
            return jsonify({
                'error': 'Expression parameter required',
                'message': 'Please provide target expression'
            }), 400
        
        # Validate file
        validation = validate_file(file)
        if not validation['valid']:
            return jsonify({
                'error': 'File validation failed',
                'message': validation['error']
            }), 400
        
        # Get or create session
        session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
        
        # Process image
        service = get_model_service()
        result = service.transform_single_image(
            file=file,
            target_expression=target_expression,
            session_id=session_id,
            api_request=True
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'conversion_id': result['conversion_id'],
                'result_url': f"/api/v1/results/{result['conversion_id']}",
                'metadata': result.get('metadata', {}),
                'processing_time': result.get('processing_time', 0)
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Transformation failed'),
                'message': result.get('message', 'Unknown error occurred')
            }), 500
            
    except Exception as e:
        logger.error(f"Image transformation failed: {str(e)}")
        return jsonify({
            'error': 'Transformation failed',
            'message': str(e)
        }), 500

@api_bp.route('/batch', methods=['POST'])
@require_jwt
def batch_transform():
    """Batch transform multiple images"""
    try:
        # Check for files
        if 'images' not in request.files:
            return jsonify({
                'error': 'No image files provided',
                'message': 'Please upload image files with key "images"'
            }), 400
        
        files = request.files.getlist('images')
        target_expression = request.form.get('expression')
        
        if not target_expression:
            return jsonify({
                'error': 'Expression parameter required'
            }), 400
        
        if len(files) > current_app.config.get('MAX_BATCH_SIZE', 10):
            return jsonify({
                'error': 'Too many files',
                'message': f'Maximum {current_app.config.get("MAX_BATCH_SIZE", 10)} files allowed'
            }), 400
        
        # Validate all files first
        validated_files = []
        for i, file in enumerate(files):
            validation = validate_file(file)
            if not validation['valid']:
                return jsonify({
                    'error': f'File {i+1} validation failed',
                    'message': validation['error']
                }), 400
            validated_files.append(file)
        
        # Process batch
        service = get_model_service()
        session_id = request.session_id
        
        batch_result = service.process_batch(
            files=validated_files,
            target_expression=target_expression,
            session_id=session_id
        )
        
        return jsonify({
            'batch_id': batch_result['batch_id'],
            'total_files': len(validated_files),
            'status': 'processing',
            'results_url': f"/api/v1/batch/{batch_result['batch_id']}/status",
            'estimated_completion': batch_result.get('estimated_completion')
        })
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        return jsonify({
            'error': 'Batch processing failed',
            'message': str(e)
        }), 500

@api_bp.route('/batch/<batch_id>/status', methods=['GET'])
@require_jwt
def get_batch_status(batch_id):
    """Get batch processing status"""
    try:
        service = get_model_service()
        status = service.get_batch_status(batch_id)
        
        if not status:
            return jsonify({
                'error': 'Batch not found',
                'message': f'No batch found with ID: {batch_id}'
            }), 404
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Failed to get batch status: {str(e)}")
        return jsonify({
            'error': 'Failed to get batch status',
            'message': str(e)
        }), 500

@api_bp.route('/results/<conversion_id>', methods=['GET'])
@require_api_key
def get_conversion_result(conversion_id):
    """Get conversion result by ID"""
    try:
        service = get_model_service()
        result = service.get_conversion_by_id(conversion_id)
        
        if not result:
            return jsonify({
                'error': 'Conversion not found',
                'message': f'No conversion found with ID: {conversion_id}'
            }), 404
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to get conversion result: {str(e)}")
        return jsonify({
            'error': 'Failed to get conversion result',
            'message': str(e)
        }), 500

@api_bp.route('/history', methods=['GET'])
@require_jwt
def get_conversion_history():
    """Get conversion history for authenticated user"""
    try:
        session_id = request.session_id
        limit = min(int(request.args.get('limit', 50)), 100)  # Max 100
        offset = int(request.args.get('offset', 0))
        
        service = get_model_service()
        history = service.get_conversion_history(
            session_id=session_id,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'history': history,
            'limit': limit,
            'offset': offset,
            'has_more': len(history) == limit
        })
        
    except Exception as e:
        logger.error(f"Failed to get conversion history: {str(e)}")
        return jsonify({
            'error': 'Failed to get conversion history',
            'message': str(e)
        }), 500

@api_bp.route('/stats', methods=['GET'])
@require_api_key
def get_processing_stats():
    """Get processing statistics"""
    try:
        service = get_model_service()
        stats = service.get_processing_stats()
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Failed to get processing stats: {str(e)}")
        return jsonify({
            'error': 'Failed to get processing stats',
            'message': str(e)
        }), 500

@api_bp.route('/models', methods=['GET'])
@require_api_key
def get_available_models():
    """Get list of available AI models"""
    try:
        service = get_model_service()
        models = service.get_available_models()
        
        return jsonify({
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        logger.error(f"Failed to get available models: {str(e)}")
        return jsonify({
            'error': 'Failed to get available models',
            'message': str(e)
        }), 500

@api_bp.route('/models/<model_name>/load', methods=['POST'])
@require_api_key
def load_model(model_name):
    """Load specific AI model"""
    try:
        service = get_model_service()
        result = service.load_model(model_name)
        
        if result['success']:
            return jsonify({
                'success': True,
                'model': model_name,
                'message': f'Model {model_name} loaded successfully',
                'load_time': result.get('load_time', 0)
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to load model'),
                'message': result.get('message', 'Unknown error')
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        return jsonify({
            'error': 'Failed to load model',
            'message': str(e)
        }), 500

@api_bp.route('/cleanup', methods=['POST'])
@require_api_key
def cleanup_resources():
    """Cleanup old files and sessions"""
    try:
        days_old = int(request.json.get('days_old', 7)) if request.json else 7
        
        service = get_model_service()
        result = service.cleanup_old_data(days_old=days_old)
        
        return jsonify({
            'success': True,
            'cleaned_sessions': result.get('cleaned_sessions', 0),
            'cleaned_files': result.get('cleaned_files', 0),
            'freed_space_mb': result.get('freed_space_mb', 0)
        })
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return jsonify({
            'error': 'Cleanup failed',
            'message': str(e)
        }), 500

# Error handlers for API blueprint
@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad Request',
        'message': 'The request was invalid or cannot be served'
    }), 400

@api_bp.errorhandler(401)
def unauthorized(error):
    return jsonify({
        'error': 'Unauthorized',
        'message': 'Authentication required'
    }), 401

@api_bp.errorhandler(403)
def forbidden(error):
    return jsonify({
        'error': 'Forbidden',
        'message': 'Access denied'
    }), 403

@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found'
    }), 404

@api_bp.errorhandler(413)
def payload_too_large(error):
    return jsonify({
        'error': 'Payload Too Large',
        'message': 'The uploaded file is too large'
    }), 413

@api_bp.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({
        'error': 'Rate Limit Exceeded',
        'message': 'Too many requests. Please try again later'
    }), 429

@api_bp.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500

# API Documentation endpoint
@api_bp.route('/docs', methods=['GET'])
def api_documentation():
    """API documentation endpoint"""
    docs = {
        'title': 'Facial Expression Transformer API',
        'version': '1.0.0',
        'description': 'REST API for AI-powered facial expression transformation',
        'base_url': '/api/v1',
        'authentication': {
            'api_key': {
                'type': 'header',
                'name': 'X-API-Key',
                'description': 'API key for basic authentication'
            },
            'jwt': {
                'type': 'header',
                'name': 'Authorization',
                'format': 'Bearer <token>',
                'description': 'JWT token for advanced features'
            }
        },
        'endpoints': {
            'GET /health': 'Health check and service status',
            'POST /auth/token': 'Generate JWT token',
            'GET /expressions': 'Get supported expressions',
            'POST /transform': 'Transform single image',
            'POST /batch': 'Batch transform multiple images',
            'GET /batch/<id>/status': 'Get batch processing status',
            'GET /results/<id>': 'Get conversion result',
            'GET /history': 'Get conversion history',
            'GET /stats': 'Get processing statistics',
            'GET /models': 'Get available AI models',
            'POST /models/<name>/load': 'Load specific model',
            'POST /cleanup': 'Cleanup old resources'
        },
        'rate_limits': {
            'transform': '10 requests per minute',
            'batch': '2 requests per minute',
            'general': '100 requests per minute'
        },
        'file_limits': {
            'max_size': '16MB',
            'supported_formats': ['PNG', 'JPEG', 'GIF', 'WebP'],
            'max_batch_size': 10
        }
    }
    
    return jsonify(docs)

def init_api_routes(app):
    """Initialize API routes with the Flask app"""
    app.register_blueprint(api_bp)
    
    # Add API-specific configuration
    if not hasattr(app.config, 'API_KEYS'):
        app.config['API_KEYS'] = ['demo-api-key']
    
    if not hasattr(app.config, 'MAX_BATCH_SIZE'):
        app.config['MAX_BATCH_SIZE'] = 10
    
    logger.info("API routes initialized successfully")