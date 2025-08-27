"""
API Authentication Module
Provides JWT token management, API key validation, user session handling,
and security utilities for the facial expression transformation API.
"""

import jwt
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Optional, Tuple, Any
import logging
import os
from flask import request, jsonify, current_app, g
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logger = logging.getLogger(__name__)

class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class TokenManager:
    """JWT token management and validation"""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)  # Default 24 hour expiry
        
    def generate_token(self, user_data: Dict[str, Any], expires_in: Optional[timedelta] = None) -> str:
        """
        Generate JWT token with user data and expiration
        
        Args:
            user_data: Dictionary containing user information
            expires_in: Token expiration time (default: 24 hours)
            
        Returns:
            JWT token string
        """
        try:
            expiry = expires_in or self.token_expiry
            payload = {
                'user_data': user_data,
                'exp': datetime.utcnow() + expiry,
                'iat': datetime.utcnow(),
                'jti': secrets.token_hex(16)  # Unique token ID
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Generated token for user: {user_data.get('session_id', 'unknown')}")
            return token
            
        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise AuthenticationError("Token generation failed", 500)
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token and extract user data
        
        Args:
            token: JWT token string
            
        Returns:
            Dictionary containing user data
            
        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
                
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                raise AuthenticationError("Token has expired", 401)
                
            logger.debug(f"Token validated for user: {payload['user_data'].get('session_id', 'unknown')}")
            return payload['user_data']
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired", 401)
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise AuthenticationError("Invalid token", 401)
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            raise AuthenticationError("Token validation failed", 401)
    
    def refresh_token(self, token: str) -> str:
        """
        Refresh an existing token with new expiration
        
        Args:
            token: Current JWT token
            
        Returns:
            New JWT token with extended expiration
        """
        try:
            user_data = self.validate_token(token)
            return self.generate_token(user_data)
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Token refresh failed: {str(e)}")
            raise AuthenticationError("Token refresh failed", 500)

class APIKeyManager:
    """API key management and validation"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.rate_limits = {}  # Track API key usage for rate limiting
        
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from configuration or environment"""
        api_keys = {}
        
        # Load from environment variables
        env_keys = os.getenv('API_KEYS', '')
        if env_keys:
            for key_data in env_keys.split(','):
                if ':' in key_data:
                    key, name = key_data.split(':', 1)
                    api_keys[key] = {
                        'name': name,
                        'created_at': datetime.utcnow(),
                        'active': True,
                        'rate_limit': 1000  # requests per hour
                    }
        
        # Generate default API key if none configured
        if not api_keys:
            default_key = secrets.token_urlsafe(32)
            api_keys[default_key] = {
                'name': 'default',
                'created_at': datetime.utcnow(),
                'active': True,
                'rate_limit': 100
            }
            logger.warning(f"Generated default API key: {default_key}")
            
        return api_keys
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate API key and return key information
        
        Args:
            api_key: API key string
            
        Returns:
            Dictionary containing API key information
            
        Raises:
            AuthenticationError: If API key is invalid or inactive
        """
        if not api_key:
            raise AuthenticationError("API key is required", 401)
            
        key_info = self.api_keys.get(api_key)
        if not key_info:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            raise AuthenticationError("Invalid API key", 401)
            
        if not key_info.get('active', False):
            raise AuthenticationError("API key is inactive", 401)
            
        # Check rate limiting
        if self._is_rate_limited(api_key):
            raise AuthenticationError("Rate limit exceeded", 429)
            
        self._update_usage(api_key)
        logger.debug(f"API key validated: {key_info['name']}")
        return key_info
    
    def _is_rate_limited(self, api_key: str) -> bool:
        """Check if API key has exceeded rate limit"""
        current_hour = int(time.time() // 3600)
        key_usage = self.rate_limits.get(api_key, {})
        
        if key_usage.get('hour') != current_hour:
            return False
            
        key_info = self.api_keys.get(api_key, {})
        rate_limit = key_info.get('rate_limit', 100)
        
        return key_usage.get('count', 0) >= rate_limit
    
    def _update_usage(self, api_key: str):
        """Update API key usage statistics"""
        current_hour = int(time.time() // 3600)
        
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = {}
            
        key_usage = self.rate_limits[api_key]
        
        if key_usage.get('hour') != current_hour:
            key_usage['hour'] = current_hour
            key_usage['count'] = 1
        else:
            key_usage['count'] = key_usage.get('count', 0) + 1
    
    def generate_api_key(self, name: str, rate_limit: int = 1000) -> str:
        """
        Generate new API key
        
        Args:
            name: Name/description for the API key
            rate_limit: Requests per hour limit
            
        Returns:
            Generated API key string
        """
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            'name': name,
            'created_at': datetime.utcnow(),
            'active': True,
            'rate_limit': rate_limit
        }
        
        logger.info(f"Generated new API key: {name}")
        return api_key

class SessionManager:
    """User session management"""
    
    def __init__(self):
        self.sessions = {}  # In-memory session storage (use Redis in production)
        
    def create_session(self, user_identifier: str = None) -> str:
        """
        Create new user session
        
        Args:
            user_identifier: Optional user identifier
            
        Returns:
            Session ID string
        """
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'user_identifier': user_identifier or f"user_{session_id[:8]}",
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'conversion_count': 0,
            'active': True
        }
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session = self.sessions.get(session_id)
        if session and session.get('active'):
            # Update last activity
            session['last_activity'] = datetime.utcnow()
            return session
        return None
    
    def update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        session = self.sessions.get(session_id)
        if session:
            session['last_activity'] = datetime.utcnow()
            session['conversion_count'] = session.get('conversion_count', 0) + 1
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Remove expired sessions"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if session['last_activity'] < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Global instances
token_manager = None
api_key_manager = None
session_manager = None

def init_auth(app):
    """Initialize authentication components with Flask app"""
    global token_manager, api_key_manager, session_manager
    
    secret_key = app.config.get('SECRET_KEY') or app.config.get('JWT_SECRET_KEY')
    if not secret_key:
        logger.warning("No secret key configured, using default")
        secret_key = 'dev-secret-key-change-in-production'
    
    token_manager = TokenManager(secret_key)
    api_key_manager = APIKeyManager()
    session_manager = SessionManager()
    
    logger.info("Authentication system initialized")

# Decorator functions for route protection
def require_api_key(f):
    """Decorator to require valid API key"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                return jsonify({'error': 'API key required'}), 401
                
            key_info = api_key_manager.validate_api_key(api_key)
            g.api_key_info = key_info
            
            return f(*args, **kwargs)
            
        except AuthenticationError as e:
            return jsonify({'error': e.message}), e.status_code
        except Exception as e:
            logger.error(f"API key validation error: {str(e)}")
            return jsonify({'error': 'Authentication failed'}), 500
            
    return decorated_function

def require_jwt(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({'error': 'Authorization header required'}), 401
                
            user_data = token_manager.validate_token(auth_header)
            g.user_data = user_data
            
            # Update session activity if session_id present
            session_id = user_data.get('session_id')
            if session_id:
                session_manager.update_session_activity(session_id)
            
            return f(*args, **kwargs)
            
        except AuthenticationError as e:
            return jsonify({'error': e.message}), e.status_code
        except Exception as e:
            logger.error(f"JWT validation error: {str(e)}")
            return jsonify({'error': 'Authentication failed'}), 500
            
    return decorated_function

def optional_auth(f):
    """Decorator for optional authentication (creates session if no auth provided)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Try JWT first
            auth_header = request.headers.get('Authorization')
            if auth_header:
                try:
                    user_data = token_manager.validate_token(auth_header)
                    g.user_data = user_data
                    return f(*args, **kwargs)
                except AuthenticationError:
                    pass  # Fall through to API key or session creation
            
            # Try API key
            api_key = request.headers.get('X-API-Key')
            if api_key:
                try:
                    key_info = api_key_manager.validate_api_key(api_key)
                    g.api_key_info = key_info
                    # Create temporary session for API key users
                    session_id = session_manager.create_session(f"api_user_{key_info['name']}")
                    g.user_data = {'session_id': session_id, 'auth_type': 'api_key'}
                    return f(*args, **kwargs)
                except AuthenticationError:
                    pass  # Fall through to session creation
            
            # Create anonymous session
            session_id = session_manager.create_session()
            g.user_data = {'session_id': session_id, 'auth_type': 'anonymous'}
            
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Optional auth error: {str(e)}")
            return jsonify({'error': 'Authentication setup failed'}), 500
            
    return decorated_function

# Utility functions
def generate_user_token(username: str, password: str = None) -> Dict[str, Any]:
    """
    Generate JWT token for user authentication
    
    Args:
        username: User identifier
        password: Optional password (for validation)
        
    Returns:
        Dictionary containing token and expiration info
    """
    try:
        # Create session for user
        session_id = session_manager.create_session(username)
        
        user_data = {
            'username': username,
            'session_id': session_id,
            'auth_type': 'jwt'
        }
        
        token = token_manager.generate_token(user_data)
        
        return {
            'token': token,
            'expires_in': int(token_manager.token_expiry.total_seconds()),
            'token_type': 'Bearer',
            'session_id': session_id
        }
        
    except Exception as e:
        logger.error(f"User token generation failed: {str(e)}")
        raise AuthenticationError("Token generation failed", 500)

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current authenticated user data from Flask g object"""
    return getattr(g, 'user_data', None)

def get_current_api_key() -> Optional[Dict[str, Any]]:
    """Get current API key info from Flask g object"""
    return getattr(g, 'api_key_info', None)

def validate_auth_requirements() -> Tuple[bool, list]:
    """
    Validate that all required authentication dependencies are available
    
    Returns:
        Tuple of (success: bool, missing_requirements: list)
    """
    missing = []
    
    try:
        import jwt
    except ImportError:
        missing.append("PyJWT package required for JWT authentication")
    
    try:
        from werkzeug.security import generate_password_hash
    except ImportError:
        missing.append("Werkzeug package required for password hashing")
    
    if not token_manager:
        missing.append("Token manager not initialized")
        
    if not api_key_manager:
        missing.append("API key manager not initialized")
        
    if not session_manager:
        missing.append("Session manager not initialized")
    
    return len(missing) == 0, missing

# Export main components
__all__ = [
    'AuthenticationError',
    'TokenManager',
    'APIKeyManager', 
    'SessionManager',
    'init_auth',
    'require_api_key',
    'require_jwt',
    'optional_auth',
    'generate_user_token',
    'get_current_user',
    'get_current_api_key',
    'validate_auth_requirements'
]