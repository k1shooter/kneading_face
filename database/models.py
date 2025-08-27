"""
Database models for the Facial Expression Transformer application.
Defines SQLAlchemy models for storing conversion history, user uploads, and metadata.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
import json
import uuid

db = SQLAlchemy()

class ConversionHistory(db.Model):
    """
    Model for storing facial expression conversion history and results.
    Tracks all conversion operations with metadata and file paths.
    """
    __tablename__ = 'conversion_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    conversion_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # File information
    original_filename = Column(String(255), nullable=False)
    original_file_path = Column(String(500), nullable=False)
    result_file_path = Column(String(500), nullable=True)
    thumbnail_path = Column(String(500), nullable=True)
    
    # Conversion parameters
    source_expression = Column(String(100), nullable=True)
    target_expression = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False, default='v1.0')
    processing_parameters = Column(Text, nullable=True)  # JSON string
    
    # Processing metadata
    processing_status = Column(String(50), nullable=False, default='pending')  # pending, processing, completed, failed
    processing_start_time = Column(DateTime, nullable=True)
    processing_end_time = Column(DateTime, nullable=True)
    processing_duration = Column(Float, nullable=True)  # seconds
    error_message = Column(Text, nullable=True)
    
    # File metadata
    original_file_size = Column(Integer, nullable=True)  # bytes
    result_file_size = Column(Integer, nullable=True)  # bytes
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    image_format = Column(String(10), nullable=True)  # JPEG, PNG, WebP
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Quality metrics
    confidence_score = Column(Float, nullable=True)  # AI model confidence
    quality_score = Column(Float, nullable=True)  # Image quality assessment
    
    def __init__(self, session_id, original_filename, original_file_path, target_expression, **kwargs):
        self.session_id = session_id
        self.original_filename = original_filename
        self.original_file_path = original_file_path
        self.target_expression = target_expression
        
        # Set optional parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self):
        """Convert model instance to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'conversion_id': self.conversion_id,
            'original_filename': self.original_filename,
            'original_file_path': self.original_file_path,
            'result_file_path': self.result_file_path,
            'thumbnail_path': self.thumbnail_path,
            'source_expression': self.source_expression,
            'target_expression': self.target_expression,
            'model_version': self.model_version,
            'processing_parameters': json.loads(self.processing_parameters) if self.processing_parameters else None,
            'processing_status': self.processing_status,
            'processing_start_time': self.processing_start_time.isoformat() if self.processing_start_time else None,
            'processing_end_time': self.processing_end_time.isoformat() if self.processing_end_time else None,
            'processing_duration': self.processing_duration,
            'error_message': self.error_message,
            'original_file_size': self.original_file_size,
            'result_file_size': self.result_file_size,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'image_format': self.image_format,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'confidence_score': self.confidence_score,
            'quality_score': self.quality_score
        }
    
    def set_processing_parameters(self, params_dict):
        """Set processing parameters as JSON string."""
        self.processing_parameters = json.dumps(params_dict) if params_dict else None
    
    def get_processing_parameters(self):
        """Get processing parameters as dictionary."""
        return json.loads(self.processing_parameters) if self.processing_parameters else {}
    
    def update_status(self, status, error_message=None):
        """Update processing status and timestamps."""
        self.processing_status = status
        self.updated_at = datetime.utcnow()
        
        if status == 'processing' and not self.processing_start_time:
            self.processing_start_time = datetime.utcnow()
        elif status in ['completed', 'failed']:
            if not self.processing_end_time:
                self.processing_end_time = datetime.utcnow()
            if self.processing_start_time:
                self.processing_duration = (self.processing_end_time - self.processing_start_time).total_seconds()
        
        if error_message:
            self.error_message = error_message
    
    def __repr__(self):
        return f'<ConversionHistory {self.conversion_id}: {self.original_filename} -> {self.target_expression}>'


class UserSession(db.Model):
    """
    Model for tracking user sessions and temporary file management.
    Handles session-based file cleanup and user activity tracking.
    """
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Session metadata
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_activity = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Usage statistics
    total_conversions = Column(Integer, nullable=False, default=0)
    successful_conversions = Column(Integer, nullable=False, default=0)
    failed_conversions = Column(Integer, nullable=False, default=0)
    total_processing_time = Column(Float, nullable=False, default=0.0)  # seconds
    
    # File management
    temp_files_count = Column(Integer, nullable=False, default=0)
    temp_files_size = Column(Integer, nullable=False, default=0)  # bytes
    cleanup_scheduled = Column(Boolean, nullable=False, default=False)
    
    def __init__(self, session_id, ip_address=None, user_agent=None):
        self.session_id = session_id
        self.ip_address = ip_address
        self.user_agent = user_agent
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def increment_conversion_stats(self, success=True, processing_time=0.0):
        """Update conversion statistics."""
        self.total_conversions += 1
        if success:
            self.successful_conversions += 1
        else:
            self.failed_conversions += 1
        self.total_processing_time += processing_time
        self.update_activity()
    
    def update_temp_files(self, count_delta=0, size_delta=0):
        """Update temporary file statistics."""
        self.temp_files_count = max(0, self.temp_files_count + count_delta)
        self.temp_files_size = max(0, self.temp_files_size + size_delta)
        self.update_activity()
    
    def to_dict(self):
        """Convert model instance to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'is_active': self.is_active,
            'total_conversions': self.total_conversions,
            'successful_conversions': self.successful_conversions,
            'failed_conversions': self.failed_conversions,
            'total_processing_time': self.total_processing_time,
            'temp_files_count': self.temp_files_count,
            'temp_files_size': self.temp_files_size,
            'cleanup_scheduled': self.cleanup_scheduled
        }
    
    def __repr__(self):
        return f'<UserSession {self.session_id}: {self.total_conversions} conversions>'


class ModelCache(db.Model):
    """
    Model for caching AI model metadata and performance statistics.
    Tracks model loading times, memory usage, and inference performance.
    """
    __tablename__ = 'model_cache'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # diffusion, transformer, etc.
    
    # Model metadata
    model_path = Column(String(500), nullable=True)
    model_size = Column(Integer, nullable=True)  # bytes
    model_parameters = Column(Text, nullable=True)  # JSON string
    
    # Performance metrics
    load_time = Column(Float, nullable=True)  # seconds
    memory_usage = Column(Integer, nullable=True)  # bytes
    average_inference_time = Column(Float, nullable=True)  # seconds
    total_inferences = Column(Integer, nullable=False, default=0)
    
    # Cache management
    is_loaded = Column(Boolean, nullable=False, default=False)
    last_used = Column(DateTime, nullable=True)
    cache_priority = Column(Integer, nullable=False, default=1)  # 1=high, 5=low
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __init__(self, model_name, model_version, model_type, **kwargs):
        self.model_name = model_name
        self.model_version = model_version
        self.model_type = model_type
        
        # Set optional parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def update_performance(self, inference_time):
        """Update performance metrics with new inference time."""
        if self.total_inferences == 0:
            self.average_inference_time = inference_time
        else:
            # Calculate running average
            total_time = self.average_inference_time * self.total_inferences
            self.average_inference_time = (total_time + inference_time) / (self.total_inferences + 1)
        
        self.total_inferences += 1
        self.last_used = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def set_model_parameters(self, params_dict):
        """Set model parameters as JSON string."""
        self.model_parameters = json.dumps(params_dict) if params_dict else None
    
    def get_model_parameters(self):
        """Get model parameters as dictionary."""
        return json.loads(self.model_parameters) if self.model_parameters else {}
    
    def to_dict(self):
        """Convert model instance to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_type': self.model_type,
            'model_path': self.model_path,
            'model_size': self.model_size,
            'model_parameters': self.get_model_parameters(),
            'load_time': self.load_time,
            'memory_usage': self.memory_usage,
            'average_inference_time': self.average_inference_time,
            'total_inferences': self.total_inferences,
            'is_loaded': self.is_loaded,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'cache_priority': self.cache_priority,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<ModelCache {self.model_name} v{self.model_version}: {self.total_inferences} inferences>'


# Database utility functions
def init_db(app):
    """Initialize database with Flask app."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database tables created successfully")

def get_session_history(session_id, limit=50, offset=0):
    """Get conversion history for a specific session."""
    return ConversionHistory.query.filter_by(session_id=session_id)\
                                 .order_by(ConversionHistory.created_at.desc())\
                                 .limit(limit)\
                                 .offset(offset)\
                                 .all()

def get_conversion_by_id(conversion_id):
    """Get conversion record by conversion ID."""
    return ConversionHistory.query.filter_by(conversion_id=conversion_id).first()

def cleanup_old_sessions(days_old=7):
    """Clean up old inactive sessions and their associated data."""
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    
    # Find old sessions
    old_sessions = UserSession.query.filter(UserSession.last_activity < cutoff_date).all()
    session_ids = [session.session_id for session in old_sessions]
    
    if session_ids:
        # Delete associated conversion history
        ConversionHistory.query.filter(ConversionHistory.session_id.in_(session_ids)).delete(synchronize_session=False)
        
        # Delete old sessions
        UserSession.query.filter(UserSession.session_id.in_(session_ids)).delete(synchronize_session=False)
        
        db.session.commit()
        print(f"Cleaned up {len(session_ids)} old sessions")
    
    return len(session_ids)

def get_model_stats():
    """Get aggregated model performance statistics."""
    models = ModelCache.query.all()
    stats = {
        'total_models': len(models),
        'loaded_models': len([m for m in models if m.is_loaded]),
        'total_inferences': sum(m.total_inferences for m in models),
        'average_load_time': sum(m.load_time for m in models if m.load_time) / len([m for m in models if m.load_time]) if models else 0,
        'models': [m.to_dict() for m in models]
    }
    return stats