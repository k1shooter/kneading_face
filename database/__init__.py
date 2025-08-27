"""
Database module initialization for Facial Expression App.

This module provides database initialization, model imports, and utility functions
for the facial expression transformation application.
"""

from flask_sqlalchemy import SQLAlchemy
from .models import (
    db,
    ConversionHistory,
    UserSession,
    ModelCache,
    init_db,
    get_session_history,
    get_conversion_by_id,
    cleanup_old_sessions,
    get_model_stats
)

# Export database instance and models
__all__ = [
    'db',
    'ConversionHistory',
    'UserSession', 
    'ModelCache',
    'init_db',
    'get_session_history',
    'get_conversion_by_id',
    'cleanup_old_sessions',
    'get_model_stats',
    'DatabaseManager'
]


class DatabaseManager:
    """
    Database management utility class for centralized database operations.
    
    Provides high-level interface for common database operations including
    initialization, session management, and cleanup tasks.
    """
    
    def __init__(self, app=None):
        """
        Initialize DatabaseManager.
        
        Args:
            app: Flask application instance (optional)
        """
        self.app = app
        self.db = db
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """
        Initialize database with Flask application.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        init_db(app)
    
    def create_tables(self):
        """Create all database tables."""
        with self.app.app_context():
            self.db.create_all()
    
    def drop_tables(self):
        """Drop all database tables."""
        with self.app.app_context():
            self.db.drop_all()
    
    def reset_database(self):
        """Reset database by dropping and recreating all tables."""
        self.drop_tables()
        self.create_tables()
    
    def get_user_session(self, session_id, create_if_not_exists=True):
        """
        Get or create user session.
        
        Args:
            session_id: Session identifier
            create_if_not_exists: Create session if it doesn't exist
            
        Returns:
            UserSession object or None
        """
        session = UserSession.query.filter_by(session_id=session_id).first()
        
        if not session and create_if_not_exists:
            session = UserSession(session_id=session_id)
            self.db.session.add(session)
            self.db.session.commit()
        
        return session
    
    def create_conversion_record(self, session_id, original_filename, 
                               target_expression, **kwargs):
        """
        Create new conversion history record.
        
        Args:
            session_id: User session identifier
            original_filename: Original image filename
            target_expression: Target facial expression
            **kwargs: Additional conversion parameters
            
        Returns:
            ConversionHistory object
        """
        conversion = ConversionHistory(
            session_id=session_id,
            original_filename=original_filename,
            target_expression=target_expression,
            **kwargs
        )
        
        self.db.session.add(conversion)
        self.db.session.commit()
        
        return conversion
    
    def update_conversion_status(self, conversion_id, status, **kwargs):
        """
        Update conversion record status and metadata.
        
        Args:
            conversion_id: Conversion record ID
            status: New status
            **kwargs: Additional fields to update
            
        Returns:
            Updated ConversionHistory object or None
        """
        conversion = get_conversion_by_id(conversion_id)
        
        if conversion:
            conversion.update_status(status, **kwargs)
            self.db.session.commit()
        
        return conversion
    
    def get_session_conversions(self, session_id, limit=50, offset=0):
        """
        Get conversion history for session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of records
            offset: Record offset for pagination
            
        Returns:
            List of ConversionHistory objects
        """
        return get_session_history(session_id, limit, offset)
    
    def cleanup_expired_data(self, days_old=7):
        """
        Clean up expired sessions and conversion records.
        
        Args:
            days_old: Age threshold in days
            
        Returns:
            Number of cleaned sessions
        """
        return cleanup_old_sessions(days_old)
    
    def get_model_performance_stats(self):
        """
        Get aggregated model performance statistics.
        
        Returns:
            Dictionary with model statistics
        """
        return get_model_stats()
    
    def cache_model_info(self, model_name, model_type, **kwargs):
        """
        Cache AI model information and performance data.
        
        Args:
            model_name: Name of the AI model
            model_type: Type/category of model
            **kwargs: Additional model parameters
            
        Returns:
            ModelCache object
        """
        # Check if model already cached
        cached_model = ModelCache.query.filter_by(
            model_name=model_name,
            model_type=model_type
        ).first()
        
        if cached_model:
            # Update existing cache
            cached_model.update_performance(**kwargs)
        else:
            # Create new cache entry
            cached_model = ModelCache(
                model_name=model_name,
                model_type=model_type,
                **kwargs
            )
            self.db.session.add(cached_model)
        
        self.db.session.commit()
        return cached_model
    
    def get_cached_models(self, model_type=None):
        """
        Get cached model information.
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            List of ModelCache objects
        """
        query = ModelCache.query
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        
        return query.all()
    
    def health_check(self):
        """
        Perform database health check.
        
        Returns:
            Dictionary with health status
        """
        try:
            # Test database connection
            self.db.session.execute('SELECT 1')
            
            # Get basic statistics
            total_sessions = UserSession.query.count()
            total_conversions = ConversionHistory.query.count()
            cached_models = ModelCache.query.count()
            
            return {
                'status': 'healthy',
                'connection': 'active',
                'total_sessions': total_sessions,
                'total_conversions': total_conversions,
                'cached_models': cached_models,
                'timestamp': self.db.func.now()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connection': 'failed',
                'error': str(e),
                'timestamp': None
            }


# Create default database manager instance
database_manager = DatabaseManager()


def init_database(app):
    """
    Initialize database with Flask application.
    
    Convenience function for database initialization.
    
    Args:
        app: Flask application instance
    """
    database_manager.init_app(app)
    return database_manager


def get_db_manager():
    """
    Get database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    return database_manager