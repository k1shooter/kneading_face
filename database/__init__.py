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
    # get_conversion_by_id, # 이 함수는 클래스 메서드로 직접 구현할 것이므로 주석 처리하거나 삭제합니다.
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
    # 'get_conversion_by_id',
    'cleanup_old_sessions',
    'get_model_stats',
    'DatabaseManager'
]


class DatabaseManager:
    """
    Database management utility class for centralized database operations.
    """
    
    def __init__(self, app=None):
        self.app = app
        self.db = db
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        self.app = app
        init_db(app)

    # ★★★★★ 여기에 빠진 메서드를 추가합니다 ★★★★★
    def get_conversion_by_id(self, conversion_id):
        """
        Get a single conversion record by its ID.
        
        Args:
            conversion_id: The ID of the conversion to retrieve.
            
        Returns:
            ConversionHistory object or None if not found.
        """
        return ConversionHistory.query.get(conversion_id)
    
    def create_conversion_record(self, session_id, original_filename, 
                                 target_expression, **kwargs):
        """
        Create new conversion history record.
        """
        # **kwargs에 original_file_path가 포함되어 전달됩니다.
        conversion = ConversionHistory(
            session_id=session_id,
            original_filename=original_filename,
            target_expression=target_expression,
            **kwargs
        )
        self.db.session.add(conversion)
        # commit은 app.py의 호출부에서 처리하므로 여기서는 제거하는 것이 좋습니다.
        # self.db.session.commit() 
        return conversion
    
    def update_conversion_status(self, conversion_id, status, **kwargs):
        """
        Update conversion record status and metadata.
        """
        conversion = self.get_conversion_by_id(conversion_id) # 이제 self.get_conversion_by_id 호출이 가능합니다.
        
        if conversion:
            conversion.update_status(status, **kwargs)
            self.db.session.commit()
        
        return conversion
    
    def get_session_conversions(self, session_id, limit=50, offset=0):
        """
        Get conversion history for session.
        """
        return get_session_history(session_id, limit, offset)

    def health_check(self):
        """
        Perform database health check.
        """
        try:
            self.db.session.execute('SELECT 1')
            return { 'status': 'healthy', 'connection': 'active' }
        except Exception as e:
            return { 'status': 'unhealthy', 'connection': 'failed', 'error': str(e) }

    # ... (클래스의 나머지 부분은 동일) ...
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
        session = UserSession.query.filter_by(session_id=session_id).first()
        if not session and create_if_not_exists:
            session = UserSession(session_id=session_id)
            self.db.session.add(session)
            self.db.session.commit()
        return session

    def cleanup_expired_data(self, days_old=7):
        return cleanup_old_sessions(days_old)
    
    def get_model_performance_stats(self):
        return get_model_stats()
    
    def cache_model_info(self, model_name, model_type, **kwargs):
        cached_model = ModelCache.query.filter_by(model_name=model_name, model_type=model_type).first()
        if cached_model:
            cached_model.update_performance(**kwargs)
        else:
            cached_model = ModelCache(model_name=model_name, model_type=model_type, **kwargs)
            self.db.session.add(cached_model)
        self.db.session.commit()
        return cached_model
    
    def get_cached_models(self, model_type=None):
        query = ModelCache.query
        if model_type:
            query = query.filter_by(model_type=model_type)
        return query.all()

# Create default database manager instance
database_manager = DatabaseManager()

def init_database(app):
    database_manager.init_app(app)
    return database_manager

def get_db_manager():
    return database_manager
