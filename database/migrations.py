#!/usr/bin/env python3
"""
Database Migrations and Setup
Handles database initialization, table creation, and data migrations for the facial expression app.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from flask import Flask
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from database.models import db, ConversionHistory, UserSession, ModelCache
from database import init_database, get_db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """
    Handles database migrations and setup operations.
    """
    
    def __init__(self, app=None):
        """
        Initialize the database migrator.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        self.db = db
        
    def init_app(self, app):
        """
        Initialize with Flask application.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        
    def create_tables(self):
        """
        Create all database tables.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.app.app_context():
                # Create all tables
                self.db.create_all()
                logger.info("Database tables created successfully")
                
                # Verify tables were created
                inspector = inspect(self.db.engine)
                tables = inspector.get_table_names()
                
                expected_tables = ['conversion_history', 'user_sessions', 'model_cache']
                missing_tables = [table for table in expected_tables if table not in tables]
                
                if missing_tables:
                    logger.warning(f"Missing tables: {missing_tables}")
                    return False
                    
                logger.info(f"All tables created: {tables}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {str(e)}")
            return False
            
    def drop_tables(self):
        """
        Drop all database tables.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.app.app_context():
                self.db.drop_all()
                logger.info("Database tables dropped successfully")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error dropping tables: {str(e)}")
            return False
            
    def reset_database(self):
        """
        Reset database by dropping and recreating all tables.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Resetting database...")
        
        if not self.drop_tables():
            return False
            
        if not self.create_tables():
            return False
            
        logger.info("Database reset completed successfully")
        return True
        
    def check_database_health(self):
        """
        Check database connection and table integrity.
        
        Returns:
            dict: Health check results
        """
        health_status = {
            'connection': False,
            'tables_exist': False,
            'table_count': 0,
            'sample_data': False,
            'errors': []
        }
        
        try:
            with self.app.app_context():
                # Test connection
                result = self.db.session.execute(text('SELECT 1'))
                health_status['connection'] = True
                
                # Check tables
                inspector = inspect(self.db.engine)
                tables = inspector.get_table_names()
                health_status['table_count'] = len(tables)
                
                expected_tables = ['conversion_history', 'user_sessions', 'model_cache']
                health_status['tables_exist'] = all(table in tables for table in expected_tables)
                
                # Check for sample data
                conversion_count = ConversionHistory.query.count()
                session_count = UserSession.query.count()
                health_status['sample_data'] = conversion_count > 0 or session_count > 0
                
                logger.info(f"Database health check passed: {health_status}")
                
        except SQLAlchemyError as e:
            health_status['errors'].append(str(e))
            logger.error(f"Database health check failed: {str(e)}")
            
        return health_status
        
    def seed_sample_data(self):
        """
        Insert sample data for testing and development.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.app.app_context():
                # Create sample user session
                sample_session = UserSession(
                    session_id='sample_session_123',
                    ip_address='127.0.0.1',
                    user_agent='Sample User Agent',
                    created_at=datetime.utcnow(),
                    last_activity=datetime.utcnow(),
                    conversion_count=2,
                    temp_files=['sample1.jpg', 'sample2.jpg']
                )
                
                self.db.session.add(sample_session)
                self.db.session.flush()  # Get the ID
                
                # Create sample conversion records
                sample_conversions = [
                    ConversionHistory(
                        conversion_id='conv_sample_001',
                        session_id='sample_session_123',
                        original_filename='sample_input.jpg',
                        target_expression='happy',
                        status='completed',
                        created_at=datetime.utcnow() - timedelta(hours=2),
                        completed_at=datetime.utcnow() - timedelta(hours=2, minutes=30),
                        processing_time=1.5,
                        result_filename='sample_output_happy.jpg',
                        confidence_score=0.92,
                        model_used='diffusion_v1',
                        processing_parameters={
                            'strength': 0.8,
                            'guidance_scale': 7.5,
                            'num_inference_steps': 20
                        }
                    ),
                    ConversionHistory(
                        conversion_id='conv_sample_002',
                        session_id='sample_session_123',
                        original_filename='sample_input2.jpg',
                        target_expression='surprised',
                        status='completed',
                        created_at=datetime.utcnow() - timedelta(hours=1),
                        completed_at=datetime.utcnow() - timedelta(minutes=58),
                        processing_time=2.1,
                        result_filename='sample_output_surprised.jpg',
                        confidence_score=0.87,
                        model_used='diffusion_v1',
                        processing_parameters={
                            'strength': 0.9,
                            'guidance_scale': 8.0,
                            'num_inference_steps': 25
                        }
                    )
                ]
                
                for conversion in sample_conversions:
                    self.db.session.add(conversion)
                
                # Create sample model cache entry
                sample_model_cache = ModelCache(
                    model_name='diffusion_v1',
                    model_path='/models/diffusion_v1',
                    model_type='diffusion',
                    load_time=5.2,
                    memory_usage=1024.5,
                    last_used=datetime.utcnow(),
                    usage_count=15,
                    average_inference_time=1.8,
                    model_parameters={
                        'model_id': 'runwayml/stable-diffusion-v1-5',
                        'torch_dtype': 'float16',
                        'device': 'cuda'
                    }
                )
                
                self.db.session.add(sample_model_cache)
                
                # Commit all changes
                self.db.session.commit()
                
                logger.info("Sample data seeded successfully")
                return True
                
        except SQLAlchemyError as e:
            self.db.session.rollback()
            logger.error(f"Error seeding sample data: {str(e)}")
            return False
            
    def cleanup_old_data(self, days_old=30):
        """
        Clean up old conversion records and sessions.
        
        Args:
            days_old: Number of days to keep data
            
        Returns:
            dict: Cleanup statistics
        """
        cleanup_stats = {
            'conversions_deleted': 0,
            'sessions_deleted': 0,
            'errors': []
        }
        
        try:
            with self.app.app_context():
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                
                # Delete old conversion records
                old_conversions = ConversionHistory.query.filter(
                    ConversionHistory.created_at < cutoff_date
                ).all()
                
                for conversion in old_conversions:
                    self.db.session.delete(conversion)
                    cleanup_stats['conversions_deleted'] += 1
                
                # Delete old sessions
                old_sessions = UserSession.query.filter(
                    UserSession.created_at < cutoff_date
                ).all()
                
                for session in old_sessions:
                    self.db.session.delete(session)
                    cleanup_stats['sessions_deleted'] += 1
                
                self.db.session.commit()
                
                logger.info(f"Cleanup completed: {cleanup_stats}")
                
        except SQLAlchemyError as e:
            self.db.session.rollback()
            cleanup_stats['errors'].append(str(e))
            logger.error(f"Error during cleanup: {str(e)}")
            
        return cleanup_stats


def create_app():
    """
    Create Flask application for migrations.
    
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    config = get_config()
    app.config.from_object(config)
    
    # Initialize database
    init_database(app)
    
    return app


def main():
    """
    Main migration script entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Migration Tool')
    parser.add_argument('--action', choices=['create', 'drop', 'reset', 'health', 'seed', 'cleanup'],
                       default='create', help='Migration action to perform')
    parser.add_argument('--days', type=int, default=30,
                       help='Days to keep data for cleanup action')
    parser.add_argument('--force', action='store_true',
                       help='Force action without confirmation')
    
    args = parser.parse_args()
    
    # Create Flask app
    app = create_app()
    
    # Initialize migrator
    migrator = DatabaseMigrator(app)
    
    # Execute action
    if args.action == 'create':
        logger.info("Creating database tables...")
        if migrator.create_tables():
            print("✅ Database tables created successfully!")
        else:
            print("❌ Failed to create database tables!")
            sys.exit(1)
            
    elif args.action == 'drop':
        if not args.force:
            confirm = input("⚠️  This will drop all tables. Continue? (y/N): ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                sys.exit(0)
        
        logger.info("Dropping database tables...")
        if migrator.drop_tables():
            print("✅ Database tables dropped successfully!")
        else:
            print("❌ Failed to drop database tables!")
            sys.exit(1)
            
    elif args.action == 'reset':
        if not args.force:
            confirm = input("⚠️  This will reset the entire database. Continue? (y/N): ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                sys.exit(0)
        
        logger.info("Resetting database...")
        if migrator.reset_database():
            print("✅ Database reset successfully!")
        else:
            print("❌ Failed to reset database!")
            sys.exit(1)
            
    elif args.action == 'health':
        logger.info("Checking database health...")
        health = migrator.check_database_health()
        
        print("\n📊 Database Health Report:")
        print(f"Connection: {'✅' if health['connection'] else '❌'}")
        print(f"Tables Exist: {'✅' if health['tables_exist'] else '❌'}")
        print(f"Table Count: {health['table_count']}")
        print(f"Sample Data: {'✅' if health['sample_data'] else '❌'}")
        
        if health['errors']:
            print(f"Errors: {health['errors']}")
            
    elif args.action == 'seed':
        logger.info("Seeding sample data...")
        if migrator.seed_sample_data():
            print("✅ Sample data seeded successfully!")
        else:
            print("❌ Failed to seed sample data!")
            sys.exit(1)
            
    elif args.action == 'cleanup':
        logger.info(f"Cleaning up data older than {args.days} days...")
        stats = migrator.cleanup_old_data(args.days)
        
        print(f"\n🧹 Cleanup Results:")
        print(f"Conversions deleted: {stats['conversions_deleted']}")
        print(f"Sessions deleted: {stats['sessions_deleted']}")
        
        if stats['errors']:
            print(f"Errors: {stats['errors']}")
            sys.exit(1)
        else:
            print("✅ Cleanup completed successfully!")


if __name__ == '__main__':
    main()