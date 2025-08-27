#!/usr/bin/env python3
"""
AI Facial Expression Transformer - Main Flask Application
=========================================================

Main Flask application for AI-powered facial expression modification using deep learning.
Provides web interface, REST API, and manages the complete application lifecycle.

Features:
- Web interface with drag-drop upload and expression selection
- REST API endpoints for external integration
- Session management and conversion history tracking
- Real-time processing status and result preview
- File upload validation and temporary file management
- Database integration with automatic migrations
- Error handling and logging
- Performance monitoring and health checks

Author: AI Facial Expression Transformer Team
Version: 1.0.0
"""

import os
import sys
import logging
import traceback
from datetime import datetime, timedelta
from functools import wraps
import uuid
import json

# Flask and extensions
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Application modules
from config import get_config
from database import init_database, get_db_manager
from database.models import ConversionHistory, UserSession, ModelCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facial_expression_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FacialExpressionApp:
    """
    Main application class for the Facial Expression Transformer.
    
    Manages Flask application lifecycle, configuration, database connections,
    session management, file uploads, and core application functionality.
    """
    
    def __init__(self, config_name=None):
        """
        Initialize the Flask application with configuration and extensions.
        
        Args:
            config_name (str, optional): Configuration environment name
        """
        self.app = Flask(__name__)
        self.config_name = config_name or os.getenv('FLASK_ENV', 'development')
        
        # Load configuration
        config_class = get_config()
        self.app.config.from_object(config_class)
        
        # Initialize extensions
        self._init_extensions()
        
        # Initialize database
        self.db_manager = init_database(self.app)
        
        # Register routes
        self._register_routes()
        
        # Setup error handlers
        self._setup_error_handlers()
        
        # Setup request hooks
        self._setup_request_hooks()
        
        logger.info(f"Facial Expression App initialized with config: {self.config_name}")
    
    def _init_extensions(self):
        """Initialize Flask extensions."""
        # Enable CORS for API endpoints
        CORS(self.app, resources={
            r"/api/*": {"origins": "*"},
            r"/health": {"origins": "*"}
        })
        
        # Ensure upload directory exists
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(os.path.join(self.app.config['UPLOAD_FOLDER'], 'results'), exist_ok=True)
        
        logger.info("Extensions initialized successfully")
    
    def _register_routes(self):
        """Register all application routes."""
        
        # Web Interface Routes
        @self.app.route('/')
        def index():
            """Main upload interface."""
            return render_template('index.html', 
                                 max_file_size=self.app.config['MAX_CONTENT_LENGTH'],
                                 allowed_extensions=self.app.config['ALLOWED_EXTENSIONS'])
        
        @self.app.route('/results/<conversion_id>')
        def results(conversion_id):
            """Display conversion results."""
            conversion = self.db_manager.get_conversion_by_id(conversion_id)
            if not conversion:
                flash('Conversion not found', 'error')
                return redirect(url_for('index'))
            
            return render_template('results.html', conversion=conversion)
        
        @self.app.route('/history')
        def history():
            """Display conversion history for current session."""
            session_id = session.get('session_id')
            if not session_id:
                return render_template('history.html', conversions=[])
            
            conversions = self.db_manager.get_session_conversions(session_id)
            return render_template('history.html', conversions=conversions)
        
        # File Upload Routes
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file upload and initiate conversion."""
            try:
                # 1. --- 요청 유효성 검사 ---
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                target_expression = request.form.get('expression', 'happy')
                
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if not self._allowed_file(file.filename):
                    return jsonify({'error': 'Invalid file type'}), 400
                
                # 2. --- 세션 및 파일 저장 ---
                session_id = self._ensure_session()
                
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{filename}"
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], unique_filename)
                
                file.save(file_path)
                
                # 3. --- 데이터베이스 처리 (수정된 부분) ---
                conversion = None # 변수 초기화
                try:
                    # Create conversion record
                    conversion = self.db_manager.create_conversion_record(
                        session_id=session_id,
                        original_filename=filename,
                        target_expression=target_expression,
                        file_path=file_path,
                        file_size=os.path.getsize(file_path)
                    )
                    # ★★★★★ 중요: 변경사항을 데이터베이스에 커밋합니다. ★★★★★
                    self.db_manager.db.session.commit()

                except Exception as db_error:
                    # 데이터베이스 작업 실패 시, 생성된 파일을 삭제하고 세션을 롤백합니다.
                    logger.error(f"Database error during upload: {str(db_error)}")
                    self.db_manager.db.session.rollback() # 변경사항 롤백
                    if os.path.exists(file_path):
                        os.remove(file_path) # 업로드된 파일 정리
                    return jsonify({'error': 'Failed to create conversion record'}), 500

                # 4. --- 성공 응답 반환 ---
                logger.info(f"File uploaded successfully: {unique_filename}, conversion_id: {conversion.id}")
                
                return jsonify({
                    'success': True,
                    'conversion_id': conversion.id,
                    'message': 'File uploaded successfully',
                    'status': 'uploaded'
                })
                
            except RequestEntityTooLarge:
                return jsonify({'error': 'File too large'}), 413
            except Exception as e:
                # 이 외의 예외 처리
                logger.error(f"An unexpected error occurred during upload: {str(e)}")
                logger.error(traceback.format_exc()) # 전체 트레이스백 로깅
                return jsonify({'error': 'Upload failed due to an unexpected error'}), 500
            
        # API Routes
        @self.app.route('/api/convert', methods=['POST'])
        def api_convert():
            """API endpoint for conversion requests."""
            try:
                # This would integrate with the model service
                # For now, return a placeholder response
                return jsonify({
                    'message': 'Conversion API endpoint - Model service integration pending',
                    'status': 'pending_implementation'
                })
            except Exception as e:
                logger.error(f"API convert error: {str(e)}")
                return jsonify({'error': 'Conversion failed'}), 500
        
        @self.app.route('/api/status/<conversion_id>')
        def api_status(conversion_id):
            """Get conversion status via API."""
            try:
                conversion = self.db_manager.get_conversion_by_id(conversion_id)
                if not conversion:
                    return jsonify({'error': 'Conversion not found'}), 404
                
                return jsonify(conversion.to_dict())
            except Exception as e:
                logger.error(f"API status error: {str(e)}")
                return jsonify({'error': 'Status check failed'}), 500
        
        @self.app.route('/api/history/<session_id>')
        def api_history(session_id):
            """Get conversion history via API."""
            try:
                limit = request.args.get('limit', 50, type=int)
                offset = request.args.get('offset', 0, type=int)
                
                conversions = self.db_manager.get_session_conversions(session_id, limit, offset)
                return jsonify([conv.to_dict() for conv in conversions])
            except Exception as e:
                logger.error(f"API history error: {str(e)}")
                return jsonify({'error': 'History retrieval failed'}), 500
        
        # Health and Status Routes
        @self.app.route('/health')
        def health_check():
            """Application health check endpoint."""
            try:
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'version': '1.0.0',
                    'database': self.db_manager.health_check(),
                    'upload_directory': {
                        'exists': os.path.exists(self.app.config['UPLOAD_FOLDER']),
                        'writable': os.access(self.app.config['UPLOAD_FOLDER'], os.W_OK)
                    }
                }
                
                # Check if any critical components are failing
                if not health_status['database']['connection']:
                    health_status['status'] = 'unhealthy'
                
                status_code = 200 if health_status['status'] == 'healthy' else 503
                return jsonify(health_status), status_code
                
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 503
        
        @self.app.route('/stats')
        def app_stats():
            """Application statistics endpoint."""
            try:
                stats = {
                    'total_conversions': ConversionHistory.query.count(),
                    'active_sessions': UserSession.query.filter(
                        UserSession.last_activity > datetime.utcnow() - timedelta(hours=24)
                    ).count(),
                    'successful_conversions': ConversionHistory.query.filter_by(status='completed').count(),
                    'failed_conversions': ConversionHistory.query.filter_by(status='failed').count(),
                    'pending_conversions': ConversionHistory.query.filter_by(status='processing').count(),
                    'disk_usage': self._get_disk_usage(),
                    'uptime': self._get_uptime()
                }
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Stats error: {str(e)}")
                return jsonify({'error': 'Stats unavailable'}), 500
    
    def _setup_error_handlers(self):
        """Setup application error handlers."""
        
        @self.app.errorhandler(404)
        def not_found(error):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Endpoint not found'}), 404
            return render_template('index.html'), 404
        
        @self.app.errorhandler(413)
        def file_too_large(error):
            return jsonify({'error': 'File too large'}), 413
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {str(error)}")
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Internal server error'}), 500
            flash('An error occurred. Please try again.', 'error')
            return render_template('index.html'), 500
    
    def _setup_request_hooks(self):
        """Setup request hooks for session management and logging."""
        
        @self.app.before_request
        def before_request():
            """Execute before each request."""
            # Ensure session exists for web requests
            if not request.path.startswith('/api/') and not request.path.startswith('/static/'):
                self._ensure_session()
            
            # Log request
            logger.debug(f"Request: {request.method} {request.path}")
        
        @self.app.after_request
        def after_request(response):
            """Execute after each request."""
            # Update session activity
            session_id = session.get('session_id')
            if session_id:
                try:
                    user_session = UserSession.query.filter_by(session_id=session_id).first()
                    if user_session:
                        user_session.update_activity()
                        self.db_manager.db.session.commit()
                except Exception as e:
                    logger.error(f"Session update error: {str(e)}")
            
            return response
    
    def _ensure_session(self):
        """Ensure user session exists and return session ID."""
        if 'session_id' not in session:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            session.permanent = True
            
            # Create database session record
            try:
                user_session = UserSession(
                    session_id=session_id,
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent', '')
                )
                self.db_manager.db.session.add(user_session)
                self.db_manager.db.session.commit()
                logger.info(f"New session created: {session_id}")
            except Exception as e:
                logger.error(f"Session creation error: {str(e)}")
        
        return session['session_id']
    
    def _allowed_file(self, filename):
        """Check if file extension is allowed."""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.app.config['ALLOWED_EXTENSIONS'])
    
    def _get_disk_usage(self):
        """Get disk usage statistics for upload directory."""
        try:
            upload_dir = self.app.config['UPLOAD_FOLDER']
            total_size = 0
            file_count = 0
            
            for dirpath, dirnames, filenames in os.walk(upload_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                        file_count += 1
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_count': file_count
            }
        except Exception as e:
            logger.error(f"Disk usage calculation error: {str(e)}")
            return {'error': 'Unable to calculate disk usage'}
    
    def _get_uptime(self):
        """Get application uptime."""
        # This would be more accurate with a startup timestamp
        return "Uptime tracking not implemented"
    
    def run(self, host='0.0.0.0', port=5000, debug=None):
        """Run the Flask application."""
        if debug is None:
            debug = self.app.config.get('DEBUG', False)
        
        logger.info(f"Starting Facial Expression App on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def create_app(config_name=None):
    """
    Application factory function.
    
    Args:
        config_name (str, optional): Configuration environment name
        
    Returns:
        Flask: Configured Flask application instance
    """
    app_instance = FacialExpressionApp(config_name)
    return app_instance.app

def init_database_tables():
    """Initialize database tables if they don't exist."""
    try:
        from database.migrations import DatabaseMigrator
        
        app = create_app()
        with app.app_context():
            migrator = DatabaseMigrator(app)
            success = migrator.create_tables()
            
            if success:
                logger.info("Database tables initialized successfully")
                
                # Check if we should seed sample data
                if os.getenv('SEED_SAMPLE_DATA', 'false').lower() == 'true':
                    migrator.seed_sample_data()
                    logger.info("Sample data seeded successfully")
            else:
                logger.error("Failed to initialize database tables")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        return False

if __name__ == '__main__':
    """Main application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Facial Expression Transformer')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--init-db', action='store_true', help='Initialize database tables')
    parser.add_argument('--config', default=None, help='Configuration environment')
    
    args = parser.parse_args()
    
    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database tables...")
        if init_database_tables():
            logger.info("Database initialization completed successfully")
        else:
            logger.error("Database initialization failed")
            sys.exit(1)
    
    # Create and run application
    try:
        app_instance = FacialExpressionApp(args.config)
        app_instance.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Application shutdown requested")
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)