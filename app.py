#!/usr/bin/env python3
"""
AI Facial Expression Transformer - Main Flask Application
=========================================================

Main Flask application for AI-powered facial expression modification using deep learning.
Provides web interface, REST API, and manages the complete application lifecycle.
"""

import os
import sys
import logging
import traceback
from datetime import datetime
import uuid
import threading # ★ 1. 백그라운드 작업을 위한 스레딩 라이브러리

# Flask and extensions
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Application modules
from config import get_config
from database import init_database
from database.models import db, ConversionHistory, UserSession

# ★ 2. AI 모델 서비스를 가져옵니다.
from services.model_service import get_model_service

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facial_expression_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ★ 3. 백그라운드에서 AI 모델을 실행할 워커 함수 정의
def run_model_processing(app, conversion_id, file_path, target_expression, session_id):
    """
    웹 서버를 차단하지 않고 별도의 스레드에서 이미지를 처리하는 함수.
    """
    # Flask의 애플리케이션 컨텍스트 안에서 실행해야 DB 접근 등이 가능합니다.
    with app.app_context():
        logger.info(f"[Thread] Conversion ID {conversion_id}에 대한 처리 시작")
        conversion = db.session.query(ConversionHistory).filter_by(id=conversion_id).first()
        if not conversion:
            logger.error(f"[Thread] DB에서 Conversion ID {conversion_id}를 찾을 수 없음.")
            return

        try:
            # 상태를 'processing'으로 업데이트
            conversion.update_status('processing')
            db.session.commit()

            # AI 모델 서비스 가져오기 및 실행
            model_service = get_model_service()
            
            # model_service는 FileStorage 객체를 기대하므로, 파일 경로를 다시 열어 전달합니다.
            with open(file_path, 'rb') as f:
                from werkzeug.datastructures import FileStorage
                file_storage = FileStorage(f, filename=os.path.basename(file_path))
                
                # AI 엔진의 핵심 함수 호출
                result = model_service.transform_expression(
                    file=file_storage,
                    target_expression=target_expression,
                    session_id=session_id,
                    conversion_id=conversion_id # model_service에 id 전달
                )

            if result and result.get('success'):
                # 성공 시, 결과 파일 경로와 함께 DB 업데이트
                conversion.result_file_path = result.get('result_file_path')
                conversion.update_status('completed')
                logger.info(f"[Thread] Conversion {conversion_id} 성공적으로 완료.")
            else:
                # 실패 시, 에러 메시지와 함께 DB 업데이트
                error_msg = result.get('error', 'Unknown error in model service')
                conversion.update_status('failed', error_message=error_msg)
                logger.error(f"[Thread] Conversion {conversion_id} 실패: {error_msg}")
            
            db.session.commit()

        except Exception as e:
            logger.error(f"[Thread] Conversion ID {conversion_id} 처리 중 예외 발생: {e}")
            logger.error(traceback.format_exc())
            conversion.update_status('failed', error_message=str(e))
            db.session.commit()
        finally:
            # 임시 원본 파일 삭제 (선택 사항)
            # if os.path.exists(file_path):
            #     # os.remove(file_path)
            #     logger.info(f"[Thread] 원본 임시 파일 {file_path} 삭제됨.")
            pass


class FacialExpressionApp:
    def __init__(self, config_name=None):
        self.app = Flask(__name__)
        self.config_name = config_name or os.getenv('FLASK_ENV', 'development')
        config_class = get_config()
        self.app.config.from_object(config_class)
        self._init_extensions()
        self.db_manager = init_database(self.app)
        self.db = self.db_manager.db
        self._register_routes()
        self._setup_error_handlers()
        self._setup_request_hooks()
        logger.info(f"Facial Expression App initialized with config: {self.config_name}")

    def _init_extensions(self):
        CORS(self.app, resources={r"/api/*": {"origins": "*"}})
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(os.path.join(self.app.config['UPLOAD_FOLDER'], 'results'), exist_ok=True)
        logger.info("Extensions initialized successfully")
    
    def _register_routes(self):
        self.app.add_url_rule('/', 'index', self.index, strict_slashes=False)
        self.app.add_url_rule('/results/<conversion_id>', 'results', self.results, strict_slashes=False)
        self.app.add_url_rule('/history', 'history', self.history, strict_slashes=False)
        self.app.add_url_rule('/upload', 'upload_file', self.upload_file, methods=['POST'], strict_slashes=False)
        self.app.add_url_rule('/api/status/<conversion_id>', 'api_status', self.api_status, strict_slashes=False)
        self.app.add_url_rule('/status/<conversion_id>', 'status_legacy', self.api_status, strict_slashes=False)
        self.app.add_url_rule('/api/history', 'api_history', self.api_history, strict_slashes=False)
        self.app.add_url_rule('/api/health', 'health_check', self.health_check, strict_slashes=False)

    # --- File Upload Route (핵심 변경 부분) ---
    def upload_file(self):
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            target_expression = request.form.get('expression', 'sad')
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not self._allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            session_id = self._ensure_session()
            
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{filename}"
            file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], unique_filename)
            
            file.save(file_path)
            
            # ★ 4. DB에 'pending' 상태로 우선 기록
            conversion = self.db_manager.create_conversion_record(
                session_id=session_id,
                original_filename=filename,
                target_expression=target_expression,
                original_file_path=file_path,
                file_size=os.path.getsize(file_path)
            )
            self.db.session.commit()
            
            # ★ 5. 백그라운드 스레드를 시작하고 즉시 응답 전송
            thread = threading.Thread(
                target=run_model_processing,
                args=(self.app, conversion.id, file_path, target_expression, session_id)
            )
            thread.start()
            
            logger.info(f"파일 업로드 완료. Conversion ID {conversion.id}에 대한 백그라운드 처리 시작됨.")
            
            return jsonify({
                'success': True,
                'conversion_id': conversion.id,
                'message': 'File uploaded, processing has started.',
                'status': 'pending'
            })
            
        except Exception as e:
            logger.error(f"업로드 중 예외 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Upload failed due to an unexpected error'}), 500

    # --- 나머지 메서드들은 기존과 동일 ---
    def index(self):
        return render_template('index.html')

    def results(self, conversion_id):
        conversion = self.db_manager.get_conversion_by_id(conversion_id)
        if not conversion:
            flash('Conversion not found', 'error')
            return redirect(url_for('index'))
        return render_template('results.html', conversion=conversion)

    def history(self):
        session_id = session.get('session_id')
        if not session_id:
            return render_template('history.html', conversions=[])
        conversions = self.db_manager.get_session_conversions(session_id)
        return render_template('history.html', conversions=conversions)

    def api_status(self, conversion_id):
        conversion = self.db_manager.get_conversion_by_id(conversion_id)
        if not conversion:
            return jsonify({'error': 'Conversion not found'}), 404
        return jsonify(conversion.to_dict())

    def api_history(self):
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session not found'}), 400
        conversions = self.db_manager.get_session_conversions(session_id)
        return jsonify([conv.to_dict() for conv in conversions])

    def health_check(self):
        return jsonify({'status': 'healthy'})

    def _setup_error_handlers(self):
        @self.app.errorhandler(404)
        def not_found(error):
            if request.headers.get('Accept') and 'application/json' in request.headers.get('Accept'):
                return jsonify({'error': 'Endpoint not found'}), 404
            return render_template('index.html'), 200

    def _setup_request_hooks(self):
        @self.app.before_request
        def before_request():
            if not request.path.startswith(('/api/', '/static/')):
                self._ensure_session()

    def _ensure_session(self):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        return session.get('session_id')

    def _allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

    def run(self, host='0.0.0.0', port=5000, debug=None):
        logger.info(f"Starting Facial Expression App on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# if __name__ == '__main__' 블록
if __name__ == '__main__':
    app_instance = FacialExpressionApp()
    app_instance.run()
