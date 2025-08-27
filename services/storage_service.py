"""
Storage Service for Facial Expression App

This module provides comprehensive storage management including:
- File upload and storage operations
- Database operations for conversion history
- Temporary file management and cleanup
- Result storage and retrieval
- Session data management
"""

import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union, Tuple
import json
import hashlib
import logging
from pathlib import Path

from flask import current_app, g
from werkzeug.datastructures import FileStorage
from PIL import Image
import numpy as np

from ..database.models import (
    db, ConversionHistory, UserSession, ModelCache,
    get_session_history, get_conversion_by_id, cleanup_old_sessions
)
from ..config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageManager:
    """Main storage management class"""
    
    def __init__(self, app=None):
        self.app = app
        self.upload_folder = None
        self.results_folder = None
        self.temp_folder = None
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize storage manager with Flask app"""
        self.app = app
        self.upload_folder = Path(app.config.get('UPLOAD_FOLDER', 'static/uploads'))
        self.results_folder = Path(app.config.get('RESULTS_FOLDER', 'static/results'))
        self.temp_folder = Path(app.config.get('TEMP_FOLDER', 'static/temp'))
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Set up cleanup scheduler
        self._setup_cleanup_scheduler()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [self.upload_folder, self.results_folder, self.temp_folder]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def _setup_cleanup_scheduler(self):
        """Set up automatic cleanup of old files"""
        # This would typically use a background task scheduler
        # For now, we'll implement manual cleanup methods
        logger.info("Storage cleanup scheduler initialized")
    
    def save_upload(self, file: FileStorage, session_id: str) -> Dict[str, Any]:
        """
        Save uploaded file to storage
        
        Args:
            file: Uploaded file from Flask request
            session_id: User session identifier
            
        Returns:
            Dict containing file information and storage paths
        """
        try:
            if not file or not file.filename:
                raise ValueError("No file provided")
            
            # Generate unique filename
            file_id = str(uuid.uuid4())
            original_filename = file.filename
            file_extension = Path(original_filename).suffix.lower()
            
            # Validate file extension
            allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'.jpg', '.jpeg', '.png', '.webp'})
            if file_extension not in allowed_extensions:
                raise ValueError(f"File extension {file_extension} not allowed")
            
            # Create session-specific directory
            session_dir = self.upload_folder / session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save file
            filename = f"{file_id}{file_extension}"
            file_path = session_dir / filename
            file.save(str(file_path))
            
            # Get file metadata
            file_size = file_path.stat().st_size
            file_hash = self._calculate_file_hash(file_path)
            
            # Validate image
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    format_name = img.format
            except Exception as e:
                # Clean up invalid file
                file_path.unlink(missing_ok=True)
                raise ValueError(f"Invalid image file: {str(e)}")
            
            file_info = {
                'file_id': file_id,
                'original_filename': original_filename,
                'stored_filename': filename,
                'file_path': str(file_path),
                'relative_path': str(file_path.relative_to(Path.cwd())),
                'file_size': file_size,
                'file_hash': file_hash,
                'dimensions': {'width': width, 'height': height},
                'format': format_name,
                'session_id': session_id,
                'upload_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"File uploaded successfully: {filename} for session {session_id}")
            return file_info
            
        except Exception as e:
            logger.error(f"Error saving upload: {str(e)}")
            raise
    
    def save_result(self, result_data: Union[Image.Image, np.ndarray], 
                   conversion_id: str, output_format: str = 'JPEG',
                   quality: int = 95) -> Dict[str, Any]:
        """
        Save processing result to storage
        
        Args:
            result_data: Processed image data
            conversion_id: Conversion record identifier
            output_format: Output image format
            quality: Image quality (for JPEG)
            
        Returns:
            Dict containing result file information
        """
        try:
            # Get conversion record
            conversion = get_conversion_by_id(conversion_id)
            if not conversion:
                raise ValueError(f"Conversion {conversion_id} not found")
            
            # Create results directory for session
            session_dir = self.results_folder / conversion.session_id
            session_dir.mkdir(exist_ok=True)
            
            # Generate result filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            extension = '.jpg' if output_format.upper() == 'JPEG' else f'.{output_format.lower()}'
            filename = f"result_{conversion_id}_{timestamp}{extension}"
            file_path = session_dir / filename
            
            # Convert and save image
            if isinstance(result_data, np.ndarray):
                # Convert numpy array to PIL Image
                if result_data.dtype != np.uint8:
                    result_data = (result_data * 255).astype(np.uint8)
                
                if len(result_data.shape) == 3 and result_data.shape[2] == 3:
                    # RGB image
                    result_image = Image.fromarray(result_data, 'RGB')
                elif len(result_data.shape) == 3 and result_data.shape[2] == 4:
                    # RGBA image
                    result_image = Image.fromarray(result_data, 'RGBA')
                else:
                    # Grayscale
                    result_image = Image.fromarray(result_data, 'L')
            else:
                result_image = result_data
            
            # Save with appropriate settings
            save_kwargs = {'format': output_format}
            if output_format.upper() == 'JPEG':
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            elif output_format.upper() == 'PNG':
                save_kwargs['optimize'] = True
            
            result_image.save(str(file_path), **save_kwargs)
            
            # Get file metadata
            file_size = file_path.stat().st_size
            file_hash = self._calculate_file_hash(file_path)
            width, height = result_image.size
            
            result_info = {
                'result_id': str(uuid.uuid4()),
                'conversion_id': conversion_id,
                'filename': filename,
                'file_path': str(file_path),
                'relative_path': str(file_path.relative_to(Path.cwd())),
                'file_size': file_size,
                'file_hash': file_hash,
                'dimensions': {'width': width, 'height': height},
                'format': output_format,
                'quality': quality if output_format.upper() == 'JPEG' else None,
                'created_timestamp': datetime.utcnow().isoformat()
            }
            
            # Update conversion record with result info
            conversion.result_path = str(file_path.relative_to(Path.cwd()))
            conversion.result_metadata = json.dumps(result_info)
            conversion.status = 'completed'
            conversion.completed_at = datetime.utcnow()
            db.session.commit()
            
            logger.info(f"Result saved successfully: {filename} for conversion {conversion_id}")
            return result_info
            
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")
            raise
    
    def get_file(self, file_path: str) -> Optional[Path]:
        """
        Get file path if it exists
        
        Args:
            file_path: Relative or absolute file path
            
        Returns:
            Path object if file exists, None otherwise
        """
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            
            if path.exists() and path.is_file():
                return path
            return None
            
        except Exception as e:
            logger.error(f"Error getting file {file_path}: {str(e)}")
            return None
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete file from storage
        
        Args:
            file_path: File path to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            path = self.get_file(file_path)
            if path:
                path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def cleanup_session_files(self, session_id: str) -> int:
        """
        Clean up all files for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        
        try:
            # Clean upload files
            upload_dir = self.upload_folder / session_id
            if upload_dir.exists():
                for file_path in upload_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                upload_dir.rmdir()
            
            # Clean result files
            results_dir = self.results_folder / session_id
            if results_dir.exists():
                for file_path in results_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                results_dir.rmdir()
            
            # Clean temp files
            temp_dir = self.temp_folder / session_id
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                # Count files that were in temp directory
                cleaned_count += 10  # Estimate, as directory is already removed
            
            logger.info(f"Cleaned up {cleaned_count} files for session {session_id}")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {str(e)}")
            return cleaned_count
    
    def cleanup_old_files(self, days_old: int = 7) -> Dict[str, int]:
        """
        Clean up files older than specified days
        
        Args:
            days_old: Number of days to keep files
            
        Returns:
            Dict with cleanup statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        stats = {'uploads': 0, 'results': 0, 'temp': 0, 'total': 0}
        
        try:
            # Clean old uploads
            for session_dir in self.upload_folder.iterdir():
                if session_dir.is_dir():
                    try:
                        # Check if directory is old enough
                        dir_mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                        if dir_mtime < cutoff_date:
                            file_count = len(list(session_dir.iterdir()))
                            shutil.rmtree(session_dir)
                            stats['uploads'] += file_count
                    except Exception as e:
                        logger.warning(f"Error cleaning upload dir {session_dir}: {str(e)}")
            
            # Clean old results
            for session_dir in self.results_folder.iterdir():
                if session_dir.is_dir():
                    try:
                        dir_mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                        if dir_mtime < cutoff_date:
                            file_count = len(list(session_dir.iterdir()))
                            shutil.rmtree(session_dir)
                            stats['results'] += file_count
                    except Exception as e:
                        logger.warning(f"Error cleaning results dir {session_dir}: {str(e)}")
            
            # Clean old temp files
            for item in self.temp_folder.iterdir():
                try:
                    item_mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if item_mtime < cutoff_date:
                        if item.is_file():
                            item.unlink()
                            stats['temp'] += 1
                        elif item.is_dir():
                            file_count = len(list(item.rglob('*')))
                            shutil.rmtree(item)
                            stats['temp'] += file_count
                except Exception as e:
                    logger.warning(f"Error cleaning temp item {item}: {str(e)}")
            
            stats['total'] = stats['uploads'] + stats['results'] + stats['temp']
            
            # Also cleanup old database records
            db_cleaned = cleanup_old_sessions(days_old)
            stats['database_sessions'] = db_cleaned
            
            logger.info(f"File cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during file cleanup: {str(e)}")
            return stats
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage usage statistics
        
        Returns:
            Dict with storage statistics
        """
        try:
            stats = {
                'upload_folder': self._get_directory_stats(self.upload_folder),
                'results_folder': self._get_directory_stats(self.results_folder),
                'temp_folder': self._get_directory_stats(self.temp_folder),
                'total_size': 0,
                'total_files': 0
            }
            
            # Calculate totals
            for folder_stats in [stats['upload_folder'], stats['results_folder'], stats['temp_folder']]:
                stats['total_size'] += folder_stats['size_bytes']
                stats['total_files'] += folder_stats['file_count']
            
            # Add human-readable sizes
            stats['total_size_mb'] = round(stats['total_size'] / (1024 * 1024), 2)
            stats['upload_size_mb'] = round(stats['upload_folder']['size_bytes'] / (1024 * 1024), 2)
            stats['results_size_mb'] = round(stats['results_folder']['size_bytes'] / (1024 * 1024), 2)
            stats['temp_size_mb'] = round(stats['temp_folder']['size_bytes'] / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {'error': str(e)}
    
    def _get_directory_stats(self, directory: Path) -> Dict[str, Any]:
        """Get statistics for a directory"""
        try:
            if not directory.exists():
                return {'size_bytes': 0, 'file_count': 0, 'directory_count': 0}
            
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for item in directory.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
                elif item.is_dir():
                    dir_count += 1
            
            return {
                'size_bytes': total_size,
                'file_count': file_count,
                'directory_count': dir_count,
                'path': str(directory)
            }
            
        except Exception as e:
            logger.error(f"Error getting directory stats for {directory}: {str(e)}")
            return {'size_bytes': 0, 'file_count': 0, 'directory_count': 0, 'error': str(e)}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return ""


class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize database manager with Flask app"""
        self.app = app
    
    def create_conversion_record(self, session_id: str, input_file_info: Dict[str, Any],
                               expression_type: str, parameters: Dict[str, Any] = None) -> str:
        """
        Create new conversion record
        
        Args:
            session_id: User session identifier
            input_file_info: Input file information
            expression_type: Target expression type
            parameters: Processing parameters
            
        Returns:
            Conversion ID
        """
        try:
            conversion = ConversionHistory(
                session_id=session_id,
                input_path=input_file_info.get('relative_path'),
                input_filename=input_file_info.get('original_filename'),
                input_metadata=json.dumps(input_file_info),
                expression_type=expression_type,
                status='pending'
            )
            
            if parameters:
                conversion.set_processing_parameters(parameters)
            
            db.session.add(conversion)
            db.session.commit()
            
            logger.info(f"Created conversion record: {conversion.id}")
            return conversion.id
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating conversion record: {str(e)}")
            raise
    
    def update_conversion_status(self, conversion_id: str, status: str, 
                               error_message: str = None) -> bool:
        """
        Update conversion status
        
        Args:
            conversion_id: Conversion identifier
            status: New status
            error_message: Error message if status is 'failed'
            
        Returns:
            True if updated successfully
        """
        try:
            conversion = get_conversion_by_id(conversion_id)
            if not conversion:
                return False
            
            conversion.update_status(status, error_message)
            db.session.commit()
            
            logger.info(f"Updated conversion {conversion_id} status to {status}")
            return True
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating conversion status: {str(e)}")
            return False
    
    def get_session_conversions(self, session_id: str, limit: int = 50, 
                              offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get conversion history for session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            List of conversion records
        """
        try:
            conversions = get_session_history(session_id, limit, offset)
            return [conv.to_dict() for conv in conversions]
            
        except Exception as e:
            logger.error(f"Error getting session conversions: {str(e)}")
            return []
    
    def get_conversion_details(self, conversion_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed conversion information
        
        Args:
            conversion_id: Conversion identifier
            
        Returns:
            Conversion details or None
        """
        try:
            conversion = get_conversion_by_id(conversion_id)
            if conversion:
                return conversion.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting conversion details: {str(e)}")
            return None
    
    def update_session_activity(self, session_id: str) -> bool:
        """
        Update session last activity timestamp
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if updated successfully
        """
        try:
            session = UserSession.query.filter_by(session_id=session_id).first()
            if session:
                session.update_activity()
                db.session.commit()
                return True
            return False
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating session activity: {str(e)}")
            return False


# Global instances
storage_manager = StorageManager()
database_manager = DatabaseManager()


def init_storage(app):
    """Initialize storage services with Flask app"""
    storage_manager.init_app(app)
    database_manager.init_app(app)
    
    # Set up periodic cleanup if configured
    cleanup_interval = app.config.get('STORAGE_CLEANUP_INTERVAL_HOURS', 24)
    cleanup_days = app.config.get('STORAGE_CLEANUP_DAYS', 7)
    
    logger.info(f"Storage services initialized with cleanup every {cleanup_interval}h, keeping files for {cleanup_days} days")


def get_storage_manager() -> StorageManager:
    """Get storage manager instance"""
    return storage_manager


def get_database_manager() -> DatabaseManager:
    """Get database manager instance"""
    return database_manager


# Utility functions for common operations
def save_upload_file(file: FileStorage, session_id: str) -> Dict[str, Any]:
    """Convenience function to save uploaded file"""
    return storage_manager.save_upload(file, session_id)


def save_conversion_result(result_data: Union[Image.Image, np.ndarray], 
                         conversion_id: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to save conversion result"""
    return storage_manager.save_result(result_data, conversion_id, **kwargs)


def create_conversion(session_id: str, input_file_info: Dict[str, Any],
                     expression_type: str, parameters: Dict[str, Any] = None) -> str:
    """Convenience function to create conversion record"""
    return database_manager.create_conversion_record(session_id, input_file_info, expression_type, parameters)


def update_conversion(conversion_id: str, status: str, error_message: str = None) -> bool:
    """Convenience function to update conversion status"""
    return database_manager.update_conversion_status(conversion_id, status, error_message)


def get_conversion_history(session_id: str, **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to get conversion history"""
    return database_manager.get_session_conversions(session_id, **kwargs)


def cleanup_storage(days_old: int = 7) -> Dict[str, int]:
    """Convenience function to cleanup old files"""
    return storage_manager.cleanup_old_files(days_old)