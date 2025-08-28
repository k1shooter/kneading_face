"""
Model Service - Core service for AI facial expression transformation

This service orchestrates the AI models, image processing, and database operations
to provide a unified interface for facial expression transformation.
"""

import os
import logging
import time
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from PIL import Image
import numpy as np
from werkzeug.datastructures import FileStorage

# Import our custom modules
from models import ExpressionModel, FacialExpressionPipeline, create_expression_model, create_pipeline
from .image_processor import ImageProcessor
from database.models import ConversionHistory, UserSession, ModelCache, db
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelService:
    """
    Core service for managing AI facial expression transformation operations.
    
    This service provides a high-level interface for:
    - Loading and managing AI models
    - Processing image transformations
    - Tracking conversion history
    - Managing user sessions
    - Batch processing capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Model Service.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.models: Dict[str, ExpressionModel] = {}
        self.pipelines: Dict[str, FacialExpressionPipeline] = {}
        self.model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.image_processor = ImageProcessor()

        # Performance tracking
        self.stats = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'average_processing_time': 0.0,
            'models_loaded': 0
        }
        
        logger.info("ModelService initialized")
    
    def get_or_create_model(self, model_name: str = "default", **kwargs) -> ExpressionModel:
        """
        Get existing model or create new one.
        
        Args:
            model_name: Name/identifier for the model
            **kwargs: Additional model configuration
            
        Returns:
            ExpressionModel instance
        """
        with self.model_lock:
            actual_model_name = Config.DEFAULT_MODEL_NAME if model_name == 'default' else model_name
            if actual_model_name not in self.models:
                try:
                    logger.info(f"Creating new model: {actual_model_name}")
                    self.models[actual_model_name] = create_expression_model(
                        model_name=actual_model_name,
                        device=kwargs.get('device', 'auto'),
                        precision=kwargs.get('precision', 'fp16')
                    )
                    self.stats['models_loaded'] += 1
                    
                    # Cache model info in database
                    self._cache_model_info(actual_model_name, kwargs)
                    
                except Exception as e:
                    logger.error(f"Failed to create model {actual_model_name}: {str(e)}")
                    raise
            
            return self.models[actual_model_name]
    
    def get_or_create_pipeline(self, pipeline_name: str = "default", **kwargs) -> FacialExpressionPipeline:
        """
        Get existing pipeline or create new one.
        
        Args:
            pipeline_name: Name/identifier for the pipeline
            **kwargs: Additional pipeline configuration
            
        Returns:
            FacialExpressionPipeline instance
        """
        with self.model_lock:
            if pipeline_name not in self.pipelines:
                try:
                    logger.info(f"Creating new pipeline: {pipeline_name}")
                    self.pipelines[pipeline_name] = create_pipeline(
                        model_id=kwargs.get('model_id', 'runwayml/stable-diffusion-v1-5'),
                        pipeline_type=kwargs.get('pipeline_type', 'img2img'),
                        device=kwargs.get('device', 'auto'),
                        **kwargs
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to create pipeline {pipeline_name}: {str(e)}")
                    raise
            
            return self.pipelines[pipeline_name]
    
    def transform_expression(
        self,
        file: FileStorage,
        target_expression: str,
        session_id: str,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
        model_name: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transform facial expression in uploaded image.
        
        Args:
            file: Uploaded image file
            target_expression: Target expression to transform to
            session_id: User session identifier
            strength: Transformation strength (0.0-1.0)
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            model_name: Model to use for transformation
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary containing transformation results and metadata
        """
        start_time = time.time()
        conversion_id = kwargs.get('conversion_id') # conversion_id는 반드시 전달되어야 함
        if not conversion_id:
            return {'success': False, 'error': 'conversion_id is required'}
        
        try:
            conversion = db.session.query(ConversionHistory).filter_by(id=conversion_id).first()
            if not conversion:
                logger.error(f"Conversion record with id {conversion_id} not found.")
                return {'success': False, 'error': 'Conversion record not found'}
            
            # Step 1: Validate upload
            logger.info(f"Validating upload for conversion {conversion_id}")
            validation_result = self.image_processor.validate_upload(file)
            if not validation_result['valid']:
                conversion.update_status('failed', validation_result['error'])
                db.session.commit()
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'conversion_id': conversion_id
                }
            
            # Step 2: Preprocess image
            logger.info(f"Preprocessing image for conversion {conversion_id}")
            preprocess_result = self.image_processor.preprocess_image(file, target_size=(512, 512))
            if not preprocess_result['success']:
                conversion.update_status('failed', preprocess_result['error'])
                db.session.commit()
                return {
                    'success': False,
                    'error': preprocess_result['error'],
                    'conversion_id': conversion_id
                }
            
            processed_image = preprocess_result['image_data']
            original_metadata = preprocess_result['metadata']
            
            # Step 3: Get or create model
            logger.info(f"Loading model {model_name} for conversion {conversion_id}")
            model = self.get_or_create_model(model_name, **kwargs)
            
            # Step 4: Transform expression
            logger.info(f"Transforming expression to {target_expression} for conversion {conversion_id}")
            transform_result = model.transform_expression(
                image=processed_image,
                target_expression=target_expression,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed
            )
            
            if not transform_result['success']:
                conversion.update_status('failed', transform_result.get('error', 'Transformation failed'))
                db.session.commit()
                return {
                    'success': False,
                    'error': transform_result.get('error', 'Transformation failed'),
                    'conversion_id': conversion_id
                }
            
            # Step 5: Postprocess result
            logger.info(f"Postprocessing result for conversion {conversion_id}")
            postprocess_result_data = self.image_processor.postprocess_result(
                result_image=transform_result['image'],
                original_metadata=original_metadata,
                output_format=kwargs.get('output_format', 'JPEG'),
                quality=kwargs.get('quality', 95)
            )
            
            # Step 6: Save results and update database
            processing_time = time.time() - start_time
            
            conversion.result_file_path = postprocess_result_data.get('output_path')
            conversion.update_status('completed')

            conversion.processing_time = processing_time
            conversion.result_metadata = {
                'output_size': postprocess_result_data['metadata']['file_size'],
                'output_format': postprocess_result_data['metadata']['output_format'],
                'faces_detected': len(transform_result.get('faces', [])),
                'model_used': model_name,
                'processing_time': processing_time
            }
            db.session.commit()
            
            # Update session stats
            self._update_session_stats(session_id)
            
            # Update service stats
            self.stats['total_conversions'] += 1
            self.stats['successful_conversions'] += 1
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (self.stats['successful_conversions'] - 1) + processing_time) /
                self.stats['successful_conversions']
            )
            
            logger.info(f"Conversion {conversion_id} completed successfully in {processing_time:.2f}s")
            
            return {
                'success': True,
                'conversion_id': conversion_id,
                'image': postprocess_result_data['image_data'],
                'metadata': {
                    'original_filename': file.filename,
                    'target_expression': target_expression,
                    'processing_time': processing_time,
                    'faces_detected': len(transform_result.get('faces', [])),
                    'model_used': model_name,
                    'output_format': postprocess_result_data['metadata']['output_format'],
                    'output_size': postprocess_result_data['metadata']['file_size']
                }
            }
            
        except Exception as e:
            # Handle any unexpected errors
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error during transformation: {str(e)}"
            logger.error(f"Conversion {conversion_id} failed: {error_msg}")
            
            # Update database
            try:
                conversion = ConversionHistory.query.get(conversion_id)
                if conversion:
                    conversion.update_status('failed', error_msg)
                    conversion.processing_time = processing_time
                    db.session.commit()
            except Exception as db_error:
                logger.error(f"Failed to update database for failed conversion: {str(db_error)}")
            
            # Update service stats
            self.stats['total_conversions'] += 1
            self.stats['failed_conversions'] += 1
            
            return {
                'success': False,
                'error': error_msg,
                'conversion_id': conversion_id
            }
    
    def batch_transform(
        self,
        files: List[FileStorage],
        expressions: List[str],
        session_id: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transform multiple images with different expressions.
        
        Args:
            files: List of uploaded image files
            expressions: List of target expressions (must match files length)
            session_id: User session identifier
            **kwargs: Additional processing parameters
            
        Returns:
            List of transformation results
        """
        if len(files) != len(expressions):
            raise ValueError("Number of files must match number of expressions")
        
        logger.info(f"Starting batch transformation of {len(files)} images for session {session_id}")
        
        results = []
        for i, (file, expression) in enumerate(zip(files, expressions)):
            logger.info(f"Processing batch item {i+1}/{len(files)}: {expression}")
            result = self.transform_expression(
                file=file,
                target_expression=expression,
                session_id=session_id,
                **kwargs
            )
            results.append(result)
        
        logger.info(f"Batch transformation completed: {len(results)} results")
        return results
    
    def get_conversion_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get conversion history for a session.
        
        Args:
            session_id: User session identifier
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of conversion history records
        """
        try:
            from database.models import get_session_history
            conversions = get_session_history(session_id, limit, offset)
            return [conv.to_dict() for conv in conversions]
        except Exception as e:
            logger.error(f"Failed to get conversion history: {str(e)}")
            return []
    
    def get_conversion_by_id(self, conversion_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific conversion by ID.
        
        Args:
            conversion_id: Conversion identifier
            
        Returns:
            Conversion data or None if not found
        """
        try:
            from database.models import get_conversion_by_id
            conversion = get_conversion_by_id(conversion_id)
            return conversion.to_dict() if conversion else None
        except Exception as e:
            logger.error(f"Failed to get conversion by ID: {str(e)}")
            return None
    
    def get_supported_expressions(self, model_name: str = "default") -> List[str]:
        """
        Get list of supported expressions for a model.
        
        Args:
            model_name: Model to query for supported expressions
            
        Returns:
            List of supported expression names
        """
        try:
            model = self.get_or_create_model(model_name)
            return model.get_supported_expressions()
        except Exception as e:
            logger.error(f"Failed to get supported expressions: {str(e)}")
            return ['happy', 'sad', 'angry', 'surprised', 'neutral']  # Fallback
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service performance statistics.
        
        Returns:
            Dictionary containing service statistics
        """
        return {
            **self.stats,
            'models_loaded': len(self.models),
            'pipelines_loaded': len(self.pipelines),
            'uptime': time.time() - getattr(self, '_start_time', time.time())
        }
    
    def cleanup_resources(self):
        """Clean up resources and temporary files."""
        logger.info("Cleaning up ModelService resources")
        
        # Clear models and pipelines
        with self.model_lock:
            self.models.clear()
            self.pipelines.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clean up temporary files
        try:
            cleanup_temp_files([])  # This will clean up old temp files
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {str(e)}")
    
    def _cache_model_info(self, model_name: str, config: Dict[str, Any]):
        """Cache model information in database."""
        try:
            model_cache = ModelCache(
                model_name=model_name,
                model_type=config.get('model_type', 'stable-diffusion'),
                model_version=config.get('model_version', '1.0'),
                device=config.get('device', 'auto'),
                precision=config.get('precision', 'fp16'),
                model_parameters=config
            )
            model_cache.set_model_parameters(config)
            db.session.add(model_cache)
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to cache model info: {str(e)}")
            db.session.rollback()
    
    def _update_session_stats(self, session_id: str):
        """Update session statistics."""
        try:
            session = UserSession.query.filter_by(session_id=session_id).first()
            if session:
                session.increment_conversion_stats()
                db.session.commit()
        except Exception as e:
            logger.error(f"Failed to update session stats: {str(e)}")


# Global service instance
_model_service = None
_service_lock = threading.Lock()


def get_model_service(config: Optional[Dict[str, Any]] = None) -> ModelService:
    """
    Get or create global ModelService instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ModelService instance
    """
    global _model_service
    
    with _service_lock:
        if _model_service is None:
            _model_service = ModelService(config)
            _model_service._start_time = time.time()
        
        return _model_service


def cleanup_model_service():
    """Clean up global ModelService instance."""
    global _model_service
    
    with _service_lock:
        if _model_service is not None:
            _model_service.cleanup_resources()
            _model_service = None


# Convenience functions for common operations
def transform_single_image(
    file: FileStorage,
    target_expression: str,
    session_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for single image transformation.
    
    Args:
        file: Uploaded image file
        target_expression: Target expression
        session_id: User session ID
        **kwargs: Additional parameters
        
    Returns:
        Transformation result
    """
    service = get_model_service()
    return service.transform_expression(
        file=file,
        target_expression=target_expression,
        session_id=session_id,
        **kwargs
    )


def get_expression_list(model_name: str = "default") -> List[str]:
    """
    Get list of supported expressions.
    
    Args:
        model_name: Model to query
        
    Returns:
        List of expression names
    """
    service = get_model_service()
    return service.get_supported_expressions(model_name)


def get_processing_stats() -> Dict[str, Any]:
    """
    Get processing statistics.
    
    Returns:
        Statistics dictionary
    """
    service = get_model_service()
    return service.get_service_stats()


if __name__ == "__main__":
    # Test the service
    print("Testing ModelService...")
    
    service = get_model_service()
    print(f"Service initialized: {service}")
    print(f"Supported expressions: {service.get_supported_expressions()}")
    print(f"Service stats: {service.get_service_stats()}")
    
    cleanup_model_service()
    print("Service cleanup completed")