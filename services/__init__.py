"""
Services Package for Facial Expression App

This package contains all service layer components for the facial expression
transformation application, including image processing, model inference,
and storage operations.

Services:
- ImageProcessor: Image preprocessing and postprocessing
- ModelService: AI model inference and management
- StorageService: File and database operations
"""

from .image_processor import ImageProcessor
from .model_service import ModelService
from .storage_service import StorageService

__all__ = [
    'ImageProcessor',
    'ModelService', 
    'StorageService'
]

# Service factory functions for dependency injection
def create_image_processor(config=None):
    """
    Factory function to create ImageProcessor instance
    
    Args:
        config: Configuration object with image processing settings
        
    Returns:
        ImageProcessor: Configured image processor instance
    """
    return ImageProcessor(config)

def create_model_service(config=None):
    """
    Factory function to create ModelService instance
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        ModelService: Configured model service instance
    """
    return ModelService(config)

def create_storage_service(config=None, db=None):
    """
    Factory function to create StorageService instance
    
    Args:
        config: Configuration object with storage settings
        db: Database instance for storage operations
        
    Returns:
        StorageService: Configured storage service instance
    """
    return StorageService(config, db)

# Service registry for centralized service management
class ServiceRegistry:
    """
    Central registry for managing service instances
    Implements singleton pattern for service lifecycle management
    """
    
    _instance = None
    _services = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
        return cls._instance
    
    def register_service(self, name, service_instance):
        """
        Register a service instance
        
        Args:
            name (str): Service name
            service_instance: Service instance to register
        """
        self._services[name] = service_instance
    
    def get_service(self, name):
        """
        Get registered service instance
        
        Args:
            name (str): Service name
            
        Returns:
            Service instance or None if not found
        """
        return self._services.get(name)
    
    def initialize_services(self, config=None, db=None):
        """
        Initialize all core services
        
        Args:
            config: Configuration object
            db: Database instance
        """
        # Initialize image processor
        image_processor = create_image_processor(config)
        self.register_service('image_processor', image_processor)
        
        # Initialize model service
        model_service = create_model_service(config)
        self.register_service('model_service', model_service)
        
        # Initialize storage service
        storage_service = create_storage_service(config, db)
        self.register_service('storage_service', storage_service)
    
    def cleanup_services(self):
        """
        Cleanup all registered services
        """
        for service_name, service in self._services.items():
            if hasattr(service, 'cleanup'):
                try:
                    service.cleanup()
                except Exception as e:
                    print(f"Error cleaning up service {service_name}: {e}")
        
        self._services.clear()

# Global service registry instance
service_registry = ServiceRegistry()

# Convenience functions for accessing services
def get_image_processor():
    """Get the registered image processor service"""
    return service_registry.get_service('image_processor')

def get_model_service():
    """Get the registered model service"""
    return service_registry.get_service('model_service')

def get_storage_service():
    """Get the registered storage service"""
    return service_registry.get_service('storage_service')