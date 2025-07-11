"""
Configuration settings for the Multimodal AI API
"""

import os
from typing import Optional

class Settings:
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "Multimodal AI API for Policy Analysis and Trade Decisions"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = """
    A multimodal AI API for processing time-series, geospatial, image, and other data to support policy analysis and trade decision-making. 
    It provides predictive analytics, explainability, and strategic recommendations for policymakers and traders.
    """
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Security Settings
    API_KEY: str = os.getenv("API_KEY", "demo-api-key")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Database Settings
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Model Settings
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./models")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # CORS Settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "True").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    
    # Analysis Settings
    DEFAULT_TIME_HORIZON: str = os.getenv("DEFAULT_TIME_HORIZON", "1y")
    DEFAULT_EXPLAINABILITY_LEVEL: str = os.getenv("DEFAULT_EXPLAINABILITY_LEVEL", "basic")
    MAX_PREDICTION_STEPS: int = int(os.getenv("MAX_PREDICTION_STEPS", "12"))
    
    # File Storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    VISUALIZATION_DIR: str = os.getenv("VISUALIZATION_DIR", "./visualizations")
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL with fallback to SQLite"""
        if cls.DATABASE_URL:
            return cls.DATABASE_URL
        return "sqlite:///./multimodal_api.db"
    
    @classmethod
    def validate_settings(cls) -> None:
        """Validate required settings"""
        required_dirs = [cls.UPLOAD_DIR, cls.VISUALIZATION_DIR, cls.MODEL_CACHE_DIR]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)

# Create settings instance
settings = Settings()

# Validate settings on import
settings.validate_settings() 