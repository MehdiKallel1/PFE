# config.py - Add this TestingConfig class and update the config dictionary

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    
    # Create directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_ECHO = True  # Show SQL queries in console

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_ECHO = False

# ADD THIS NEW CLASS:
class TestingConfig(Config):
    """Testing configuration for pytest"""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # In-memory database for tests
    SQLALCHEMY_ECHO = False  # Don't show SQL during tests
    WTF_CSRF_ENABLED = False  # Disable CSRF for testing
    SECRET_KEY = 'test-secret-key'

# UPDATE THIS DICTIONARY - ADD 'testing':
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,  # ADD THIS LINE
    'default': DevelopmentConfig
}