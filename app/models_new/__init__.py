# app/models_new/__init__.py (Updated)
from flask_sqlalchemy import SQLAlchemy

# Create database instance
db = SQLAlchemy()

def init_database(app):
    """Initialize database with Flask app"""
    db.init_app(app)
    
    # Import models after db is initialized
    from . import user_model
    from . import dataset_model
    from . import model_performance
    
    return db

# Export for easy importing
__all__ = ['db', 'init_database']