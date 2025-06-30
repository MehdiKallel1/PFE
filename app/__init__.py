# app/__init__.py (Final PostgreSQL Version)
from flask import Flask
from flask_login import LoginManager
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_app(config_name='development'):
    """Application factory pattern"""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Load configuration
    from config import config
    app.config.from_object(config[config_name])
    
    # Initialize database (PostgreSQL)
    from app.models_new import init_database
    init_database(app)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        from app.models_new.user_model import User
        return User.get_by_id(int(user_id))
    
    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app