from flask import Flask
from flask_login import LoginManager
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_app():
    app = Flask(__name__, 
               template_folder='templates',
               static_folder='static')
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this-in-production'
    app.config['WTF_CSRF_ENABLED'] = True
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        from app.models.user import User
        return User.get_by_id(int(user_id))
    
    # Initialize database
    with app.app_context():
        from app.models.user import init_db
        init_db()
    
    # Import and register blueprints/routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app