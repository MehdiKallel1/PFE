# app/models_new/user_model.py
from . import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

class User(UserMixin, db.Model):
    """User model for PostgreSQL"""
    __tablename__ = 'users'
    
    # Basic fields
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Role: simple string for now (we'll enhance later)
    role = db.Column(db.String(20), default='viewer', nullable=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    # Password methods
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)
    
    # Role methods (same as your current ones)
    def is_admin(self):
        return self.role == 'admin'
    
    def is_analyst(self):
        return self.role in ['admin', 'analyst']
    
    def can_upload_data(self):
        return self.role in ['admin', 'analyst']
    
    def can_modify_models(self):
        return self.role == 'admin'
    
    # Update login time
    def update_last_login(self):
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    # Class methods
    @classmethod
    def create_user(cls, username, email, password, role='viewer'):
        """Create new user"""
        try:
            user = cls(username=username, email=email, role=role)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            return user
        except Exception as e:
            db.session.rollback()
            print(f"Error creating user: {e}")
            return None
    
    @classmethod
    def get_by_username(cls, username):
        """Get user by username"""
        return cls.query.filter_by(username=username, is_active=True).first()
    
    @classmethod
    def get_by_email(cls, email):
        """Get user by email"""
        return cls.query.filter_by(email=email, is_active=True).first()
    
    @classmethod
    def get_by_id(cls, user_id):
        """Get user by ID"""
        return cls.query.get(int(user_id))
    
    @classmethod
    def get_all_users(cls):
        """Get all users"""
        return cls.query.filter_by(is_active=True).order_by(cls.created_at.desc()).all()