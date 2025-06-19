"""
User Model for Authentication System
"""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from datetime import datetime

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, role='viewer', created_at=None):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role  # 'admin', 'analyst', 'viewer'
        self.created_at = created_at or datetime.now()
    
    def check_password(self, password):
        """Check if provided password matches the stored hash"""
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role == 'admin'
    
    def is_analyst(self):
        """Check if user is analyst or admin"""
        return self.role in ['admin', 'analyst']
    
    def can_upload_data(self):
        """Check if user can upload data"""
        return self.role in ['admin', 'analyst']
    
    def can_modify_models(self):
        """Check if user can modify model settings"""
        return self.role == 'admin'
    
    @staticmethod
    def create_user(username, email, password, role='viewer'):
        """Create a new user"""
        password_hash = generate_password_hash(password)
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, role, datetime.now()))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            # Return the new user
            return User(user_id, username, email, password_hash, role)
            
        except sqlite3.IntegrityError:
            return None  # User already exists
        finally:
            conn.close()
    
    @staticmethod
    def get_by_id(user_id):
        """Get user by ID"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Parse datetime string if it exists
            created_at = None
            if row[5]:  # created_at column
                try:
                    if isinstance(row[5], str):
                        created_at = datetime.fromisoformat(row[5].replace('Z', '+00:00'))
                    else:
                        created_at = row[5]
                except:
                    created_at = row[5]  # Keep as string if parsing fails
            
            return User(row[0], row[1], row[2], row[3], row[4], created_at)
        return None
    
    @staticmethod
    def get_by_username(username):
        """Get user by username"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Parse datetime string if it exists
            created_at = None
            if row[5]:  # created_at column
                try:
                    if isinstance(row[5], str):
                        created_at = datetime.fromisoformat(row[5].replace('Z', '+00:00'))
                    else:
                        created_at = row[5]
                except:
                    created_at = row[5]  # Keep as string if parsing fails
            
            return User(row[0], row[1], row[2], row[3], row[4], created_at)
        return None
    
    @staticmethod
    def get_by_email(email):
        """Get user by email"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Parse datetime string if it exists
            created_at = None
            if row[5]:  # created_at column
                try:
                    if isinstance(row[5], str):
                        created_at = datetime.fromisoformat(row[5].replace('Z', '+00:00'))
                    else:
                        created_at = row[5]
                except:
                    created_at = row[5]  # Keep as string if parsing fails
            
            return User(row[0], row[1], row[2], row[3], row[4], created_at)
        return None
    
    @staticmethod
    def get_all_users():
        """Get all users (admin only)"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        users = []
        for row in rows:
            # Parse datetime string if it exists
            created_at = None
            if row[5]:  # created_at column
                try:
                    if isinstance(row[5], str):
                        created_at = datetime.fromisoformat(row[5].replace('Z', '+00:00'))
                    else:
                        created_at = row[5]
                except:
                    created_at = row[5]  # Keep as string if parsing fails
            
            users.append(User(row[0], row[1], row[2], row[3], row[4], created_at))
        return users


def get_db_connection():
    """Get database connection"""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'users.db')
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the user database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin user if no users exist
    cursor.execute('SELECT COUNT(*) FROM users')
    user_count = cursor.fetchone()[0]
    
    if user_count == 0:
        print("Creating default admin user...")
        default_admin = User.create_user(
            username='admin',
            email='admin@dashboard.com',
            password='admin123',  # Change this in production!
            role='admin'
        )
        print("Default admin user created:")
        print("Username: admin")
        print("Password: admin123")
        print("Please change the password after first login!")
        
        # Create sample analyst user
        User.create_user(
            username='analyst',
            email='analyst@dashboard.com',
            password='analyst123',
            role='analyst'
        )
        
        # Create sample viewer user
        User.create_user(
            username='viewer',
            email='viewer@dashboard.com',
            password='viewer123',
            role='viewer'
        )
        
        print("Sample users created - analyst/analyst123 and viewer/viewer123")
    
    conn.commit()
    conn.close()


if __name__ == '__main__':
    # Initialize database when run directly
    init_db()