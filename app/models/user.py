# app/models/user.py (New PostgreSQL ORM version)
from app.models_new import db 
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from enum import Enum
import sqlite3
import os

class UserRole(Enum):
    ADMIN = 'admin'
    ANALYST = 'analyst'
    VIEWER = 'viewer'

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    last_login = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches stored hash"""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role == UserRole.ADMIN
    
    def is_analyst(self):
        """Check if user is analyst or admin"""
        return self.role in [UserRole.ADMIN, UserRole.ANALYST]
    
    def can_upload_data(self):
        """Check if user can upload data"""
        return self.role in [UserRole.ADMIN, UserRole.ANALYST]
    
    def can_modify_models(self):
        """Check if user can modify model settings"""
        return self.role == UserRole.ADMIN
    
    def to_dict(self):
        """Convert user to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }
    
    @classmethod
    def create_user(cls, username, email, password, role='viewer'):
        """Create a new user"""
        try:
            # Convert string role to enum
            if isinstance(role, str):
                role_enum = UserRole(role.lower())
            else:
                role_enum = role
            
            user = cls(
                username=username,
                email=email,
                role=role_enum
            )
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
    def get_all_users(cls):
        """Get all users (admin only)"""
        return cls.query.order_by(cls.created_at.desc()).all()
    
    @classmethod
    def get_by_id(cls, user_id):
        """Get user by ID"""
        return cls.query.get(int(user_id))

# =============================================================================
# MIGRATION UTILITIES
# =============================================================================

def get_sqlite_db_connection():
    """Get SQLite database connection for migration"""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'users.db')
    
    if not os.path.exists(db_path):
        print(f"SQLite database not found at: {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def migrate_users_from_sqlite():
    """Migrate users from SQLite to PostgreSQL"""
    print("🔄 Starting user migration from SQLite to PostgreSQL...")
    
    # Get SQLite connection
    sqlite_conn = get_sqlite_db_connection()
    if not sqlite_conn:
        print("❌ Could not connect to SQLite database")
        return False
    
    try:
        # Fetch all users from SQLite
        cursor = sqlite_conn.cursor()
        cursor.execute('SELECT * FROM users')
        sqlite_users = cursor.fetchall()
        
        print(f"📊 Found {len(sqlite_users)} users in SQLite database")
        
        migrated_count = 0
        skipped_count = 0
        
        for row in sqlite_users:
            try:
                # Parse created_at
                created_at = datetime.utcnow()
                if row['created_at']:
                    try:
                        if isinstance(row['created_at'], str):
                            created_at = datetime.fromisoformat(row['created_at'].replace('Z', '+00:00'))
                        else:
                            created_at = row['created_at']
                    except:
                        created_at = datetime.utcnow()
                
                # Check if user already exists in PostgreSQL
                existing_user = User.query.filter_by(username=row['username']).first()
                if existing_user:
                    print(f"⚠️  User '{row['username']}' already exists, skipping...")
                    skipped_count += 1
                    continue
                
                # Convert role to enum
                try:
                    role_enum = UserRole(row['role'].lower())
                except ValueError:
                    role_enum = UserRole.VIEWER
                
                # Create new user in PostgreSQL
                new_user = User(
                    username=row['username'],
                    email=row['email'],
                    password_hash=row['password_hash'],  # Keep existing hash
                    role=role_enum,
                    created_at=created_at,
                    is_active=True
                )
                
                db.session.add(new_user)
                migrated_count += 1
                print(f"✅ Migrated user: {row['username']} ({row['role']})")
                
            except Exception as e:
                print(f"❌ Error migrating user {row['username']}: {e}")
                continue
        
        # Commit all changes
        db.session.commit()
        
        print(f"\n🎉 Migration completed!")
        print(f"✅ Migrated: {migrated_count} users")
        print(f"⚠️  Skipped: {skipped_count} users")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        db.session.rollback()
        return False
    finally:
        sqlite_conn.close()

def create_default_users():
    """Create default users if none exist"""
    user_count = User.query.count()
    
    if user_count == 0:
        print("📝 Creating default users...")
        
        # Create default admin user
        admin_user = User.create_user(
            username='admin',
            email='admin@dashboard.com',
            password='admin123',
            role='admin'
        )
        
        if admin_user:
            print("✅ Default admin user created:")
            print("   Username: admin")
            print("   Password: admin123")
            print("   ⚠️  Please change the password after first login!")
        
        # Create sample analyst user
        analyst_user = User.create_user(
            username='analyst',
            email='analyst@dashboard.com',
            password='analyst123',
            role='analyst'
        )
        
        # Create sample viewer user
        viewer_user = User.create_user(
            username='viewer',
            email='viewer@dashboard.com',
            password='viewer123',
            role='viewer'
        )
        
        if analyst_user and viewer_user:
            print("✅ Sample users created:")
            print("   analyst/analyst123 and viewer/viewer123")
        
        return True
    else:
        print(f"📊 Found {user_count} existing users, skipping default user creation")
        return False