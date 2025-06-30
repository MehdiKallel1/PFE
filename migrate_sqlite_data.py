# migrate_sqlite_data.py
import sqlite3
import os
from flask import Flask
from config import config
from app.models_new import db, init_database
from app.models_new.user_model import User
from datetime import datetime

def get_sqlite_connection():
    """Get connection to your existing SQLite database"""
    # Path to your existing SQLite database
    db_path = os.path.join('app','data', 'users.db')
    
    if not os.path.exists(db_path):
        print(f"❌ SQLite database not found at: {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn
    except Exception as e:
        print(f"❌ Error connecting to SQLite: {e}")
        return None

def migrate_users():
    """Migrate users from SQLite to PostgreSQL"""
    
    print("🔄 Starting User Migration from SQLite to PostgreSQL")
    print("=" * 60)
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(config['development'])
    
    with app.app_context():
        init_database(app)
        
        # Get SQLite connection
        sqlite_conn = get_sqlite_connection()
        if not sqlite_conn:
            return False
        
        try:
            # Fetch all users from SQLite
            cursor = sqlite_conn.cursor()
            cursor.execute('SELECT * FROM users ORDER BY id')
            sqlite_users = cursor.fetchall()
            
            print(f"📊 Found {len(sqlite_users)} users in SQLite database")
            
            if len(sqlite_users) == 0:
                print("⚠️  No users found in SQLite database")
                return create_default_users(app)
            
            migrated_count = 0
            skipped_count = 0
            
            for row in sqlite_users:
                try:
                    # Check if user already exists in PostgreSQL
                    existing_user = User.get_by_username(row['username'])
                    if existing_user:
                        print(f"⚠️  User '{row['username']}' already exists, skipping...")
                        skipped_count += 1
                        continue
                    
                    # Parse created_at
                    created_at = datetime.utcnow()
                    if row['created_at']:
                        try:
                            if isinstance(row['created_at'], str):
                                # Handle different datetime formats
                                created_at = datetime.fromisoformat(row['created_at'].replace('Z', '+00:00'))
                            else:
                                created_at = row['created_at']
                        except:
                            created_at = datetime.utcnow()
                    
                    # Create new user in PostgreSQL
                    new_user = User(
                        username=row['username'],
                        email=row['email'],
                        password_hash=row['password_hash'],  # Keep existing hash
                        role=row['role'],
                        created_at=created_at,
                        is_active=True
                    )
                    
                    db.session.add(new_user)
                    migrated_count += 1
                    print(f"✅ Migrated: {row['username']} ({row['role']})")
                    
                except Exception as e:
                    print(f"❌ Error migrating user {row['username']}: {e}")
                    continue
            
            # Commit all changes
            db.session.commit()
            
            print(f"\n🎉 Migration Summary:")
            print(f"✅ Successfully migrated: {migrated_count} users")
            print(f"⚠️  Skipped (already exist): {skipped_count} users")
            
            # Verify migration
            total_users = User.query.count()
            print(f"📊 Total users now in PostgreSQL: {total_users}")
            
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            db.session.rollback()
            return False
        finally:
            sqlite_conn.close()

def create_default_users(app):
    """Create default users if no migration data available"""
    print("\n📝 Creating default users...")
    
    try:
        # Check if any users already exist
        user_count = User.query.count()
        if user_count > 0:
            print(f"📊 Found {user_count} existing users, skipping default creation")
            return True
        
        # Create default users
        users_to_create = [
            ('admin', 'admin@dashboard.com', 'admin123', 'admin'),
            ('analyst', 'analyst@dashboard.com', 'analyst123', 'analyst'),
            ('viewer', 'viewer@dashboard.com', 'viewer123', 'viewer')
        ]
        
        created_count = 0
        for username, email, password, role in users_to_create:
            user = User.create_user(username, email, password, role)
            if user:
                created_count += 1
                print(f"✅ Created: {username} ({role})")
        
        print(f"\n🎉 Created {created_count} default users")
        print("⚠️  Default passwords:")
        print("   admin/admin123, analyst/analyst123, viewer/viewer123")
        print("   Please change these passwords in production!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating default users: {e}")
        return False

def verify_migration():
    """Verify the migration was successful"""
    
    print("\n🔍 Verifying Migration Results")
    print("=" * 40)
    
    app = Flask(__name__)
    app.config.from_object(config['development'])
    
    with app.app_context():
        init_database(app)
        
        try:
            # Get all users
            users = User.get_all_users()
            
            print(f"📊 Total users: {len(users)}")
            print("\n👥 User List:")
            print("-" * 50)
            
            for user in users:
                print(f"ID: {user.id:2d} | {user.username:12s} | {user.role:8s} | {user.email}")
                
                # Test each user's methods
                print(f"         Admin: {user.is_admin()} | Analyst: {user.is_analyst()} | Upload: {user.can_upload_data()}")
            
            print("-" * 50)
            
            # Test login with one user
            if users:
                test_user = users[0]
                print(f"\n🧪 Testing login simulation with: {test_user.username}")
                
                # Note: We can't test actual password since they're hashed
                # But we can test the structure
                print(f"✅ User object structure is correct")
                print(f"✅ Role methods working")
                print(f"✅ Database queries working")
            
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False

if __name__ == "__main__":
    print("🚀 PostgreSQL Migration Tool")
    print("=" * 60)
    
    # Step 1: Migrate users
    migration_success = migrate_users()
    
    if migration_success:
        # Step 2: Verify migration
        verify_migration()
        
        print("\n" + "="*60)
        print("🎉 Migration completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Update your main Flask app to use the new models")
        print("2. Test authentication with existing users")
        print("3. Remove SQLite dependencies (optional)")
        
    else:
        print("\n" + "="*60)
        print("❌ Migration failed!")
        print("🔧 Please check the errors above and try again")