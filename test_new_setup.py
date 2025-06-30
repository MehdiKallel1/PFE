# test_new_setup.py (UPDATED VERSION)
from flask import Flask
from config import config
from app.models_new import db, init_database
from app.models_new.user_model import User

def test_postgresql_setup():
    """Test the new PostgreSQL setup"""
    
    print("🧪 Testing New PostgreSQL Setup")
    print("=" * 50)
    
    # Step 1: Create Flask app
    print("1️⃣ Creating Flask app...")
    try:
        app = Flask(__name__)
        app.config.from_object(config['development'])
        print("✅ Flask app created")
    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        return False
    
    # Step 2: Initialize database
    print("\n2️⃣ Initializing database...")
    try:
        with app.app_context():
            init_database(app)
            print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        print("🔧 Check your DATABASE_URL in .env file")
        return False
    
    # Step 3: Test database connection (FIXED)
    print("\n3️⃣ Testing database connection...")
    try:
        with app.app_context():
            # Test connection (SQLAlchemy 2.0 compatible)
            with db.engine.connect() as conn:
                result = conn.execute(db.text('SELECT 1'))
                print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("🔧 Make sure PostgreSQL is running and credentials are correct")
        print(f"🔧 Check your .env file - DATABASE_URL should be:")
        print(f"   postgresql://dashboard_user:YOUR_PASSWORD@localhost:5432/financial_dashboard_dev")
        return False
    
    # Step 4: Create tables
    print("\n4️⃣ Creating database tables...")
    try:
        with app.app_context():
            db.create_all()
            print("✅ Tables created successfully")
    except Exception as e:
        print(f"❌ Table creation failed: {e}")
        return False
    
    # Step 5: Test user operations
    print("\n5️⃣ Testing user operations...")
    try:
        with app.app_context():
            # Clean up any existing test user first
            existing_user = User.get_by_username('test_admin')
            if existing_user:
                db.session.delete(existing_user)
                db.session.commit()
            
            # Create test user
            test_user = User.create_user(
                username='test_admin',
                email='test@example.com',
                password='test123',
                role='admin'
            )
            
            if test_user:
                print(f"✅ User created: {test_user.username}")
                
                # Test user retrieval
                found_user = User.get_by_username('test_admin')
                if found_user:
                    print("✅ User retrieval working")
                
                # Test password check
                if found_user.check_password('test123'):
                    print("✅ Password verification working")
                
                # Test role methods
                print(f"✅ Role methods: is_admin={found_user.is_admin()}")
                
                # Clean up
                db.session.delete(test_user)
                db.session.commit()
                print("✅ Test user cleaned up")
            else:
                print("❌ User creation failed")
                return False
                
    except Exception as e:
        print(f"❌ User operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed!")
    print("🚀 Ready to proceed with migration!")
    return True

if __name__ == "__main__":
    success = test_postgresql_setup()
    
    if success:
        print("\n" + "="*50)
        print("✅ PostgreSQL setup is working correctly!")
        print("📋 Next steps:")
        print("   1. Migrate existing SQLite data")
        print("   2. Update your main app to use new models")
        print("   3. Test the complete integration")
    else:
        print("\n" + "="*50)
        print("❌ Setup failed. Please fix the errors above.")
        print("💡 Common fixes:")
        print("   - Check your .env file has correct DATABASE_URL")
        print("   - Make sure PostgreSQL is running")
        print("   - Verify database and user exist in PostgreSQL")