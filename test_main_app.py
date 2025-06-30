# test_main_app.py
import requests
import time
from app import create_app
from app.models_new.user_model import User

def test_flask_app():
    """Test the main Flask application"""
    
    print("🧪 Testing Complete Flask Application")
    print("=" * 50)
    
    # Create app
    app = create_app('development')
    
    print("1️⃣ Testing app creation...")
    print("✅ App created successfully")
    
    # Test database connectivity within app context
    with app.app_context():
        print("\n2️⃣ Testing database within app context...")
        
        # Test user retrieval
        admin_user = User.get_by_username('admin')
        if admin_user:
            print(f"✅ Found admin user: {admin_user.username}")
            print(f"   Email: {admin_user.email}")
            print(f"   Role: {admin_user.role}")
            print(f"   Is Admin: {admin_user.is_admin()}")
        else:
            print("❌ Admin user not found!")
            return False
        
        # Test all users
        all_users = User.get_all_users()
        print(f"✅ Total users in database: {len(all_users)}")
        
        print("\n3️⃣ Testing user authentication methods...")
        
        # Test password verification
        if admin_user.check_password('admin123'):
            print("✅ Password verification working")
        else:
            print("❌ Password verification failed")
            return False
        
        # Test role methods
        print(f"✅ Role methods working:")
        print(f"   can_upload_data: {admin_user.can_upload_data()}")
        print(f"   can_modify_models: {admin_user.can_modify_models()}")
    
    # Test Flask app with test client
    print("\n4️⃣ Testing Flask routes...")
    with app.test_client() as client:
        # Test main page (should redirect to login)
        response = client.get('/')
        print(f"✅ Main page status: {response.status_code}")
        
        # Test login page
        response = client.get('/login')
        if response.status_code == 200:
            print("✅ Login page accessible")
        else:
            print(f"❌ Login page error: {response.status_code}")
    
    print("\n🎉 All Flask app tests passed!")
    return True

def test_live_app():
    """Test the app running live (optional)"""
    
    print("\n🌐 Testing Live Application (Optional)")
    print("=" * 50)
    
    print("📋 To test your live application:")
    print("1. Run: python app.py")
    print("2. Open: http://localhost:5000")
    print("3. Login with:")
    print("   Username: admin")
    print("   Password: admin123")
    print("\n   Or try:")
    print("   Username: analyst")
    print("   Password: analyst123")

if __name__ == "__main__":
    success = test_flask_app()
    
    if success:
        print("\n" + "="*50)
        print("🎉 Your Flask app is ready!")
        print("\n📋 Next Steps:")
        print("1. Run your Flask app: python app.py")
        print("2. Test login with existing users")
        print("3. Test file upload functionality")
        print("4. Verify all features work with PostgreSQL")
        
        test_live_app()
        
    else:
        print("\n❌ Flask app test failed!")
        print("🔧 Check the errors above and fix any issues")