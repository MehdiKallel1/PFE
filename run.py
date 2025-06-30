# NEW run.py (Updated for PostgreSQL)
from app import create_app

# Create app using the factory
app = create_app('development')

if __name__ == '__main__':
    print("🚀 Starting Financial Dashboard with PostgreSQL...")
    print("📊 Database: PostgreSQL")
    print("🌐 URL: http://localhost:5000")
    print("👤 Login: admin/admin123")
    print("-" * 50)
    
    # Run the app (database is already initialized in create_app)
    app.run(debug=True, host='0.0.0.0', port=5000)