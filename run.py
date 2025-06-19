#!/usr/bin/env python3
"""
Financial Dashboard Application
Main entry point for the Flask application with authentication
"""

from app import create_app
from app.models.user import init_db
import os

# Create Flask application
app = create_app()

if __name__ == '__main__':
    # Initialize database on startup
    with app.app_context():
        init_db()
        print("Database initialized!")
        print("\n" + "="*50)
        print("FINANCIAL DASHBOARD STARTED")
        print("="*50)
        print("Default login credentials:")
        print("Admin    - Username: admin    | Password: admin123")
        print("Analyst  - Username: analyst  | Password: analyst123") 
        print("Viewer   - Username: viewer   | Password: viewer123")
        print("="*50)
        print("Access the application at: http://localhost:5000")
        print("="*50 + "\n")
    
    # Run the application
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )