# test_new_models.py
from flask import Flask
from config import config
from app.models_new import db, init_database
from app.models_new.user_model import User
from app.models_new.dataset_model import Dataset
from app.models_new.model_performance import ModelPerformance
from datetime import datetime, date

# test_new_models.py (CORRECTED CLEANUP SECTION)
def test_new_models():
    """Test the new database models"""
    
    print("🧪 Testing New Database Models")
    print("=" * 50)
    
    app = Flask(__name__)
    app.config.from_object(config['development'])
    
    with app.app_context():
        init_database(app)
        
        # Create all tables
        print("1️⃣ Creating database tables...")
        db.create_all()
        print("✅ Tables created")
        
        # Test User model (should already work)
        print("\n2️⃣ Testing User model...")
        test_user = User.get_by_username('admin')
        if not test_user:
            test_user = User.create_user('test_admin', 'test@admin.com', 'test123', 'admin')
        
        if test_user:
            print(f"✅ User test passed: {test_user.username}")
        
        # Test Dataset model
        print("\n3️⃣ Testing Dataset model...")
        test_dataset = Dataset.create_dataset(
            name="Test Financial Data",
            filename="test_data.csv",
            file_path="/uploads/test_data.csv",
            uploaded_by_id=test_user.id,
            file_size=1024,
            record_count=100,
            columns_metadata={"Revenue": "float", "Profit": "float"},
            processing_status="completed",
            date_range_start=date(2020, 1, 1),
            date_range_end=date(2024, 12, 31),
            description="Test dataset for ML models"
        )
        
        if test_dataset:
            print(f"✅ Dataset test passed: {test_dataset.name}")
            print(f"   ID: {test_dataset.id}")
        
        # Test ModelPerformance model
        print("\n4️⃣ Testing ModelPerformance model...")
        performance_data = {
            'r2': 0.85,
            'mae': 1000.5,
            'mape': 5.2,
            'rmse': 1200.0,
            'feature_importance': {
                'PIB_US_Courants': 0.4,
                'Masse_Monetaire': 0.3,
                'Credit_Interieur': 0.3
            },
            'training_time': 12.5,
            'composite_score': 0.82
        }
        
        test_performance = ModelPerformance.save_performance(
            dataset_id=test_dataset.id,
            metric_name="Revenue",
            model_name="RandomForest",
            performance_data=performance_data
        )
        
        if test_performance:
            print(f"✅ ModelPerformance test passed: {test_performance.model_name}")
            print(f"   R² Score: {test_performance.r2_score}")
        
        # Test relationships and queries
        print("\n5️⃣ Testing database relationships...")
        
        # Get user's datasets
        user_datasets = Dataset.query.filter_by(uploaded_by_id=test_user.id).all()
        print(f"✅ User has {len(user_datasets)} datasets")
        
        # Get dataset's performance records
        dataset_performances = ModelPerformance.query.filter_by(dataset_id=test_dataset.id).all()
        print(f"✅ Dataset has {len(dataset_performances)} performance records")
        
        # Test JSON serialization
        print("\n6️⃣ Testing JSON serialization...")
        dataset_dict = test_dataset.to_dict()
        performance_dict = test_performance.to_dict()
        print("✅ JSON serialization working")
        
        # Clean up test data (CORRECTED ORDER)
        print("\n7️⃣ Cleaning up test data...")
        try:
            # Delete in correct order: child records first, then parent records
            
            # 1. Delete model performance records first (they reference datasets)
            if test_performance:
                db.session.delete(test_performance)
                print("✅ Deleted model performance record")
            
            # 2. Delete dataset (now safe because no references)
            if test_dataset:
                db.session.delete(test_dataset)
                print("✅ Deleted dataset record")
            
            # 3. Delete test user if it was created for testing
            if test_user and test_user.username == 'test_admin':
                db.session.delete(test_user)
                print("✅ Deleted test user")
            
            # Commit all deletions
            db.session.commit()
            print("✅ All test data cleaned up successfully")
            
        except Exception as cleanup_error:
            print(f"⚠️  Cleanup error: {cleanup_error}")
            db.session.rollback()
            print("🔄 Rolled back any partial cleanup")
        
        print("\n🎉 All model tests passed!")
        return True

if __name__ == "__main__":
    success = test_new_models()
    
    if success:
        print("\n" + "="*50)
        print("✅ New models are working perfectly!")
        print("\n📋 Next steps:")
        print("1. Test your main Flask app")
        print("2. Update routes to use new models")
        print("3. Set up Flask-Migrate for future changes")
    else:
        print("\n❌ Model tests failed!")