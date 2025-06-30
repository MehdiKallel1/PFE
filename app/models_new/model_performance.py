# app/models_new/model_performance.py
from . import db
from datetime import datetime

class ModelPerformance(db.Model):
    """Model for storing ML model performance metrics"""
    __tablename__ = 'model_performance'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Link to dataset
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    
    # Model information
    metric_name = db.Column(db.String(100), nullable=False)  # Revenue, Profit, etc.
    model_name = db.Column(db.String(100), nullable=False)   # RandomForest, XGBoost, etc.
    model_type = db.Column(db.String(50))                    # ensemble, linear, neural
    
    # Performance metrics
    r2_score = db.Column(db.Numeric(8,6))
    mae_score = db.Column(db.Numeric(12,4))
    mape_score = db.Column(db.Numeric(8,4))
    rmse_score = db.Column(db.Numeric(12,4))
    
    # Additional metrics
    feature_importance = db.Column(db.JSON)
    training_time_seconds = db.Column(db.Numeric(10,4))
    
    # Selection information
    is_best_model = db.Column(db.Boolean, default=False)
    composite_score = db.Column(db.Numeric(8,6))
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelPerformance {self.model_name} for {self.metric_name}>'
    
    @classmethod
    def save_performance(cls, dataset_id, metric_name, model_name, performance_data):
        """Save model performance data"""
        try:
            performance = cls(
                dataset_id=dataset_id,
                metric_name=metric_name,
                model_name=model_name,
                r2_score=performance_data.get('r2', 0),
                mae_score=performance_data.get('mae', 0),
                mape_score=performance_data.get('mape', 0),
                rmse_score=performance_data.get('rmse', 0),
                feature_importance=performance_data.get('feature_importance', {}),
                training_time_seconds=performance_data.get('training_time', 0),
                composite_score=performance_data.get('composite_score', 0)
            )
            db.session.add(performance)
            db.session.commit()
            return performance
        except Exception as e:
            db.session.rollback()
            print(f"Error saving performance: {e}")
            return None
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'r2_score': float(self.r2_score) if self.r2_score else 0,
            'mae_score': float(self.mae_score) if self.mae_score else 0,
            'mape_score': float(self.mape_score) if self.mape_score else 0,
            'is_best_model': self.is_best_model,
            'feature_importance': self.feature_importance,
            'created_at': self.created_at.isoformat()
        }