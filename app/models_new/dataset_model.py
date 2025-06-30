# app/models_new/dataset_model.py
from . import db
from datetime import datetime

class Dataset(db.Model):
    """Model for uploaded datasets"""
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    
    # User who uploaded
    uploaded_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # File information
    file_size = db.Column(db.BigInteger)
    record_count = db.Column(db.Integer)
    columns_metadata = db.Column(db.JSON)  # Store column info as JSON
    
    # Processing status
    processing_status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    
    # Date range of the data
    date_range_start = db.Column(db.Date)
    date_range_end = db.Column(db.Date)
    
    # Description
    description = db.Column(db.Text)
    
    def __repr__(self):
        return f'<Dataset {self.name}>'
    
    @classmethod
    def create_dataset(cls, name, filename, file_path, uploaded_by_id, **kwargs):
        """Create a new dataset record"""
        try:
            dataset = cls(
                name=name,
                filename=filename,
                file_path=file_path,
                uploaded_by_id=uploaded_by_id,
                **kwargs
            )
            db.session.add(dataset)
            db.session.commit()
            return dataset
        except Exception as e:
            db.session.rollback()
            print(f"Error creating dataset: {e}")
            return None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'filename': self.filename,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'file_size': self.file_size,
            'record_count': self.record_count,
            'processing_status': self.processing_status,
            'date_range_start': self.date_range_start.isoformat() if self.date_range_start else None,
            'date_range_end': self.date_range_end.isoformat() if self.date_range_end else None,
            'uploaded_by_id': self.uploaded_by_id
        }