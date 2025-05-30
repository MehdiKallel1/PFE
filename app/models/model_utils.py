"""
Model Utils - Shared utilities for model operations and data processing
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
import pickle
import os

logger = logging.getLogger(__name__)

class ModelUtils:
    """Utility functions for model operations"""
    
    @staticmethod
    def validate_data_for_modeling(X, y, min_samples=12):
        """
        Validate that data is suitable for model training
        
        Args:
            X: Feature matrix
            y: Target vector
            min_samples: Minimum number of samples required
            
        Returns:
            bool: True if data is valid, False otherwise
            list: List of validation issues if any
        """
        issues = []
        
        # Check minimum samples
        if len(X) < min_samples:
            issues.append(f"Insufficient data: {len(X)} samples, need at least {min_samples}")
        
        # Check for missing values
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            issues.append(f"Missing values in features: {missing_cols}")
        
        if y.isnull().any():
            issues.append("Missing values in target variable")
        
        # Check for infinite values
        if np.isinf(X.values).any():
            issues.append("Infinite values found in features")
        
        if np.isinf(y.values).any():
            issues.append("Infinite values found in target variable")
        
        # Check variance
        low_variance_cols = []
        for col in X.columns:
            if X[col].var() < 1e-10:
                low_variance_cols.append(col)
        
        if low_variance_cols:
            issues.append(f"Features with very low variance: {low_variance_cols}")
        
        # Check target variance
        if y.var() < 1e-10:
            issues.append("Target variable has very low variance")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def prepare_model_data(merged_df, target_columns, exclude_columns=None):
        """
        Prepare data for model training by separating features and targets
        
        Args:
            merged_df: Combined dataframe with features and targets
            target_columns: List of target column names
            exclude_columns: Additional columns to exclude from features
            
        Returns:
            tuple: (feature_df, target_dict, feature_names)
        """
        if exclude_columns is None:
            exclude_columns = ['Date', 'is_predicted']
        
        # Combine target columns and exclude columns
        all_exclude = list(set(target_columns + exclude_columns))
        
        # Get feature columns
        feature_columns = [col for col in merged_df.columns if col not in all_exclude]
        
        # Prepare features
        X = merged_df[feature_columns].copy()
        
        # Prepare targets
        targets = {}
        for target in target_columns:
            if target in merged_df.columns:
                targets[target] = merged_df[target].copy()
        
        logger.info(f"Prepared data: {len(feature_columns)} features, {len(targets)} targets")
        logger.info(f"Feature columns: {feature_columns}")
        
        return X, targets, feature_columns
    
    @staticmethod
    def calculate_prediction_intervals(predictions, confidence_level=0.95):
        """
        Calculate prediction intervals for uncertainty quantification
        
        Args:
            predictions: Array of prediction values
            confidence_level: Confidence level for intervals (default 0.95)
            
        Returns:
            dict: Lower and upper bounds for predictions
        """
        # Simple approach: use standard deviation of predictions
        pred_std = np.std(predictions)
        pred_mean = np.mean(predictions)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        margin = z_score * pred_std
        
        return {
            'lower_bound': [pred - margin for pred in predictions],
            'upper_bound': [pred + margin for pred in predictions],
            'confidence_level': confidence_level,
            'margin': float(margin)
        }
    
    @staticmethod
    def format_model_results_for_json(results):
        """
        Format model results for JSON serialization
        
        Args:
            results: Model results dictionary
            
        Returns:
            dict: JSON-serializable results
        """
        def convert_value(obj):
            """Convert numpy types to Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, datetime):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_value(item) for item in obj]
            else:
                return obj
        
        return convert_value(results)
    
    @staticmethod
    def get_model_memory_usage(model):
        """
        Estimate memory usage of a trained model
        
        Args:
            model: Trained model object
            
        Returns:
            dict: Memory usage information
        """
        try:
            # Serialize model to estimate size
            model_bytes = pickle.dumps(model)
            size_mb = len(model_bytes) / (1024 * 1024)
            
            return {
                'size_mb': round(size_mb, 2),
                'size_category': ModelUtils._categorize_model_size(size_mb)
            }
        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")
            return {
                'size_mb': 0,
                'size_category': 'Unknown'
            }
    
    @staticmethod
    def _categorize_model_size(size_mb):
        """Categorize model size"""
        if size_mb < 1:
            return 'Small'
        elif size_mb < 10:
            return 'Medium'
        elif size_mb < 100:
            return 'Large'
        else:
            return 'Very Large'
    
    @staticmethod
    def create_model_metadata(model_name, model_config, performance_metrics, training_time=None):
        """
        Create comprehensive metadata for a trained model
        
        Args:
            model_name: Name of the model
            model_config: Model configuration dictionary
            performance_metrics: Performance metrics dictionary
            training_time: Time taken to train the model
            
        Returns:
            dict: Complete model metadata
        """
        metadata = {
            'model_name': model_name,
            'model_type': model_config.get('description', 'Unknown'),
            'complexity': model_config.get('complexity', 'Unknown'),
            'needs_scaling': model_config.get('needs_scaling', False),
            'performance': performance_metrics,
            'training_timestamp': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'quality_assessment': ModelUtils._assess_model_quality(performance_metrics)
        }
        
        return metadata
    
    @staticmethod
    def _assess_model_quality(metrics):
        """Assess overall model quality based on metrics"""
        r2 = metrics.get('r2', 0)
        mape = metrics.get('mape', 100)
        
        if r2 > 0.8 and mape < 10:
            return 'Excellent'
        elif r2 > 0.6 and mape < 20:
            return 'Good'
        elif r2 > 0.4 and mape < 30:
            return 'Moderate'
        else:
            return 'Poor'
    
    @staticmethod
    def compare_model_performances(model_results):
        """
        Compare multiple model performances and generate insights
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            dict: Comparison insights and rankings
        """
        if not model_results:
            return {}
        
        # Extract key metrics for comparison
        comparison_data = []
        for model_name, results in model_results.items():
            if 'error' not in results and 'metrics' in results:
                metrics = results['metrics']
                comparison_data.append({
                    'model': model_name,
                    'r2': metrics.get('r2', 0),
                    'mape': metrics.get('mape', 100),
                    'mae': metrics.get('mae', float('inf')),
                    'stability': metrics.get('stability_score', 0),
                    'composite': results.get('composite_score', 0)
                })
        
        if not comparison_data:
            return {}
        
        # Sort by composite score
        comparison_data.sort(key=lambda x: x['composite'], reverse=True)
        
        # Generate insights
        best_model = comparison_data[0]
        worst_model = comparison_data[-1]
        
        insights = {
            'best_performer': {
                'model': best_model['model'],
                'reasons': ModelUtils._get_performance_reasons(best_model)
            },
            'worst_performer': {
                'model': worst_model['model'],
                'reasons': ModelUtils._get_performance_reasons(worst_model, is_worst=True)
            },
            'metric_leaders': {
                'highest_r2': max(comparison_data, key=lambda x: x['r2'])['model'],
                'lowest_mape': min(comparison_data, key=lambda x: x['mape'])['model'],
                'most_stable': max(comparison_data, key=lambda x: x['stability'])['model']
            },
            'performance_spread': {
                'r2_range': max(d['r2'] for d in comparison_data) - min(d['r2'] for d in comparison_data),
                'mape_range': max(d['mape'] for d in comparison_data) - min(d['mape'] for d in comparison_data)
            },
            'ranked_models': [d['model'] for d in comparison_data]
        }
        
        return insights
    
    @staticmethod
    def _get_performance_reasons(model_data, is_worst=False):
        """Get reasons for model performance"""
        reasons = []
        
        r2 = model_data['r2']
        mape = model_data['mape']
        stability = model_data['stability']
        
        if not is_worst:
            # Reasons for good performance
            if r2 > 0.8:
                reasons.append(f"High accuracy (R² = {r2:.3f})")
            if mape < 10:
                reasons.append(f"Low error rate (MAPE = {mape:.1f}%)")
            if stability > 0.8:
                reasons.append(f"Consistent performance (Stability = {stability:.3f})")
        else:
            # Reasons for poor performance
            if r2 < 0.4:
                reasons.append(f"Low accuracy (R² = {r2:.3f})")
            if mape > 30:
                reasons.append(f"High error rate (MAPE = {mape:.1f}%)")
            if stability < 0.5:
                reasons.append(f"Inconsistent performance (Stability = {stability:.3f})")
        
        return reasons if reasons else ["Standard performance"]
    
    @staticmethod
    def generate_feature_importance_summary(feature_importance_dict, top_n=5):
        """
        Generate a summary of feature importance
        
        Args:
            feature_importance_dict: Dictionary of feature importances
            top_n: Number of top features to include
            
        Returns:
            dict: Feature importance summary
        """
        if not feature_importance_dict:
            return {}
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        total_importance = sum(feature_importance_dict.values())
        cumulative_importance = 0
        
        top_features = []
        for i, (feature, importance) in enumerate(sorted_features[:top_n]):
            cumulative_importance += importance
            top_features.append({
                'rank': i + 1,
                'feature': feature,
                'importance': float(importance),
                'importance_pct': float(importance * 100) if total_importance > 0 else 0,
                'cumulative_pct': float(cumulative_importance * 100) if total_importance > 0 else 0
            })
        
        return {
            'top_features': top_features,
            'total_features': len(feature_importance_dict),
            'top_n_coverage': float(cumulative_importance * 100) if total_importance > 0 else 0,
            'feature_diversity': len([f for f in sorted_features if f[1] > 0.05])  # Features with >5% importance
        }