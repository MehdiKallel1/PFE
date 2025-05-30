"""
Model Selector - Handles automatic selection of best performing models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import json
import os

logger = logging.getLogger(__name__)

class ModelSelector:
    """Selects the best model based on multiple criteria"""
    
    def __init__(self, model_factory, model_evaluator):
        self.model_factory = model_factory
        self.model_evaluator = model_evaluator
        self.selection_results = {}
    
    def select_best_models(self, merged_data, target_columns, test_size=0.2, cv_folds=5):
        """
        Select the best model for each target variable
        
        Args:
            merged_data: DataFrame with features and targets
            target_columns: List of column names to predict
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best model selection results for each target
        """
        logger.info(f"Starting model selection for {len(target_columns)} targets")
        
        # Prepare feature matrix (exclude target columns and non-feature columns)
        feature_columns = [col for col in merged_data.columns 
                          if col not in target_columns + ['Date', 'is_predicted']]
        X = merged_data[feature_columns]
        
        results = {}
        
        for target in target_columns:
            logger.info(f"Selecting best model for {target}")
            
            try:
                # Prepare target variable
                y = merged_data[target]
                
                # Get all available models
                models_dict = self.model_factory.get_models()
                
                # Evaluate all models
                evaluation_results = self.model_evaluator.evaluate_models(
                    models_dict, X, y, cv_folds=cv_folds, test_size=test_size
                )
                
                # Rank models and select best
                ranking_results = self.model_evaluator.rank_models()
                best_model_name = ranking_results['best_model']
                
                logger.info(f"Best model for {target}: {best_model_name}")
                
                # Train the best model on the full dataset for final predictions
                best_model_config = models_dict[best_model_name]
                final_model, final_scaler = self._train_final_model(
                    best_model_config, X, y, best_model_name
                )
                
                # Generate predictions for future data
                future_predictions = self._generate_future_predictions(
                    final_model, final_scaler, best_model_name, X.columns
                )
                
                # Compile complete results
                results[target] = {
                    'best_model': best_model_name,
                    'all_models': evaluation_results,
                    'rankings': ranking_results['rankings'],
                    'model_comparison': self.model_evaluator.get_model_comparison_data(),
                    'final_model_performance': evaluation_results[best_model_name]['metrics'],
                    'feature_importance': evaluation_results[best_model_name]['feature_importance'],
                    'future_predictions': future_predictions,
                    'selection_criteria': self._get_selection_criteria(evaluation_results[best_model_name]),
                    'confidence_assessment': self._assess_prediction_confidence(evaluation_results[best_model_name])
                }
                
                logger.info(f"Model selection completed for {target}")
                
            except Exception as e:
                logger.error(f"Error in model selection for {target}: {str(e)}")
                # Fallback to a basic result structure
                results[target] = {
                    'error': str(e),
                    'best_model': 'RandomForest',  # Fallback
                    'final_model_performance': {'r2': 0, 'mape': 100, 'mae': float('inf')},
                    'feature_importance': {},
                    'future_predictions': []
                }
        
        self.selection_results = results
        return results
    
    def _train_final_model(self, model_config, X, y, model_name):
        """Train the selected model on the full dataset"""
        logger.info(f"Training final {model_name} model on full dataset")
        
        # Prepare data
        if model_config['needs_scaling']:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_processed = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_processed = X.copy()
            scaler = None
        
        # Create and train model
        model = model_config['model']
        model.fit(X_processed, y)
        
        return model, scaler
    
    def _generate_future_predictions(self, model, scaler, model_name, feature_columns):
        """Generate predictions for future periods (2025-2026)"""
        try:
            # Load future macroeconomic data
            from app.routes import load_macro_data
            macro_df = load_macro_data()
            
            # Get future data (2025-2026)
            future_data = macro_df[macro_df['is_predicted']].copy()
            
            if future_data.empty:
                logger.warning("No future macroeconomic data available")
                return []
            
            # Prepare features (same columns as training)
            future_features = future_data[[col for col in feature_columns if col in future_data.columns]]
            
            # Handle missing columns
            for col in feature_columns:
                if col not in future_features.columns:
                    logger.warning(f"Feature {col} not available in future data, using mean")
                    future_features[col] = 0  # or use historical mean
            
            # Ensure correct column order
            future_features = future_features[feature_columns]
            
            # Apply scaling if needed
            if scaler is not None:
                future_features_scaled = scaler.transform(future_features)
                future_features_processed = pd.DataFrame(
                    future_features_scaled, 
                    columns=feature_columns, 
                    index=future_features.index
                )
            else:
                future_features_processed = future_features
            
            # Generate predictions
            predictions = model.predict(future_features_processed)
            
            # Format results
            prediction_results = []
            for i, (date, pred) in enumerate(zip(future_data['Date'], predictions)):
                prediction_results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_value': float(pred)
                })
            
            logger.info(f"Generated {len(prediction_results)} future predictions")
            return prediction_results
            
        except Exception as e:
            logger.error(f"Error generating future predictions: {str(e)}")
            return []
    
    def _get_selection_criteria(self, model_results):
        """Get the criteria used for model selection"""
        metrics = model_results['metrics']
        
        return {
            'r2_score': {
                'value': metrics['r2'],
                'weight': 0.4,
                'assessment': self._assess_r2(metrics['r2'])
            },
            'mape_score': {
                'value': metrics['mape'],
                'weight': 0.3,
                'assessment': self._assess_mape(metrics['mape'])
            },
            'mae_score': {
                'value': metrics['normalized_mae'],
                'weight': 0.2,
                'assessment': self._assess_mae(metrics['normalized_mae'])
            },
            'stability_score': {
                'value': metrics.get('stability_score', 0),
                'weight': 0.1,
                'assessment': self._assess_stability(metrics.get('stability_score', 0))
            },
            'composite_score': model_results.get('composite_score', 0)
        }
    
    def _assess_prediction_confidence(self, model_results):
        """Assess overall confidence in the model's predictions"""
        r2 = model_results['metrics']['r2']
        mape = model_results['metrics']['mape']
        stability = model_results['metrics'].get('stability_score', 0)
        
        # Confidence based on multiple factors
        if r2 > 0.8 and mape < 10 and stability > 0.8:
            confidence = 'High'
            description = 'Strong performance across all metrics'
        elif r2 > 0.6 and mape < 20 and stability > 0.6:
            confidence = 'Good'
            description = 'Reliable performance with acceptable error rates'
        elif r2 > 0.4 and mape < 30:
            confidence = 'Moderate'
            description = 'Adequate performance but with some uncertainty'
        else:
            confidence = 'Limited'
            description = 'Predictions should be interpreted with caution'
        
        return {
            'level': confidence,
            'description': description,
            'factors': {
                'accuracy': self._assess_r2(r2),
                'error_rate': self._assess_mape(mape),
                'consistency': self._assess_stability(stability)
            }
        }
    
    def _assess_r2(self, r2_score):
        """Assess R² score quality"""
        if r2_score > 0.9:
            return 'Excellent'
        elif r2_score > 0.8:
            return 'Very Good'
        elif r2_score > 0.6:
            return 'Good'
        elif r2_score > 0.4:
            return 'Moderate'
        else:
            return 'Poor'
    
    def _assess_mape(self, mape_score):
        """Assess MAPE score quality"""
        if mape_score < 5:
            return 'Excellent'
        elif mape_score < 10:
            return 'Very Good'
        elif mape_score < 20:
            return 'Good'
        elif mape_score < 30:
            return 'Moderate'
        else:
            return 'Poor'
    
    def _assess_mae(self, mae_score):
        """Assess normalized MAE score quality"""
        if mae_score < 0.05:
            return 'Excellent'
        elif mae_score < 0.1:
            return 'Very Good'
        elif mae_score < 0.2:
            return 'Good'
        elif mae_score < 0.3:
            return 'Moderate'
        else:
            return 'Poor'
    
    def _assess_stability(self, stability_score):
        """Assess model stability/consistency"""
        if stability_score > 0.9:
            return 'Very Stable'
        elif stability_score > 0.8:
            return 'Stable'
        elif stability_score > 0.6:
            return 'Moderately Stable'
        elif stability_score > 0.4:
            return 'Somewhat Unstable'
        else:
            return 'Unstable'
    
    def get_model_comparison_summary(self):
        """Get a summary of model comparisons across all targets"""
        if not self.selection_results:
            return {}
        
        summary = {
            'targets': list(self.selection_results.keys()),
            'model_frequency': {},  # How often each model was selected
            'average_performance': {},  # Average performance by model type
            'selection_insights': []
        }
        
        # Count model selections
        for target, results in self.selection_results.items():
            if 'error' not in results:
                best_model = results['best_model']
                summary['model_frequency'][best_model] = summary['model_frequency'].get(best_model, 0) + 1
        
        # Calculate average performance by model type
        model_performances = {}
        for target, results in self.selection_results.items():
            if 'error' not in results and 'all_models' in results:
                for model_name, model_result in results['all_models'].items():
                    if 'error' not in model_result:
                        if model_name not in model_performances:
                            model_performances[model_name] = {'r2': [], 'mape': [], 'composite': []}
                        
                        model_performances[model_name]['r2'].append(model_result['metrics']['r2'])
                        model_performances[model_name]['mape'].append(model_result['metrics']['mape'])
                        model_performances[model_name]['composite'].append(model_result.get('composite_score', 0))
        
        # Calculate averages
        for model_name, perfs in model_performances.items():
            summary['average_performance'][model_name] = {
                'avg_r2': np.mean(perfs['r2']),
                'avg_mape': np.mean(perfs['mape']),
                'avg_composite': np.mean(perfs['composite']),
                'consistency': 1 / (1 + np.std(perfs['r2']))  # Consistency across targets
            }
        
        return summary
    
    def save_results(self, file_path):
        """Save model selection results to JSON file"""
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            # Deep convert the results
            serializable_results = json.loads(
                json.dumps(self.selection_results, default=convert_numpy)
            )
            
            # Save to file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Model selection results saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model selection results: {str(e)}")
            return False
    
    def load_results(self, file_path):
        """Load model selection results from JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.selection_results = json.load(f)
                logger.info(f"Model selection results loaded from {file_path}")
                return True
            else:
                logger.warning(f"Model selection results file not found: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model selection results: {str(e)}")
            return False