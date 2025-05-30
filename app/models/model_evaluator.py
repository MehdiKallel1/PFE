"""
Model Evaluator - Handles performance evaluation and comparison of ML models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates and compares multiple ML models"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.scalers = {}  # Store scalers for each model that needs them
    
    def evaluate_models(self, models_dict, X, y, cv_folds=5, test_size=0.2):
        """
        Evaluate all models using cross-validation and test set
        
        Args:
            models_dict: Dictionary of models from ModelFactory
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        logger.info(f"Evaluating {len(models_dict)} models")
        
        # Split data for final testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Use TimeSeriesSplit for cross-validation (appropriate for time series data)
        cv = TimeSeriesSplit(n_splits=cv_folds)
        
        results = {}
        
        for model_name, model_config in models_dict.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Prepare data (scaling if needed)
                X_train_processed, X_test_processed = self._prepare_data(
                    X_train, X_test, model_name, model_config['needs_scaling']
                )
                
                # Get model instance
                model = model_config['model']
                
                # Cross-validation evaluation
                cv_scores = self._cross_validate_model(
                    model, X_train_processed, y_train, cv
                )
                
                # Train on full training set and evaluate on test set
                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_test_processed)
                
                # Calculate all metrics
                metrics = self._calculate_metrics(y_test, y_pred)
                
                # Add cross-validation metrics
                metrics.update({
                    'cv_r2_mean': np.mean(cv_scores['r2']),
                    'cv_r2_std': np.std(cv_scores['r2']),
                    'cv_mae_mean': np.mean(cv_scores['mae']),
                    'cv_mae_std': np.std(cv_scores['mae']),
                    'stability_score': 1 / (1 + np.std(cv_scores['r2']))  # Higher is better
                })
                
                # Extract feature importance
                feature_importance = self._extract_feature_importance(
                    model, X.columns, model_name
                )
                
                # Store complete results
                results[model_name] = {
                    'metrics': metrics,
                    'feature_importance': feature_importance,
                    'test_predictions': y_pred.tolist(),
                    'test_actual': y_test.tolist(),
                    'test_dates': X_test.index.strftime('%Y-%m-%d').tolist() if hasattr(X_test.index, 'strftime') else list(range(len(y_test))),
                    'model_info': {
                        'description': model_config['description'],
                        'complexity': model_config['complexity'],
                        'needs_scaling': model_config['needs_scaling']
                    }
                }
                
                logger.info(f"{model_name} - R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                # Store error information but continue with other models
                results[model_name] = {
                    'error': str(e),
                    'metrics': {'r2': 0, 'mape': 100, 'mae': float('inf')},
                    'feature_importance': {},
                    'test_predictions': [],
                    'test_actual': [],
                    'test_dates': []
                }
        
        self.evaluation_results = results
        return results
    
    def _prepare_data(self, X_train, X_test, model_name, needs_scaling):
        """Prepare data for a specific model (scaling if needed)"""
        if needs_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler for later use
            self.scalers[model_name] = scaler
            
            # Convert back to DataFrame to preserve column names
            X_train_processed = pd.DataFrame(
                X_train_scaled, 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_test_processed = pd.DataFrame(
                X_test_scaled, 
                columns=X_test.columns, 
                index=X_test.index
            )
        else:
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()
            
        return X_train_processed, X_test_processed
    
    def _cross_validate_model(self, model, X, y, cv):
        """Perform cross-validation for a model"""
        cv_results = {'r2': [], 'mae': []}
        
        for train_idx, val_idx in cv.split(X):
            X_train_cv = X.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_train_cv = y.iloc[train_idx]
            y_val_cv = y.iloc[val_idx]
            
            # Train model
            model.fit(X_train_cv, y_train_cv)
            
            # Predict
            y_pred_cv = model.predict(X_val_cv)
            
            # Calculate metrics
            cv_results['r2'].append(r2_score(y_val_cv, y_pred_cv))
            cv_results['mae'].append(mean_absolute_error(y_val_cv, y_pred_cv))
        
        return cv_results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate all performance metrics"""
        # Basic metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error) - handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, np.nan)) * 100)
            if np.isnan(mape) or np.isinf(mape):
                mape = 0
        
        # Normalized MAE (0-1 scale)
        y_range = np.max(y_true) - np.min(y_true)
        normalized_mae = mae / y_range if y_range > 0 else 0
        
        return {
            'r2': float(r2),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'normalized_mae': float(normalized_mae)
        }
    
    def _extract_feature_importance(self, model, feature_names, model_name):
        """Extract feature importance from trained model"""
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (RandomForest, XGBoost, LightGBM)
                importances = model.feature_importances_
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = float(importances[i])
                    
            elif hasattr(model, 'coef_'):
                # Linear models (Ridge, Lasso, LinearRegression)
                # Use absolute coefficients as importance
                coefs = np.abs(model.coef_)
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = float(coefs[i])
                
                # Normalize to sum to 1
                total = sum(importance_dict.values())
                if total > 0:
                    importance_dict = {k: v/total for k, v in importance_dict.items()}
                    
            elif model_name == 'SVR':
                # For SVR, we can't easily extract feature importance
                # Use permutation importance or set equal weights
                equal_weight = 1.0 / len(feature_names)
                for feature in feature_names:
                    importance_dict[feature] = equal_weight
                    
            elif model_name == 'Neural':
                # For neural networks, feature importance is complex
                # Use equal weights as approximation
                equal_weight = 1.0 / len(feature_names)
                for feature in feature_names:
                    importance_dict[feature] = equal_weight
            
            # Sort by importance
            importance_dict = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance for {model_name}: {e}")
            # Fall back to equal weights
            equal_weight = 1.0 / len(feature_names)
            for feature in feature_names:
                importance_dict[feature] = equal_weight
        
        return importance_dict
    
    def calculate_composite_score(self, metrics):
        """
        Calculate composite score for model ranking
        Formula: (0.4 × R²) + (0.3 × (1-MAPE/100)) + (0.2 × (1-MAE_norm)) + (0.1 × Stability)
        """
        r2_component = max(0, metrics.get('r2', 0)) * 0.4
        mape_component = max(0, (1 - metrics.get('mape', 100) / 100)) * 0.3
        mae_component = max(0, (1 - min(1, metrics.get('normalized_mae', 1)))) * 0.2
        stability_component = metrics.get('stability_score', 0) * 0.1
        
        composite_score = r2_component + mape_component + mae_component + stability_component
        return float(composite_score)
    
    def rank_models(self):
        """Rank models by composite score and return ordered results"""
        if not self.evaluation_results:
            return {}
        
        # Calculate composite scores
        model_scores = {}
        for model_name, results in self.evaluation_results.items():
            if 'error' not in results:
                composite_score = self.calculate_composite_score(results['metrics'])
                model_scores[model_name] = composite_score
                results['composite_score'] = composite_score
            else:
                model_scores[model_name] = 0
                results['composite_score'] = 0
        
        # Sort by composite score
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'rankings': ranked_models,
            'best_model': ranked_models[0][0] if ranked_models else None,
            'detailed_results': self.evaluation_results
        }
    
    def get_model_comparison_data(self):
        """Get data formatted for model comparison visualizations"""
        if not self.evaluation_results:
            return {}
        
        comparison_data = {
            'model_names': [],
            'r2_scores': [],
            'mape_scores': [],
            'mae_scores': [],
            'composite_scores': [],
            'complexity_levels': [],
            'descriptions': []
        }
        
        for model_name, results in self.evaluation_results.items():
            if 'error' not in results:
                comparison_data['model_names'].append(model_name)
                comparison_data['r2_scores'].append(results['metrics']['r2'])
                comparison_data['mape_scores'].append(results['metrics']['mape'])
                comparison_data['mae_scores'].append(results['metrics']['normalized_mae'])
                comparison_data['composite_scores'].append(results.get('composite_score', 0))
                comparison_data['complexity_levels'].append(results['model_info']['complexity'])
                comparison_data['descriptions'].append(results['model_info']['description'])
        
        return comparison_data