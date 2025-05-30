"""
Model Factory - Defines and configures all ML models for the financial prediction system
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

class ModelFactory:
    """Factory class for creating and configuring ML models"""
    
    @staticmethod
    def get_models():
        """
        Returns a dictionary of model instances with their configurations
        """
        models = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'needs_scaling': False,
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'description': 'Ensemble of decision trees with voting',
                'complexity': 'Medium'
            },
            
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42, verbosity=0),
                'needs_scaling': False,
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                },
                'description': 'Gradient boosting with advanced regularization',
                'complexity': 'High'
            },
            
            'LightGBM': {
                'model': lgb.LGBMRegressor(random_state=42, verbosity=-1),
                'needs_scaling': False,
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                },
                'description': 'Fast gradient boosting algorithm',
                'complexity': 'High'
            },
            
            'Ridge': {
                'model': Ridge(random_state=42),
                'needs_scaling': True,
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                'description': 'Linear regression with L2 regularization',
                'complexity': 'Low'
            },
            
            'Lasso': {
                'model': Lasso(random_state=42, max_iter=2000),
                'needs_scaling': True,
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                'description': 'Linear regression with L1 regularization',
                'complexity': 'Low'
            },
            
            'LinearRegression': {
                'model': LinearRegression(),
                'needs_scaling': True,
                'param_grid': {},
                'description': 'Simple linear regression baseline',
                'complexity': 'Low'
            },
            
            'SVR': {
                'model': SVR(),
                'needs_scaling': True,
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear', 'poly']
                },
                'description': 'Support Vector Regression',
                'complexity': 'Medium'
            },
            
            'Neural': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'needs_scaling': True,
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                'description': 'Multi-layer Perceptron Neural Network',
                'complexity': 'High'
            }
        }
        
        return models
    
    @staticmethod
    def get_scaler():
        """Returns a StandardScaler instance for models that need scaling"""
        return StandardScaler()
    
    @staticmethod
    def get_model_info(model_name):
        """Get information about a specific model"""
        models = ModelFactory.get_models()
        if model_name in models:
            return {
                'name': model_name,
                'description': models[model_name]['description'],
                'complexity': models[model_name]['complexity'],
                'needs_scaling': models[model_name]['needs_scaling']
            }
        return None
    
    @staticmethod
    def create_model_with_params(model_name, params=None):
        """Create a model instance with specific parameters"""
        models = ModelFactory.get_models()
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")
        
        model_config = models[model_name]
        
        if params:
            # Create model with custom parameters
            if model_name == 'RandomForest':
                return RandomForestRegressor(random_state=42, **params)
            elif model_name == 'XGBoost':
                return xgb.XGBRegressor(random_state=42, verbosity=0, **params)
            elif model_name == 'LightGBM':
                return lgb.LGBMRegressor(random_state=42, verbosity=-1, **params)
            elif model_name == 'Ridge':
                return Ridge(random_state=42, **params)
            elif model_name == 'Lasso':
                return Lasso(random_state=42, max_iter=2000, **params)
            elif model_name == 'LinearRegression':
                return LinearRegression(**params)
            elif model_name == 'SVR':
                return SVR(**params)
            elif model_name == 'Neural':
                return MLPRegressor(random_state=42, max_iter=1000, **params)
        else:
            # Return default model
            return model_config['model']