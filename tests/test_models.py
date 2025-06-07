"""
Testes para modelos
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.models.train_model import ModelTrainer
from src.models.predict_model import ModelPredictor

class TestModelTrainer:
    """Testes para a classe ModelTrainer"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.trainer = ModelTrainer()
        
        # Criar dados de classificação
        X_class, y_class = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.X_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(10)])
        self.y_class = pd.Series(y_class)
        
        # Criar dados de regressão
        X_reg, y_reg = make_regression(
            n_samples=100, n_features=10, random_state=42
        )
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(10)])
        self.y_reg = pd.Series(y_reg)
    
    def test_get_default_models_classification(self):
        """Testa obtenção de modelos padrão para classificação"""
        models = self.trainer.get_default_models('classification')
        
        assert 'random_forest' in models
        assert 'logistic_regression' in models
        assert 'svm' in models
    
    def test_get_default_models_regression(self):
        """Testa obtenção de modelos padrão para regressão"""
        models = self.trainer.get_default_models('regression')
        
        assert 'random_forest' in models
        assert 'linear_regression' in models
        assert 'svm' in models
    
    def test_train_single_model_classification(self):
        """Testa treinamento de modelo único para classificação"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = self.trainer.train_single_model(
            model, self.X_class, self.y_class, model_name='test_rf'
        )
        
        assert 'model' in results
        assert 'train_metrics' in results
        assert 'accuracy' in results['train_metrics']
        assert results['train_metrics']['accuracy'] > 0
    
    def test_train_single_model_regression(self):
        """Testa treinamento de modelo único para regressão"""
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        results = self.trainer.train_single_model(
            model, self.X_reg, self.y_reg, model_name='test_rf_reg'
        )
        
        assert 'model' in results
        assert 'train_metrics' in results
        assert 'mse' in results['train_metrics']
        assert 'r2' in results['train_metrics']
    
    def test_cross_validate_model(self):
        """Testa validação cruzada"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = self.trainer.cross_validate_model(
            model, self.X_class, self.y_class, cv=3
        )
        
        assert 'scores' in results
        assert 'mean_score' in results
        assert 'std_score' in results
        assert len(results['scores']) == 3

class TestModelPredictor:
    """Testes para a classe ModelPredictor"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.predictor = ModelPredictor()
        
        # Criar e treinar um modelo simples
        from sklearn.ensemble import RandomForestClassifier
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        self.X_test = pd.DataFrame(X[:10], columns=[f'feature_{i}' for i in range(5)])
    
    def test_predict_without_model(self):
        """Testa predição sem modelo carregado"""
        with pytest.raises(ValueError):
            self.predictor.predict(self.X_test)
    
    def test_predict_with_model(self):
        """Testa predição com modelo"""
        # Simular carregamento do modelo
        self.predictor.model = self.model
        self.predictor.model_name = 'test_model'
        
        predictions = self.predictor.predict(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Testa predição de probabilidades"""
        # Simular carregamento do modelo
        self.predictor.model = self.model
        self.predictor.model_name = 'test_model'
        
        probabilities = self.predictor.predict_proba(self.X_test)
        
        assert probabilities.shape == (len(self.X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_model_summary(self):
        """Testa resumo do modelo"""
        # Simular carregamento do modelo
        self.predictor.model = self.model
        self.predictor.model_name = 'test_model'
        
        summary = self.predictor.model_summary()
        
        assert 'model_name' in summary
        assert 'model_type' in summary
        assert summary['model_name'] == 'test_model'
        assert summary['model_type'] == 'RandomForestClassifier'
