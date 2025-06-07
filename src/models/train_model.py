"""
Classe para treinamento de modelos
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

from src.config import config
from src.utils.logger import setup_logger
from src.utils.helpers import save_pickle

logger = setup_logger(__name__)

class ModelTrainer:
    """Classe para treinamento de modelos de ML"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.random_state = config.get('model.random_state', 42)
        
    def get_default_models(self, task_type: str = 'classification') -> Dict[str, Any]:
        """
        Retorna modelos padrão para a tarefa
        
        Args:
            task_type: Tipo de tarefa ('classification' ou 'regression')
        
        Returns:
            Dicionário com modelos
        """
        if task_type == 'classification':
            return {
                'random_forest': RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.random_state
                ),
                'logistic_regression': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                ),
                'svm': SVC(
                    random_state=self.random_state,
                    probability=True
                )
            }
        else:  # regression
            return {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state
                ),
                'linear_regression': LinearRegression(),
                'svm': SVR()
            }
    
    def train_single_model(
        self, 
        model: Any, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        model_name: str = 'model'
    ) -> Dict[str, Any]:
        """
        Treina um único modelo
        
        Args:
            model: Modelo para treinar
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            model_name: Nome do modelo
        
        Returns:
            Dicionário com resultados
        """
        logger.info(f"Treinando modelo: {model_name}")
        
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Predições
        y_train_pred = model.predict(X_train)
        results = {
            'model': model,
            'model_name': model_name,
            'train_predictions': y_train_pred
        }
        
        # Métricas de treino
        if self._is_classification_task(y_train):
            results['train_metrics'] = self._calculate_classification_metrics(
                y_train, y_train_pred, model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
            )
        else:
            results['train_metrics'] = self._calculate_regression_metrics(y_train, y_train_pred)
        
        # Validação se fornecida
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            results['val_predictions'] = y_val_pred
            
            if self._is_classification_task(y_train):
                results['val_metrics'] = self._calculate_classification_metrics(
                    y_val, y_val_pred, model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                )
            else:
                results['val_metrics'] = self._calculate_regression_metrics(y_val, y_val_pred)
        
        # Salvar modelo
        self.models[model_name] = model
        self.results[model_name] = results
        
        logger.info(f"Modelo {model_name} treinado com sucesso")
        return results
    
    def train_multiple_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        models: Dict[str, Any] = None,
        task_type: str = 'classification'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Treina múltiplos modelos
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            models: Dicionário com modelos (se None, usa padrões)
            task_type: Tipo de tarefa
        
        Returns:
            Dicionário com resultados de todos os modelos
        """
        if models is None:
            models = self.get_default_models(task_type)
        
        all_results = {}
        
        for model_name, model in models.items():
            try:
                results = self.train_single_model(
                    model, X_train, y_train, X_val, y_val, model_name
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Erro ao treinar {model_name}: {str(e)}")
                continue
        
        return all_results
    
    def cross_validate_model(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Realiza validação cruzada
        
        Args:
            model: Modelo para validar
            X: Features
            y: Target
            cv: Número de folds
            scoring: Métrica para avaliação
        
        Returns:
            Resultados da validação cruzada
        """
        logger.info(f"Executando validação cruzada com {cv} folds")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scoring_metric': scoring
        }
        
        logger.info(f"CV Score: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        return results
    
    def hyperparameter_tuning(
        self, 
        model: Any, 
        param_grid: Dict[str, List],
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Otimização de hiperparâmetros
        
        Args:
            model: Modelo base
            param_grid: Grade de parâmetros
            X_train: Features de treino
            y_train: Target de treino
            cv: Número de folds
            scoring: Métrica para otimização
        
        Returns:
            Resultados da otimização
        """
        logger.info("Iniciando otimização de hiperparâmetros")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Melhores parâmetros: {results['best_params']}")
        logger.info(f"Melhor score: {results['best_score']:.4f}")
        
        return results
    
    def _is_classification_task(self, y: pd.Series) -> bool:
        """Verifica se é tarefa de classificação"""
        return len(y.unique()) <= 20 and y.dtype in ['object', 'category', 'bool'] or y.dtype == 'int64'
    
    def _calculate_classification_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """Calcula métricas de classificação"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # AUC apenas para classificação binária
        if len(np.unique(y_true)) == 2 and y_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    def _calculate_regression_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas de regressão"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Salva modelo treinado
        
        Args:
            model_name: Nome do modelo
            filepath: Caminho para salvar
        """
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            logger.info(f"Modelo {model_name} salvo em: {filepath}")
        else:
            logger.error(f"Modelo {model_name} não encontrado")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Compara resultados de todos os modelos treinados
        
        Returns:
            DataFrame com comparação dos modelos
        """
        if not self.results:
            logger.warning("Nenhum modelo treinado ainda")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {'model': model_name}
            
            # Métricas de treino
            if 'train_metrics' in results:
                for metric, value in results['train_metrics'].items():
                    row[f'train_{metric}'] = value
            
            # Métricas de validação
            if 'val_metrics' in results:
                for metric, value in results['val_metrics'].items():
                    row[f'val_{metric}'] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
