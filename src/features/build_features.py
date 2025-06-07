"""
Classe para engenharia de features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from src.utils.logger import setup_logger
from src.utils.helpers import save_pickle, load_pickle

logger = setup_logger(__name__)

class FeatureEngineer:
    """Classe para engenharia de features"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        
    def create_polynomial_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Cria features polinomiais
        
        Args:
            df: DataFrame original
            columns: Colunas para criar features polinomiais
            degree: Grau do polinômio
        
        Returns:
            DataFrame com features polinomiais
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        df_poly = df.copy()
        
        for column in columns:
            if column in df.columns:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(df[[column]])
                
                # Criar nomes para as novas features
                feature_names = [f"{column}_poly_{i}" for i in range(1, poly_features.shape[1])]
                
                # Adicionar features polinomiais (exceto a original)
                for i, name in enumerate(feature_names[1:], 1):
                    df_poly[name] = poly_features[:, i]
        
        logger.info(f"Features polinomiais criadas para: {columns}")
        return df_poly
    
    def create_interaction_features(
        self, 
        df: pd.DataFrame, 
        column_pairs: List[tuple]
    ) -> pd.DataFrame:
        """
        Cria features de interação entre variáveis
        
        Args:
            df: DataFrame original
            column_pairs: Lista de tuplas com pares de colunas
        
        Returns:
            DataFrame com features de interação
        """
        df_interaction = df.copy()
        
        for col1, col2 in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplicação
                df_interaction[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                
                # Divisão (evitando divisão por zero)
                df_interaction[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)
                
                # Soma
                df_interaction[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
                
                # Diferença
                df_interaction[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
        
        logger.info(f"Features de interação criadas para: {column_pairs}")
        return df_interaction
    
    def create_aggregation_features(
        self, 
        df: pd.DataFrame, 
        group_columns: List[str],
        agg_columns: List[str],
        agg_functions: List[str] = ['mean', 'std', 'min', 'max', 'count']
    ) -> pd.DataFrame:
        """
        Cria features de agregação
        
        Args:
            df: DataFrame original
            group_columns: Colunas para agrupamento
            agg_columns: Colunas para agregação
            agg_functions: Funções de agregação
        
        Returns:
            DataFrame com features de agregação
        """
        df_agg = df.copy()
        
        for group_col in group_columns:
            for agg_col in agg_columns:
                for func in agg_functions:
                    if group_col in df.columns and agg_col in df.columns:
                        agg_values = df.groupby(group_col)[agg_col].transform(func)
                        df_agg[f"{agg_col}_{func}_by_{group_col}"] = agg_values
        
        logger.info(f"Features de agregação criadas")
        return df_agg
    
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        method: str = 'standard',
        columns: List[str] = None
    ) -> tuple:
        """
        Escala features numéricas
        
        Args:
            X_train: Dados de treino
            X_val: Dados de validação (opcional)
            X_test: Dados de teste (opcional)
            method: Método de escalonamento ('standard', 'minmax', 'robust')
            columns: Colunas para escalar (se None, escala todas numéricas)
        
        Returns:
            Tupla com dados escalonados
        """
        if columns is None:
            columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Escolher scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Método não suportado: {method}")
        
        # Fit no treino
        X_train_scaled = X_train.copy()
        X_train_scaled[columns] = scaler.fit_transform(X_train[columns])
        
        # Salvar scaler
        self.scalers[method] = scaler
        
        results = [X_train_scaled]
        
        # Transform validação e teste
        if X_val is not None:
            X_val_scaled = X_val.copy()
            X_val_scaled[columns] = scaler.transform(X_val[columns])
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[columns] = scaler.transform(X_test[columns])
            results.append(X_test_scaled)
        
        logger.info(f"Features escalonadas usando {method}: {len(columns)} colunas")
        
        return tuple(results) if len(results) > 1 else results[0]
    
    def select_features(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        k: int = 10,
        task_type: str = 'classification'
    ) -> tuple:
        """
        Seleciona as k melhores features
        
        Args:
            X_train: Dados de treino
            y_train: Target de treino
            X_val: Dados de validação (opcional)
            X_test: Dados de teste (opcional)
            k: Número de features para selecionar
            task_type: Tipo de tarefa ('classification' ou 'regression')
        
        Returns:
            Tupla com dados com features selecionadas
        """
        # Escolher função de score
        score_func = f_classif if task_type == 'classification' else f_regression
        
        # Criar seletor
        selector = SelectKBest(score_func=score_func, k=k)
        
        # Fit e transform no treino
        X_train_selected = pd.DataFrame(
            selector.fit_transform(X_train, y_train),
            columns=X_train.columns[selector.get_support()],
            index=X_train.index
        )
        
        # Salvar seletor
        self.feature_selectors[f'top_{k}'] = selector
        
        results = [X_train_selected]
        
        # Transform validação e teste
        if X_val is not None:
            X_val_selected = pd.DataFrame(
                selector.transform(X_val),
                columns=X_train.columns[selector.get_support()],
                index=X_val.index
            )
            results.append(X_val_selected)
        
        if X_test is not None:
            X_test_selected = pd.DataFrame(
                selector.transform(X_test),
                columns=X_train.columns[selector.get_support()],
                index=X_test.index
            )
            results.append(X_test_selected)
        
        selected_features = X_train.columns[selector.get_support()].tolist()
        logger.info(f"Features selecionadas ({k}): {selected_features}")
        
        return tuple(results) if len(results) > 1 else results[0]
    
    def save_transformers(self, filepath: str) -> None:
        """
        Salva transformadores (scalers, selectors)
        
        Args:
            filepath: Caminho para salvar
        """
        transformers = {
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors
        }
        save_pickle(transformers, filepath)
        logger.info(f"Transformadores salvos em: {filepath}")
    
    def load_transformers(self, filepath: str) -> None:
        """
        Carrega transformadores salvos
        
        Args:
            filepath: Caminho do arquivo
        """
        transformers = load_pickle(filepath)
        self.scalers = transformers.get('scalers', {})
        self.feature_selectors = transformers.get('feature_selectors', {})
        logger.info(f"Transformadores carregados de: {filepath}")
