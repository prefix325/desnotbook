"""
Classe para processamento e transformação de dados
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from src.config import config
from src.utils.logger import setup_logger
from src.data.data_loader import DataLoader

logger = setup_logger(__name__)

class DataProcessor:
    """Classe para processamento de dados"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.random_state = config.get('model.random_state', 42)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpeza básica dos dados
        
        Args:
            df: DataFrame para limpar
        
        Returns:
            DataFrame limpo
        """
        logger.info("Iniciando limpeza dos dados")
        
        # Remover duplicatas
        initial_shape = df.shape
        df = df.drop_duplicates()
        logger.info(f"Duplicatas removidas: {initial_shape[0] - df.shape[0]}")
        
        # Log de valores nulos
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.info(f"Valores nulos encontrados:\n{null_counts[null_counts > 0]}")
        
        return df
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Trata valores faltantes
        
        Args:
            df: DataFrame para processar
            strategy: Dicionário com estratégias por coluna
                     {'coluna': 'mean'/'median'/'mode'/'drop'/'fill_value'}
        
        Returns:
            DataFrame processado
        """
        if strategy is None:
            strategy = {}
        
        df_processed = df.copy()
        
        for column in df_processed.columns:
            if df_processed[column].isnull().sum() > 0:
                col_strategy = strategy.get(column, 'drop')
                
                if col_strategy == 'mean' and df_processed[column].dtype in ['int64', 'float64']:
                    df_processed[column].fillna(df_processed[column].mean(), inplace=True)
                elif col_strategy == 'median' and df_processed[column].dtype in ['int64', 'float64']:
                    df_processed[column].fillna(df_processed[column].median(), inplace=True)
                elif col_strategy == 'mode':
                    df_processed[column].fillna(df_processed[column].mode()[0], inplace=True)
                elif col_strategy == 'drop':
                    df_processed.dropna(subset=[column], inplace=True)
                elif isinstance(col_strategy, (str, int, float)):
                    df_processed[column].fillna(col_strategy, inplace=True)
        
        logger.info(f"Valores faltantes tratados. Shape final: {df_processed.shape}")
        return df_processed
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None,
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Codifica variáveis categóricas
        
        Args:
            df: DataFrame para processar
            columns: Lista de colunas para codificar
            method: Método de codificação ('onehot', 'label')
        
        Returns:
            DataFrame com variáveis codificadas
        """
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        df_encoded = df.copy()
        
        for column in columns:
            if column in df_encoded.columns:
                if method == 'onehot':
                    dummies = pd.get_dummies(df_encoded[column], prefix=column)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(column, axis=1, inplace=True)
                elif method == 'label':
                    df_encoded[column] = pd.Categorical(df_encoded[column]).codes
        
        logger.info(f"Variáveis categóricas codificadas: {columns}")
        return df_encoded
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        test_size: float = None,
        validation_size: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Divide dados em treino, validação e teste
        
        Args:
            df: DataFrame completo
            target_column: Nome da coluna target
            test_size: Proporção para teste
            validation_size: Proporção para validação
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if test_size is None:
            test_size = config.get('model.test_size', 0.2)
        if validation_size is None:
            validation_size = config.get('model.validation_size', 0.2)
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Primeira divisão: treino+validação vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Segunda divisão: treino vs validação
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"Dados divididos - Treino: {X_train.shape[0]}, "
                   f"Validação: {X_val.shape[0]}, Teste: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera resumo dos dados
        
        Args:
            df: DataFrame para analisar
        
        Returns:
            Dicionário com resumo dos dados
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {
                col: df[col].value_counts().head().to_dict() 
                for col in df.select_dtypes(include=['object']).columns
            }
        }
        
        return summary
