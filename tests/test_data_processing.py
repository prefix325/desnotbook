"""
Testes para processamento de dados
"""

import pytest
import pandas as pd
import numpy as np
from src.data.make_dataset import DataProcessor
from src.data.data_loader import DataLoader

class TestDataProcessor:
    """Testes para a classe DataProcessor"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.processor = DataProcessor()
        
        # Criar dados de teste
        self.test_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5, np.nan],
            'categorical_col': ['A', 'B', 'A', 'C', 'B', 'A'],
            'target': [0, 1, 0, 1, 1, 0]
        })
    
    def test_clean_data(self):
        """Testa limpeza de dados"""
        # Adicionar duplicata
        test_data_with_dup = pd.concat([self.test_data, self.test_data.iloc[[0]]])
        
        cleaned_data = self.processor.clean_data(test_data_with_dup)
        
        assert len(cleaned_data) == len(self.test_data)
        assert not cleaned_data.duplicated().any()
    
    def test_handle_missing_values(self):
        """Testa tratamento de valores faltantes"""
        strategy = {'numeric_col': 'mean'}
        processed_data = self.processor.handle_missing_values(self.test_data, strategy)
        
        assert not processed_data['numeric_col'].isnull().any()
        assert processed_data['numeric_col'].iloc[-1] == self.test_data['numeric_col'].mean()
    
    def test_encode_categorical(self):
        """Testa codificação de variáveis categóricas"""
        encoded_data = self.processor.encode_categorical(
            self.test_data, 
            columns=['categorical_col'], 
            method='onehot'
        )
        
        assert 'categorical_col' not in encoded_data.columns
        assert 'categorical_col_A' in encoded_data.columns
        assert 'categorical_col_B' in encoded_data.columns
        assert 'categorical_col_C' in encoded_data.columns
    
    def test_split_data(self):
        """Testa divisão dos dados"""
        # Remover valores nulos para o teste
        clean_data = self.test_data.dropna()
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.split_data(
            clean_data, 'target', test_size=0.2, validation_size=0.2
        )
        
        # Verificar se as divisões somam o total
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == len(clean_data)
        
        # Verificar se não há vazamento de dados
        assert len(set(X_train.index) & set(X_test.index)) == 0
        assert len(set(X_train.index) & set(X_val.index)) == 0
        assert len(set(X_val.index) & set(X_test.index)) == 0

class TestDataLoader:
    """Testes para a classe DataLoader"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.loader = DataLoader()
    
    def test_list_files(self):
        """Testa listagem de arquivos"""
        files = self.loader.list_files('raw')
        assert isinstance(files, list)
