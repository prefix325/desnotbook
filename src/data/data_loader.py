"""
Classe para carregamento de dados
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Union
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    """Classe para carregar dados de diferentes fontes"""
    
    def __init__(self):
        self.data_paths = config.data_paths
        
    def load_csv(
        self, 
        filename: str, 
        data_type: str = 'raw',
        **kwargs
    ) -> pd.DataFrame:
        """
        Carrega arquivo CSV
        
        Args:
            filename: Nome do arquivo
            data_type: Tipo de dados (raw, interim, processed, external)
            **kwargs: Argumentos adicionais para pd.read_csv
        
        Returns:
            DataFrame carregado
        """
        filepath = self.data_paths[data_type] / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        logger.info(f"Carregando {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        return df
    
    def load_excel(
        self, 
        filename: str, 
        data_type: str = 'raw',
        **kwargs
    ) -> pd.DataFrame:
        """
        Carrega arquivo Excel
        
        Args:
            filename: Nome do arquivo
            data_type: Tipo de dados (raw, interim, processed, external)
            **kwargs: Argumentos adicionais para pd.read_excel
        
        Returns:
            DataFrame carregado
        """
        filepath = self.data_paths[data_type] / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        logger.info(f"Carregando {filepath}")
        df = pd.read_excel(filepath, **kwargs)
        logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        return df
    
    def save_csv(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        data_type: str = 'processed',
        **kwargs
    ) -> None:
        """
        Salva DataFrame como CSV
        
        Args:
            df: DataFrame para salvar
            filename: Nome do arquivo
            data_type: Tipo de dados (raw, interim, processed, external)
            **kwargs: Argumentos adicionais para df.to_csv
        """
        filepath = self.data_paths[data_type] / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Salvando {filepath}")
        df.to_csv(filepath, index=False, **kwargs)
        logger.info(f"Arquivo salvo: {filepath}")
    
    def list_files(self, data_type: str = 'raw') -> list:
        """
        Lista arquivos em um diretório de dados
        
        Args:
            data_type: Tipo de dados (raw, interim, processed, external)
        
        Returns:
            Lista de arquivos
        """
        path = self.data_paths[data_type]
        if not path.exists():
            return []
        
        return [f.name for f in path.iterdir() if f.is_file()]
