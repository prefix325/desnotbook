"""
Funções auxiliares gerais
"""

import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd

def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Salva objeto usando pickle
    
    Args:
        obj: Objeto a ser salvo
        filepath: Caminho do arquivo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Carrega objeto usando pickle
    
    Args:
        filepath: Caminho do arquivo
    
    Returns:
        Objeto carregado
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_json(obj: Dict, filepath: Union[str, Path]) -> None:
    """
    Salva dicionário como JSON
    
    Args:
        obj: Dicionário a ser salvo
        filepath: Caminho do arquivo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Carrega JSON como dicionário
    
    Args:
        filepath: Caminho do arquivo
    
    Returns:
        Dicionário carregado
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_directories(paths: List[Union[str, Path]]) -> None:
    """
    Cria diretórios se não existirem
    
    Args:
        paths: Lista de caminhos para criar
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def get_project_root() -> Path:
    """
    Retorna o diretório raiz do projeto
    
    Returns:
        Path do diretório raiz
    """
    return Path(__file__).parent.parent.parent

def memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula uso de memória de um DataFrame
    
    Args:
        df: DataFrame para analisar
    
    Returns:
        DataFrame com informações de memória
    """
    memory_usage = df.memory_usage(deep=True)
    memory_usage_mb = memory_usage / 1024**2
    
    return pd.DataFrame({
        'Column': memory_usage.index,
        'Memory_Usage_MB': memory_usage_mb.values,
        'Data_Type': [str(df[col].dtype) if col != 'Index' else 'Index' 
                     for col in memory_usage.index]
    }).sort_values('Memory_Usage_MB', ascending=False)
