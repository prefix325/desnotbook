"""
Módulo de configuração do projeto
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Classe para gerenciar configurações do projeto"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configurações do arquivo YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Arquivo de configuração {self.config_path} não encontrado.")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração usando notação de ponto"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def project_root(self) -> Path:
        """Retorna o diretório raiz do projeto"""
        return Path(__file__).parent.parent
    
    @property
    def data_paths(self) -> Dict[str, Path]:
        """Retorna caminhos dos diretórios de dados"""
        root = self.project_root
        return {
            'raw': root / self.get('data.raw_path', 'data/raw'),
            'interim': root / self.get('data.interim_path', 'data/interim'),
            'processed': root / self.get('data.processed_path', 'data/processed'),
            'external': root / self.get('data.external_path', 'data/external'),
        }
    
    @property
    def models_path(self) -> Path:
        """Retorna caminho do diretório de modelos"""
        return self.project_root / self.get('model.models_path', 'models')

# Instância global de configuração
config = Config()
