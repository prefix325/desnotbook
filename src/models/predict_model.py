"""
Classe para predições com modelos treinados
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any, Optional
import joblib
from pathlib import Path

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelPredictor:
    """Classe para fazer predições com modelos treinados"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.models_path = config.models_path
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Carrega modelo salvo
        
        Args:
            model_path: Caminho do modelo
        """
        try:
            self.model = joblib.load(model_path)
            self.model_name = Path(model_path).stem
            logger.info(f"Modelo carregado: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições
        
        Args:
            X: Features para predição
        
        Returns:
            Array com predições
        """
        if self.model is None:
            raise ValueError("Nenhum modelo carregado. Use load_model() primeiro.")
        
        try:
            predictions = self.model.predict(X)
            logger.info(f"Predições realizadas para {len(X)} amostras")
            return predictions
        except Exception as e:
            logger.error(f"Erro ao fazer predições: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições de probabilidade (apenas classificação)
        
        Args:
            X: Features para predição
        
        Returns:
            Array com probabilidades
        """
        if self.model is None:
            raise ValueError("Nenhum modelo carregado. Use load_model() primeiro.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Modelo não suporta predição de probabilidade")
        
        try:
            probabilities = self.model.predict_proba(X)
            logger.info(f"Probabilidades calculadas para {len(X)} amostras")
            return probabilities
        except Exception as e:
            logger.error(f"Erro ao calcular probabilidades: {str(e)}")
            raise
    
    def predict_with_confidence(
        self, 
        X: pd.DataFrame,
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Faz predições com análise de confiança
        
        Args:
            X: Features para predição
            confidence_threshold: Limiar de confiança
        
        Returns:
            Dicionário com predições e análise de confiança
        """
        predictions = self.predict(X)
        
        result = {
            'predictions': predictions,
            'total_samples': len(X)
        }
        
        # Se modelo suporta probabilidades
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.predict_proba(X)
            max_probabilities = np.max(probabilities, axis=1)
            
            result.update({
                'probabilities': probabilities,
                'max_probabilities': max_probabilities,
                'high_confidence_mask': max_probabilities >= confidence_threshold,
                'high_confidence_count': np.sum(max_probabilities >= confidence_threshold),
                'low_confidence_count': np.sum(max_probabilities < confidence_threshold),
                'mean_confidence': np.mean(max_probabilities)
            })
            
            logger.info(f"Predições com alta confiança: {result['high_confidence_count']}/{len(X)}")
        
        return result
    
    def batch_predict(
        self, 
        data_path: Union[str, Path],
        output_path: Union[str, Path] = None,
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """
        Faz predições em lote para arquivo grande
        
        Args:
            data_path: Caminho dos dados
            output_path: Caminho para salvar resultados
            batch_size: Tamanho do lote
        
        Returns:
            DataFrame com predições
        """
        if self.model is None:
            raise ValueError("Nenhum modelo carregado. Use load_model() primeiro.")
        
        logger.info(f"Iniciando predições em lote: {data_path}")
        
        # Carregar dados
        if str(data_path).endswith('.csv'):
            data = pd.read_csv(data_path)
        elif str(data_path).endswith('.xlsx'):
            data = pd.read_excel(data_path)
        else:
            raise ValueError("Formato de arquivo não suportado")
        
        # Predições em lote
        all_predictions = []
        
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            batch_predictions = self.predict(batch)
            all_predictions.extend(batch_predictions)
            
            logger.info(f"Processado lote {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
        
        # Criar DataFrame resultado
        result_df = data.copy()
        result_df['predictions'] = all_predictions
        
        # Adicionar probabilidades se disponível
        if hasattr(self.model, 'predict_proba'):
            probabilities = []
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                batch_proba = self.predict_proba(batch)
                probabilities.extend(batch_proba)
            
            # Adicionar colunas de probabilidade
            proba_array = np.array(probabilities)
            for i in range(proba_array.shape[1]):
                result_df[f'proba_class_{i}'] = proba_array[:, i]
        
        # Salvar se especificado
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Resultados salvos em: {output_path}")
        
        logger.info(f"Predições em lote concluídas: {len(data)} amostras")
        return result_df
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Obtém importância das features (se disponível)
        
        Returns:
            DataFrame com importância das features
        """
        if self.model is None:
            raise ValueError("Nenhum modelo carregado. Use load_model() primeiro.")
        
        if hasattr(self.model, 'feature_importances_'):
            # Para modelos tree-based
            importances = self.model.feature_importances_
            
            # Se temos nomes das features (assumindo que foram salvos)
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            else:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        elif hasattr(self.model, 'coef_'):
            # Para modelos lineares
            coefficients = self.model.coef_
            
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            else:
                feature_names = [f'feature_{i}' for i in range(len(coefficients))]
            
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients.flatten() if coefficients.ndim > 1 else coefficients
            })
            coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
            coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
            
            return coef_df
        
        else:
            logger.warning("Modelo não suporta análise de importância de features")
            return None
    
    def model_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo do modelo carregado
        
        Returns:
            Dicionário com informações do modelo
        """
        if self.model is None:
            return {"error": "Nenhum modelo carregado"}
        
        summary = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'model_params': self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }
        
        # Informações específicas do modelo
        if hasattr(self.model, 'n_features_in_'):
            summary['n_features'] = self.model.n_features_in_
        
        if hasattr(self.model, 'classes_'):
            summary['classes'] = self.model.classes_.tolist()
            summary['n_classes'] = len(self.model.classes_)
        
        return summary
