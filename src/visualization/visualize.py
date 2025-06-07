"""
Classe para visualizações de dados e resultados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataVisualizer:
    """Classe para criar visualizações"""
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (10, 6)):
        self.style = style
        self.figsize = figsize
        self.figures_path = Path(config.get('visualization.figures_path', 'reports/figures'))
        self.figures_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar estilo
        plt.style.use(self.style)
        sns.set_palette("husl")
    
    def plot_distribution(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None,
        save_path: str = None
    ) -> None:
        """
        Plota distribuição de variáveis numéricas
        
        Args:
            df: DataFrame com dados
            columns: Colunas para plotar (se None, usa todas numéricas)
            save_path: Caminho para salvar figura
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(columns):
            if i < len(axes):
                sns.histplot(data=df, x=column, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribuição de {column}')
        
        # Remover subplots vazios
        for i in range(len(columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.figures_path / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None,
        save_path: str = None
    ) -> None:
        """
        Plota matriz de correlação
        
        Args:
            df: DataFrame com dados
            columns: Colunas para incluir (se None, usa todas numéricas)
            save_path: Caminho para salvar figura
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        correlation_matrix = df[columns].corr()
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('Matriz de Correlação')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.figures_path / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_target_distribution(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        save_path: str = None
    ) -> None:
        """
        Plota distribuição da variável target
        
        Args:
            df: DataFrame com dados
            target_column: Nome da coluna target
            save_path: Caminho para salvar figura
        """
        plt.figure(figsize=self.figsize)
        
        if df[target_column].dtype in ['object', 'category']:
            # Variável categórica
            value_counts = df[target_column].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Distribuição de {target_column}')
            plt.xlabel(target_column)
            plt.ylabel('Frequência')
            
            # Adicionar percentuais
            total = len(df)
            for i, v in enumerate(value_counts.values):
                plt.text(i, v + 0.01*max(value_counts.values), 
                        f'{v}\n({100*v/total:.1f}%)', 
                        ha='center', va='bottom')
        else:
            # Variável numérica
            sns.histplot(data=df, x=target_column, kde=True)
            plt.title(f'Distribuição de {target_column}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.figures_path / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_feature_vs_target(
        self, 
        df: pd.DataFrame, 
        feature_columns: List[str],
        target_column: str,
        save_path: str = None
    ) -> None:
        """
        Plota relação entre features e target
        
        Args:
            df: DataFrame com dados
            feature_columns: Lista de features para plotar
            target_column: Nome da coluna target
            save_path: Caminho para salvar figura
        """
        n_cols = min(3, len(feature_columns))
        n_rows = (len(feature_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(feature_columns):
            if i < len(axes):
                if df[target_column].dtype in ['object', 'category']:
                    # Target categórico
                    if df[feature].dtype in ['object', 'category']:
                        # Feature categórica vs Target categórico
                        pd.crosstab(df[feature], df[target_column]).plot(kind='bar', ax=axes[i])
                    else:
                        # Feature numérica vs Target categórico
                        sns.boxplot(data=df, x=target_column, y=feature, ax=axes[i])
                else:
                    # Target numérico
                    if df[feature].dtype in ['object', 'category']:
                        # Feature categórica vs Target numérico
                        sns.boxplot(data=df, x=feature, y=target_column, ax=axes[i])
                    else:
                        # Feature numérica vs Target numérico
                        sns.scatterplot(data=df, x=feature, y=target_column, ax=axes[i])
                
                axes[i].set_title(f'{feature} vs {target_column}')
        
        # Remover subplots vazios
        for i in range(len(feature_columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.figures_path / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_model_performance(
        self, 
        results: Dict[str, Dict[str, Any]],
        metric: str = 'accuracy',
        save_path: str = None
    ) -> None:
        """
        Plota performance de múltiplos modelos
        
        Args:
            results: Resultados dos modelos
            metric: Métrica para comparar
            save_path: Caminho para salvar figura
        """
        models = []
        train_scores = []
        val_scores = []
        
        for model_name, result in results.items():
            models.append(model_name)
            
            train_metric = result.get('train_metrics', {}).get(metric, 0)
            val_metric = result.get('val_metrics', {}).get(metric, 0)
            
            train_scores.append(train_metric)
            val_scores.append(val_metric)
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.figure(figsize=self.figsize)
        plt.bar(x - width/2, train_scores, width, label='Treino', alpha=0.8)
        plt.bar(x + width/2, val_scores, width, label='Validação', alpha=0.8)
        
        plt.xlabel('Modelos')
        plt.ylabel(metric.capitalize())
        plt.title(f'Comparação de Modelos - {metric.capitalize()}')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (train, val) in enumerate(zip(train_scores, val_scores)):
            plt.text(i - width/2, train + 0.01, f'{train:.3f}', 
                    ha='center', va='bottom', fontsize=9)
            plt.text(i + width/2, val + 0.01, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.figures_path / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        class_names: List[str] = None,
        save_path: str = None
    ) -> None:
        """
        Plota matriz de confusão
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            class_names: Nomes das classes
            save_path: Caminho para salvar figura
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names or range(len(cm)),
            yticklabels=class_names or range(len(cm))
        )
        plt.title('Matriz de Confusão')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.figures_path / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self, 
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: str = None
    ) -> None:
        """
        Plota importância das features
        
        Args:
            importance_df: DataFrame com importância das features
            top_n: Número de features mais importantes para mostrar
            save_path: Caminho para salvar figura
        """
        # Pegar top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, max(6, len(top_features) * 0.3)))
        
        # Determinar coluna de importância
        importance_col = 'importance' if 'importance' in top_features.columns else 'abs_coefficient'
        
        sns.barplot(
            data=top_features, 
            y='feature', 
            x=importance_col,
            orient='h'
        )
        
        plt.title(f'Top {len(top_features)} Features Mais Importantes')
        plt.xlabel('Importância')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.figures_path / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_learning_curve(
        self, 
        train_scores: List[float],
        val_scores: List[float],
        epochs: List[int] = None,
        save_path: str = None
    ) -> None:
        """
        Plota curva de aprendizado
        
        Args:
            train_scores: Scores de treino por época
            val_scores: Scores de validação por época
            epochs: Lista de épocas (se None, usa índices)
            save_path: Caminho para salvar figura
        """
        if epochs is None:
            epochs = list(range(1, len(train_scores) + 1))
        
        plt.figure(figsize=self.figsize)
        plt.plot(epochs, train_scores, 'o-', label='Treino', alpha=0.8)
        plt.plot(epochs, val_scores, 'o-', label='Validação', alpha=0.8)
        
        plt.xlabel('Época')
        plt.ylabel('Score')
        plt.title('Curva de Aprendizado')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.figures_path / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura salva: {save_path}")
        
        plt.show()
    
    def plot_residuals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: str = None
    ) -> None:
        """
        Plota gráfico de resíduos (para regressão)
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            save_path: Caminho para salvar figura
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Resíduos vs Predições
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predições')
        ax1.set_ylabel('Resíduos')
        ax1.set_title('Resíduos vs Predições')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot dos resíduos
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot dos Resíduos')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.figures_path / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura salva: {save_path}")
        
        plt.show()
