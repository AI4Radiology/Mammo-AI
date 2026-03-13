"""
Clase base abstracta para trainers de ML.
Contiene lógica común para carga de datos, preparación y guardado de modelos.
"""
from abc import ABC, abstractmethod
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import os
import importlib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

class BaseTrainer(ABC):
    """Clase base abstracta para todos los trainers."""
    
    def __init__(self, config):
        """
        Args:
            config: Configuración ya cargada desde YAML (dict)
        """
        self.config = config
    
    # ==================== MÉTODOS COMUNES ====================
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Cargar datos desde CSV con manejo de decimales con coma."""
        # Cargar CSV con pandas
        df = pd.read_csv(csv_path, sep=',', on_bad_lines='skip')
        target = self.config['preprocessing']['target_variable']
        
        # Convierte las columnas (excepto target) de objeto a numérico
        for col in df.columns:
            if col != target and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        return df
    
    def load_features(self, features_path: str):
        """Cargar lista de features desde archivo de texto."""
        with open(features_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def prepare_data(self, df: pd.DataFrame, features=None):
        """
        Preparar datos: filtrar features y aplicar mapeo de clases.
        
        Args:
            df: DataFrame con los datos
            features: Lista opcional de features a usar
        
        Returns:
            tuple: (DataFrame preparado, nombre de columna target)
        """
        # Obtiene la configuración relevante desde YAML como la variable target y tipo de clasificación
        target_col = self.config['preprocessing']['target_variable']
        class_type = self.config['preprocessing']['classification_type']
        
        # Filtra las features si se proporcionan
        if features is not None:
            keep_cols = [c for c in features if c in df.columns]
            if target_col not in keep_cols:
                keep_cols.append(target_col)
            df = df[keep_cols]
        
        # Aplica el mapeo de clases desde YAML
        mapping = self.config['preprocessing']['class_mappings'][class_type]
        
        # Filtrar solo las filas cuyo target está en el mapeo
        valid_mask = df[target_col].isin(mapping.keys())
        df = df[valid_mask].copy()  # Usar .copy() para evitar SettingWithCopyWarning
        
        # Aplica el mapeo usando .loc para evitar warnings
        df.loc[:, target_col] = df[target_col].replace(mapping).astype('int64')
        
        print(f"Mapeo '{class_type}' aplicado: {mapping}")
        print(f"Filas después del filtrado: {len(df)}")
        
        return df, target_col
    
    def save_model(self, results, model_prefix='modelo'):
        """
        Guardar modelo y resultados en carpeta estructurada.
        
        Args:
            results: Diccionario con resultados del entrenamiento
            model_prefix: Prefijo para el nombre del modelo (ej: 'modelo', 'modelo_pt')
        
        Returns:
            Path: Ruta al directorio del modelo guardado
        """
        # Directorio base: proyecto_raíz/outputs/results
        project_root = Path(__file__).parent.parent.parent  # src/training/ → src/ → proyecto_raíz/
        base_dir = project_root / 'outputs' / 'saved_models'
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre basado en el tipo de clasificación
        classification_type = self.config['preprocessing']['classification_type']
        base_name = f"{model_prefix}_{classification_type}"
        
        # Crear timestamp para el nombre de la carpeta
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Crear carpeta con el nombre del modelo + timestamp
        model_dir = base_dir / f"{base_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar el modelo usando el método específico de cada trainer
        self._save_model_specific(results, model_dir, base_name)
        
        # Crear JSON con resultados comunes
        json_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'classification_type': classification_type,
            'model_type': self.config['model']['type'],
            'data_info': {
                'samples': results.get('n_samples', 'N/A'),
                'features': results.get('n_features', 'N/A')
            },
            'metrics': {
                'train_accuracy': float(results['train_accuracy']),
                'test_accuracy': float(results['test_accuracy']),
                'accuracy': float(results['accuracy']),
                'overfit_difference': float(results['overfit_difference']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score'])
            },
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'config_used': self.config
        }
        
        # Agrega las métricas adicionales específicas del trainer
        self._add_extra_results_to_json(results, json_results)
        
        # Guarda el JSON
        json_path = model_dir / f"{base_name}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResultados guardados en: {json_path}")
        
        return model_dir
    
    def print_results(self, results):
        """Imprime los resultados del entrenamiento de forma legible."""
        
        classification_type = self.config['preprocessing']['classification_type']
        model_type = self.config['model']['type']
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS - {model_type.upper()} ({classification_type})")
        print(f"{'='*60}")
        print(f"Train Acc: {results['train_accuracy']:.4f}")
        print(f"Test  Acc: {results['test_accuracy']:.4f}")
        print(f"Overfit Δ: {results['overfit_difference']:.4f}")
        print(f"{'-'*60}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        
        # Imprimir métricas adicionales si existen
        self._print_extra_results(results)
        
        print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
        print(f"\nClassification Report:\n{results['classification_report']}")
        print(f"{'='*60}")
    
    @abstractmethod
    def train_and_evaluate(self, *args, **kwargs):
        """Entrena y evalúa el modelo. Debe retornar dict con resultados."""
        pass

    def _add_extra_results_to_json(self, results, json_results):
        """
        Hook para que trainers específicos agreguen métricas adicionales al JSON.
        Override en clases hijas si necesitas agregar métricas específicas.
        
        Args:
            results: Diccionario con resultados del entrenamiento
            json_results: Diccionario que se guardará como JSON (modificar in-place)
        """
        pass  # Implementación vacía por defecto
    
    def _print_extra_results(self, results):
        """
        Hook para que trainers específicos impriman métricas adicionales.
        Override en clases hijas si necesitas imprimir métricas específicas.
        
        Args:
            results: Diccionario con resultados del entrenamiento
        """
        pass  # Implementación vacía por defecto
    
    @abstractmethod
    def _save_model_specific(self, results, model_dir, base_name):
        """
        Hook ABSTRACTO para guardar el modelo de forma específica.
        DEBE ser implementado por cada trainer hijo.
        
        Args:
            results: Diccionario con resultados del entrenamiento
            model_dir: Path al directorio donde guardar el modelo
            base_name: Nombre base para archivos del modelo
        """
        pass
    
    def get_model(self, *args, **kwargs):
        """
        Crear modelo usando importación dinámica desde module_map en config.
        Override si necesitas lógica más compleja.
        """
        model_config = self.config['model']
        model_type = model_config['type']
        module_path = model_config['module_map'][model_type]
        
        # Importar el módulo dinámicamente
        module = importlib.import_module(module_path)
        return module.create_model(*args, **kwargs)
    
    def _split_data(self, X, y):
        """
        Split básico train/test con configuración desde YAML.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        
        # Obtiene las configuración de split desde YAML
        data_config = self.config['data_split']

        # Realizar split y retornar los datos
        return train_test_split(
            X, y,
            test_size=data_config['test_size'],
            random_state=data_config['random_state'],
            stratify=y if data_config['stratify'] else None
        )
    
    def _split_data_with_val(self, df, target_col):
        """
        Split en train/val/test para modelos que necesitan validación.
        
        Args:
            df: DataFrame completo
            target_col: Nombre de la columna target
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        
        # Obtener configuración de split desde YAML
        data_cfg = self.config['data_split']
        
        # Primero separar test
        train_val_df, test_df = train_test_split(
            df,
            test_size=data_cfg['test_size'],
            random_state=data_cfg['random_state'],
            stratify=df[target_col] if data_cfg['stratify'] else None,
        )
        
        # Luego separar train y validation
        val_size = data_cfg.get('val_size', 0.2)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=data_cfg['random_state'],
            stratify=train_val_df[target_col] if data_cfg['stratify'] else None,
        )
        
        return train_df, val_df, test_df
    
    def _apply_smote_if_enabled(self, X_train, y_train):
        """
        Aplicar SMOTE si está habilitado en la configuración.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            
        Returns:
            tuple: (X_train_resampled, y_train_resampled)
        """
        # verificar si SMOTE está habilitado
        if not self.config['preprocessing']['use_smote']:
            return X_train, y_train
        
        # Aplicar SMOTE y retornar datos re-muestreados
        smote = SMOTE(**self.config['preprocessing']['smote_params'])
        return smote.fit_resample(X_train, y_train)
    
    def _calculate_metrics(self, y_train, y_train_pred, y_test, y_pred):
        """
        Calcular métricas comunes de clasificación.
        
        Args:
            y_train: Target real de entrenamiento
            y_train_pred: Predicciones en entrenamiento
            y_test: Target real de test
            y_pred: Predicciones en test
            
        Returns:
            dict: Diccionario con todas las métricas
        """
        
        # Obtener configuración de métricas
        metrics_config = self.config['metrics']
        
        # Calcular métricas básicas
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)
        overfit_diff = train_accuracy - test_accuracy
        
        # Retornar todas las métricas en un diccionario
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'accuracy': test_accuracy,
            'overfit_difference': overfit_diff,
            'precision': precision_score(y_test, y_pred, average=metrics_config['precision_average']),
            'recall': recall_score(y_test, y_pred, average=metrics_config['recall_average']),
            'f1_score': f1_score(y_test, y_pred, average=metrics_config['f1_average']),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }