import argparse
import joblib
import numpy as np
import pandas as pd
import yaml
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .base_trainer import BaseTrainer


class GenericMLTrainer(BaseTrainer):
    def __init__(self, config_path):
        # Cargar config y pasar a la clase base
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)['train_settings']
        super().__init__(config)
        
        # Configurar scaler específico de GenericML
        scaler_type = self.config['preprocessing']['scaler_type']
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    
    def prepare_data(self, df, features=None):
        """Override para manejar X,y separados (específico de GenericML)."""
        # Usar la lógica de la clase base para mapeo y filtrado correcto
        df_prepared, target_col = super().prepare_data(df, features)
        
        # Separa X (features) e y (variable objetivo)
        y = df_prepared[target_col]

        y = y.astype(int)
        X = df_prepared.drop(columns=[target_col])
        
        print(f"Usando {len(X.columns)} features")
        
        return X, y

    def train_and_evaluate(self, X, y):
        # Dividir datos usando método de la clase base
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # SMOTE usando método de la clase base
        X_train_scaled, y_train = self._apply_smote_if_enabled(X_train_scaled, y_train)
        
        # Entrena el modelo (usa get_model de la clase base)
        model = self.get_model()
        # Si es GridSearch, intentamos mostrar progreso con tqdm compatible con joblib
        model = self._fit_with_progress_if_gridsearch(model, X_train_scaled, y_train)
        
        # Predicciones
        predictor = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        y_pred = predictor.predict(X_test_scaled)
        y_train_pred = predictor.predict(X_train_scaled)
        
        # Cross-validation con StratifiedKFold
        cv_folds = self.config['cross_validation']['cv_folds']
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config['data_split']['random_state'])
        cv_scores = [model.best_score_] if hasattr(model, 'best_score_') else cross_val_score(model, X_train_scaled, y_train, cv=cv)
        
        # Calcula las métricas usando método de la clase base
        results = self._calculate_metrics(y_train, y_train_pred, y_test, y_pred)
        
        # Agregar datos específicos de GenericML
        results.update({
            'model': model,
            'scaler': self.scaler,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_names': list(X.columns),
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        })
        
        return results

    def _fit_with_progress_if_gridsearch(self, model, X, y):
        """Encapsula model.fit agregando una barra de progreso tqdm para GridSearchCV en paralelo.

        - Detecta si el modelo tiene atributos típicos de GridSearch (param_grid y cv).
        - Usa tqdm_joblib para integrar una barra de progreso con joblib (n_jobs=-1 soportado).
        - Si no está disponible tqdm_joblib, continúa sin barra de progreso.
        """
        # Valida si es GridSearch
        is_grid = hasattr(model, 'param_grid') and hasattr(model, 'cv')
        if not is_grid:
            model.fit(X, y)
            return model

        # Calcula el total de tareas: (#candidatos) * (#splits)
        try:
            n_candidates = len(list(ParameterGrid(model.param_grid)))
        except Exception:
            n_candidates = None

        # Calcula el número de splits en CV
        try:
            cv = model.cv
            if hasattr(cv, 'get_n_splits'):
                n_splits = cv.get_n_splits(X, y)
            else:
                n_splits = int(cv)
        except Exception:
            n_splits = None

        # Calcula el total de tareas: (#candidatos) * (#splits)
        total_tasks = n_candidates * n_splits if (n_candidates is not None and n_splits is not None) else None

        # Intenta usar tqdm_joblib
        try:
            from tqdm.auto import tqdm
            from tqdm_joblib import tqdm_joblib

            # Barra de progreso para GridSearch
            desc = f"{getattr(getattr(model, 'estimator', model), '__class__', type(model)).__name__} GridSearch"
            if total_tasks is None:
                # Sin total conocido, al menos mostramos barra indeterminada
                with tqdm_joblib(tqdm(desc=desc, unit='fit')):
                    model.fit(X, y)
            else:
                with tqdm_joblib(tqdm(total=total_tasks, desc=desc, unit='fit')):
                    model.fit(X, y)
        except Exception as e:
            print(f"No se pudo activar tqdm para GridSearch (motivo: {e}). Entrenando sin barra de progreso...")
            model.fit(X, y)

        return model
    
    # ==================== HOOKS PARA BaseTrainer ====================
    
    def _save_model_specific(self, results, model_dir, base_name):
        """Guardar modelo PKL con scaler para GenericML."""
        # Prepara el modelo (obtener best_estimator_ si es GridSearch)
        model = results['model'].best_estimator_ if hasattr(results['model'], 'best_estimator_') else results['model']
        
        # Prepara los datos del modelo
        model_data = {
            'model': model,
            'scaler': results['scaler'],
            'config': self.config,
            'feature_names': results.get('feature_names')
        }
        
        # Agrega los mejores parámetros si es GridSearch
        if hasattr(results['model'], 'best_params_'):
            model_data['best_params'] = results['model'].best_params_
            model_data['best_cv_score'] = results['model'].best_score_
        
        # Guarda el modelo PKL
        pkl_path = model_dir / f"{base_name}.pkl"
        joblib.dump(model_data, pkl_path)
        print(f"Modelo guardado en: {pkl_path}")
    
    def _add_extra_results_to_json(self, results, json_results):
        """Agregar métricas CV y GridSearch al JSON."""

        # Agregar métricas CV
        json_results['metrics']['cv_mean'] = float(results['cv_mean'])
        json_results['metrics']['cv_std'] = float(results['cv_std'])
        
        # Agregar mejores parámetros si es GridSearch
        if hasattr(results['model'], 'best_params_'):
            json_results['best_params'] = results['model'].best_params_
            json_results['best_cv_score'] = float(results['model'].best_score_)
    
    def _print_extra_results(self, results):
        """Imprimir métricas CV y GridSearch."""
        # Imprime métricas CV
        print(f"CV: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        # Imprime mejores parámetros si es GridSearch
        if hasattr(results['model'], 'best_params_'):
            print(f"Best params: {results['model'].best_params_}")
            print(f"Best CV score: {results['model'].best_score_:.4f}")


def main():

    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train_config.yaml ")
    parser.add_argument('--data', required=True)
    parser.add_argument('--features', required=False, help='Archivo con lista de features')
    parser.add_argument('--save-model', action='store_true', help='Si se incluye, guarda el modelo entrenado y resultados')
    args = parser.parse_args()
    
    # Crear entrenador
    trainer = GenericMLTrainer(args.config)
    
    # Cargar los datos
    df = trainer.load_data(args.data)
    
    # Cargar las features
    features = None
    if args.features:
        features = trainer.load_features(args.features)
    
    # Preparar los datos
    X, y = trainer.prepare_data(df, features)
    
    # Información de datos
    print(f"Datos: {X.shape[0]} muestras, {X.shape[1]} features")
    
    # Entrenar y evaluar
    results = trainer.train_and_evaluate(X, y)
    # Imprime los resultados
    trainer.print_results(results)

    # Guardar modelo
    if args.save_model:
        trainer.save_model(results, model_prefix='modelo')


if __name__ == "__main__":
    main()