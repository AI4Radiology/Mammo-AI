import os
import argparse
import pandas as pd
import yaml
import json
from datetime import datetime
from pathlib import Path
import optuna
import importlib
from .base_trainer import BaseTrainer
import numpy as _np
import importlib

try:
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        # Forzar weights_only=False si no se especifica
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass


class PyTorchTabularTrainer(BaseTrainer):
    def __init__(self, config_path: str):
        # Cargar config y pasar a la clase base
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)['train_settings']
        super().__init__(config)

    def get_model(self, continuous_cols, categorical_cols=None):
        """Override para PyTorch Tabular que necesita columnas específicas."""
        categorical_cols = categorical_cols or []

        # Cargar módulo y crear modelo
        model_cfg = self.config['model']
        model_type = 'pytorch_tabular'  # Tipo fijo para este trainer
        module_path = model_cfg['module_map'].get(model_type, 'models.pytorch_tabular.pytorch_tabular_model')
        
        # Importar módulo dinámicamente
        module = importlib.import_module(module_path)
        return module.create_model(continuous_cols=continuous_cols, categorical_cols=categorical_cols)

    def _apply_smote_to_dataframe(self, train_df, target_col):
        """
        Aplicar SMOTE a un DataFrame y retornar DataFrame (específico de PyTorch).
        Wrapper sobre el método base que trabaja con X, y separados.
        """

        # Verifica si SMOTE está habilitado
        if not self.config['preprocessing']['use_smote']:
            return train_df
        
        # Separá X e y
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        # Aplica smote usando método base
        X_res, y_res = self._apply_smote_if_enabled(X_train, y_train)
        
        # Reconstruye el DataFrame
        train_df = pd.DataFrame(X_res, columns=X_train.columns)
        train_df[target_col] = y_res
        return train_df
    
    def _prepare_dataframes(self, train_df, val_df, test_df, target_col):
        """Preparar DataFrames: columnas continuas y casting a float32."""

        # Selecciona únicamente columnas numéricas (excluyendo el target)
        numeric_cols = train_df.select_dtypes(include=[_np.number]).columns.tolist()
        continuous_cols = [c for c in numeric_cols if c != target_col]

        # Elimina las columnas que en el train sean completamente NaN
        all_nan = [c for c in continuous_cols if train_df[c].isna().all()]
        if all_nan:
            print(f"Columnas eliminadas (all NaN en train): {all_nan}")
            for c in all_nan:
                continuous_cols.remove(c)
                train_df.drop(columns=[c], inplace=True, errors='ignore')
                val_df.drop(columns=[c], inplace=True, errors='ignore')
                test_df.drop(columns=[c], inplace=True, errors='ignore')

        # Elimina las columnas constantes (poca o ninguna variación) en train
        const_cols = [c for c in continuous_cols if train_df[c].nunique(dropna=False) <= 1]
        if const_cols:
            print(f"Columnas eliminadas (constantes en train): {const_cols}")
            for c in const_cols:
                continuous_cols.remove(c)
                train_df.drop(columns=[c], inplace=True, errors='ignore')
                val_df.drop(columns=[c], inplace=True, errors='ignore')
                test_df.drop(columns=[c], inplace=True, errors='ignore')

        # Casting a float32 para todas las columnas continuas
        for _df in (train_df, val_df, test_df):
            # Valida que la columna exista en el DataFrame
            existing = [c for c in continuous_cols if c in _df.columns]
            if existing:
                _df[existing] = _df[existing].astype('float32')

        return continuous_cols

    # -------------------- TRAIN & EVAL --------------------
    def train_and_evaluate(self, df: pd.DataFrame, target_col: str):
        """Entrenar y evaluar modelo usando PyTorch Tabular."""

        # Prepara los datos separándolos en train, val, test
        train_df, val_df, test_df = self._split_data_with_val(df, target_col)
        train_df = self._apply_smote_to_dataframe(train_df, target_col)
        continuous_cols = self._prepare_dataframes(train_df, val_df, test_df, target_col)
        categorical_cols = []

        # Modelo
        model = self.get_model(continuous_cols, categorical_cols)
        model.fit(train=train_df, validation=val_df)

        # Evaluar en test
        prediction_col = f"{target_col}_prediction"
        pred_test = model.predict(test_df)
        y_test = test_df[target_col].values
        y_pred = pred_test[prediction_col].values

        # También evalua en train para detectar overfitting
        pred_train = model.predict(train_df)
        y_train = train_df[target_col].values
        y_train_pred = pred_train[prediction_col].values

        # Calcula las métricas usando método de la clase base
        results = self._calculate_metrics(y_train, y_train_pred, y_test, y_pred)

        # Evalua con método específico de PyTorch Tabular
        test_results = model.evaluate(test_df)

        # Agrega los datos específicos de PyTorch Tabular
        results.update({
            'model': model,
            'n_samples': len(df),
            'n_features': len(continuous_cols) + len(categorical_cols),
            'feature_names': continuous_cols + categorical_cols,
            'test_results': test_results,
        })

        return results

    # -------------------- OPTUNA --------------------
    def optimize_hyperparameters(self, df, target_col, n_trials=50):
        """Optimiza hiperparámetros usando Optuna con espacio de búsqueda declarativo en YAML."""
        print(f"Optuna: {n_trials} trials")

        # Prepara los datos (mismo flujo que train_and_evaluate)
        train_df, val_df, test_df = self._split_data_with_val(df, target_col)
        train_df = self._apply_smote_to_dataframe(train_df, target_col)
        continuous_cols = self._prepare_dataframes(train_df, val_df, test_df, target_col)

        # Cargar módulo de Optuna centralizado
        optuna_module = importlib.import_module('models.pytorch_tabular.pytorch_tabular_optuna')

        def objective(trial):
            """Función objetivo para Optuna"""
            try:
                # Crear modelo con hiperparámetros sugeridos (YAML + trial)
                model = optuna_module.create_model_for_optuna(
                    trial=trial,
                    continuous_cols=continuous_cols
                )

                # Entrena el modelo
                model.fit(train=train_df, validation=val_df)

                # Evaluar en validación
                result = model.evaluate(val_df)

                # Obtiene la métrica de validación (valid_loss o test_loss)
                if isinstance(result, list) and len(result) > 0:
                    result_dict = result[0]
                    val_loss = result_dict.get('valid_loss', result_dict.get('test_loss', result_dict.get('test_loss_0', float('inf'))))
                else:
                    val_loss = float('inf')
                return float(val_loss)
            except Exception:
                return float('inf')

        # Crear estudio de Optuna (sin persistencia)
        study = optuna.create_study(direction="minimize", study_name="pytorch_tabular_optimization")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Muestra resultados
        print("Best params:", study.best_params)
        print(f"Best valid_loss: {study.best_value:.4f}")

        # Guarda los mejores parámetros
        results_dir = Path(os.path.dirname(os.path.dirname(__file__))) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        best_params_path = results_dir / 'optuna_best_params.json'
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': n_trials,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"Saved best params to: {best_params_path}")

        return study.best_params, test_df
    
    
    def _save_model_specific(self, results, model_dir, base_name):
        """Guardar modelo PyTorch Tabular."""
        model_path = model_dir / "pytorch_model"
        
        results['model'].save_model(str(model_path))
        print(f"Modelo guardado en: {model_path}")
    
    def _add_extra_results_to_json(self, results, json_results):
        """Agregar métricas específicas de PyTorch Tabular al JSON."""

        json_results['pytorch_tabular_metrics'] = results['test_results']
    
    def _print_extra_results(self, results):
        """Imprime las métricas específicas de PyTorch Tabular."""

        if 'test_results' in results:
            print(f"\nPyTorch Tabular metrics:")
            test_results = results['test_results']
            if isinstance(test_results, list) and len(test_results) > 0:
                for key, value in test_results[0].items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")



def main():
    parser = argparse.ArgumentParser(description='Entrenamiento con PyTorch Tabular')
    parser.add_argument('--config', default="configs/train_config.yaml")
    parser.add_argument('--data', required=True, help='Ruta al archivo CSV con los datos')
    parser.add_argument('--features', required=False, help='Archivo con lista de features (opcional)')
    parser.add_argument('--optimize', action='store_true', help='Usar Optuna para optimizar hiperparámetros')
    parser.add_argument('--n-trials', type=int, default=50, help='Número de trials para Optuna (default: 50)')
    parser.add_argument('--save-model', action='store_true', help='Si se pasa, guarda el modelo y resultados')
    args = parser.parse_args()
    
    trainer = PyTorchTabularTrainer(args.config)
    
    df = trainer.load_data(args.data)
    
    # Cargar features solo si se especifica el archivo
    features = None
    if args.features:
        features = trainer.load_features(args.features)
    
    # Prepara los datos
    df, target_col = trainer.prepare_data(df, features)
    
    print(f"Datos: {df.shape[0]} muestras, {df.shape[1] - 1} features")
    
    # Wrapper para save_model con prefijo correcto
    def save_pytorch_model(results):
        return trainer.save_model(results, model_prefix='modelo_pt')
    
    if args.optimize:
        # Modo de optimización con Optuna
        best_params, test_df = trainer.optimize_hyperparameters(df, target_col, args.n_trials)
        
    else:
        # Modo de entrenamiento normal
        results = trainer.train_and_evaluate(df, target_col)
        trainer.print_results(results)
        if args.save_model:
            save_pytorch_model(results)


if __name__ == "__main__":
    main()
