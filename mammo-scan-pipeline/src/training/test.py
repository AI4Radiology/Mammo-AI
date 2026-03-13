"""
Búsqueda exhaustiva de hiperparámetros con multiprocessing.
Reutiliza BaseTrainer para preparación de datos.
"""
# Importaciones estándar
import gc
import sys
import yaml
import numpy as np
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# Importaciones de sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
# Importaciones de modelos y métricas
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Agregar el directorio raíz al path para importaciones absolutas
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ahora importar sin punto (importación absoluta)
from training.base_trainer import BaseTrainer


# ==================== VARIABLES GLOBALES (requeridas por multiprocessing) ====================
_SHARED = {}


def _init_worker(X_train, X_test, y_train, y_test, config):
    """Inicializa datos compartidos en cada worker."""

    # Declara las variables globales para no tener que cargar datos en cada proceso
    global _SHARED
    _SHARED['X_train'] = X_train
    _SHARED['X_test'] = X_test
    _SHARED['y_train'] = y_train
    _SHARED['y_test'] = y_test
    _SHARED['config'] = config


def _create_model(model_type: str, params: dict, random_state: int = 42):
    """Factory para crear modelos según tipo y parámetros."""
    
    # Verifica y creo el modelo según el tipo de modelo definido en los parámetros
    if model_type == 'svm':
        return SVC(
            C=params['C'],
            gamma=params['gamma'],
            tol=params['tol'],
            kernel='rbf',
            random_state=random_state
        )
    
    elif model_type == 'xgboost':
        n_classes = len(np.unique(_SHARED['y_train']))
        return XGBClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            min_child_weight=params['min_child_weight'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            objective='binary:logistic' if n_classes == 2 else 'multi:softprob',
            eval_metric='logloss' if n_classes == 2 else 'mlogloss',
            random_state=random_state,
            verbosity=0,
            n_jobs=1
        )
    
    elif model_type == 'adaboost':
        base = DecisionTreeClassifier(
            max_depth=params['estimator__max_depth'],
            criterion=params['estimator__criterion'],
            min_samples_split=params['estimator__min_samples_split'],
            min_samples_leaf=params['estimator__min_samples_leaf'],
            max_features=params.get('estimator__max_features'),
            min_impurity_decrease=params.get('estimator__min_impurity_decrease', 0.0),
            random_state=random_state
        )
        return AdaBoostClassifier(
            estimator=base,
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            random_state=random_state
        )
    
    elif model_type == 'logistic_lasso':
        return LogisticRegression(
            penalty='l1',
            C=params['C'],
            tol=params['tol'],
            solver='liblinear',
            max_iter=10000,
            random_state=random_state
        )
    
    raise ValueError(f"Modelo no soportado: {model_type}")


def _train_combination(args):
    """
    Entrena UNA combinación de hiperparámetros.
    Función genérica para todos los modelos.
    """
    model_type, params = args
    
    # Entrenar y evaluar el modelo con los datos globales
    try:
        X_train = _SHARED['X_train']
        X_test = _SHARED['X_test']
        y_train = _SHARED['y_train']
        y_test = _SHARED['y_test']
        config = _SHARED['config']
        
        # Crear y entrenar modelo
        model = _create_model(model_type, params)
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # Métricas
        metrics = config['metrics']
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        
        result = {
            **params,  # incluye todos los hiperparámetros
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'overfit_diff': train_acc - test_acc,
            'precision': precision_score(y_test, y_pred, average=metrics['precision_average'], zero_division=0),
            'recall': recall_score(y_test, y_pred, average=metrics['recall_average'], zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=metrics['f1_average'], zero_division=0)
        }
        
        del model
        gc.collect()
        return result
        
    except Exception as e:
        return {**params, 'test_accuracy': 0.0, 'error': str(e)}


class HyperparameterSearcher:
    """
    Buscador de hiperparámetros usando multiprocessing.
    Reutiliza BaseTrainer para preparación de datos.
    """
    
    def __init__(self, config_path: str, data_path: str):

        # cargar las rutas
        self.config_path = Path(config_path)
        self.data_path = Path(data_path)
        
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['train_settings']
        
        # Datos preparados (se llenan en prepare_data)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self):
        """Prepara datos usando lógica de BaseTrainer."""
        
        # Carga los datos
        df = pd.read_csv(self.data_path, sep=',', on_bad_lines='skip')
        target_col = self.config['preprocessing']['target_variable']
        
        # Convierte las columnas numéricas
        for col in df.columns:
            if col != target_col and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Mapeo de clases según configuración
        class_type = self.config['preprocessing']['classification_type']
        mapping = self.config['preprocessing']['class_mappings'][class_type]

        # Filtrar filas con clases no mapeadas
        valid_mask = df[target_col].isin(mapping.keys())
        df = df[valid_mask].copy()
        df[target_col] = df[target_col].replace(mapping).astype('int64')
        
        print(f"Mapeo '{class_type}' aplicado: {mapping}")
        print(f"Filas después del filtrado: {len(df)}")
        
        # Separar X, y
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        
        print(f"Datos: {X.shape[0]} muestras, {X.shape[1]} features")
        
        # Split en base a configuración
        data_cfg = self.config['data_split']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=data_cfg['test_size'],
            random_state=data_cfg['random_state'],
            stratify=y if data_cfg['stratify'] else None
        )
        
        # Escalar según configuración
        scaler_type = self.config['preprocessing']['scaler_type']
        scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # SMOTE si está habilitado
        if self.config['preprocessing']['use_smote']:
            smote = SMOTE(**self.config['preprocessing']['smote_params'])
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"SMOTE aplicado: {len(y_train)} muestras de entrenamiento")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return self
    
    def search(self, model_type: str, param_grid: dict, n_jobs: int = -1):
        """
        Búsqueda exhaustiva de hiperparámetros.
        
        Args:
            model_type: 'svm', 'xgboost', 'adaboost', 'logistic_lasso'
            param_grid: Diccionario con listas de valores por parámetro
            n_jobs: Número de workers (-1 = todos los CPUs)
        """
        if self.X_train is None:
            raise ValueError("Llama prepare_data() primero")
        
        # Genera las combinaciones
        param_names = list(param_grid.keys())
        combinations = [
            dict(zip(param_names, combo))
            for combo in product(*param_grid.values())
        ]
        total = len(combinations)

        # Determina número de workers y chunksize para multiprocessing
        n_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        chunksize = max(1, total // (n_workers * 8))
        
        # Imprimir configuración de la búsqueda
        print(f"\n{'='*70}")
        print(f"🔍 BÚSQUEDA {model_type.upper()}")
        print(f"{'='*70}")
        print(f"Combinaciones: {total:,}")
        print(f"Workers: {n_workers}")
        for param, values in param_grid.items():
            print(f"  {param}: {len(values)} valores")
        print(f"{'='*70}\n")
        
        # Preparar argumentos
        args_list = [(model_type, params) for params in combinations]
        
        # Resultados y mejor resultado inicial
        best_result = {'test_accuracy': 0.0}
        results = []
        
        # Ejecutar en paralelo
        with Pool(
            # Número de procesos para el pool
            processes=n_workers,
            # Función de inicialización para cada worker
            initializer=_init_worker,
            # Argumentos para la función de inicialización
            initargs=(self.X_train, self.X_test, self.y_train, self.y_test, self.config)
        ) as pool:
            
            # Barra de progreso con tqdm
            with tqdm(total=total, desc=f"🚀 {model_type.upper()}", unit="modelo") as pbar:
                # Iterar sobre los resultados conforme se completan
                for result in pool.imap_unordered(_train_combination, args_list, chunksize=chunksize):
                    results.append(result)
                    # Actualizar mejor resultado si es necesario
                    if result.get('test_accuracy', 0) > best_result.get('test_accuracy', 0):
                        best_result = result
                        tqdm.write(f"✅ MEJOR: {result['test_accuracy']:.4f}")
                    # Actualizar barra de progreso
                    pbar.update(1)
        
        # Imprimir resumen
        self._print_summary(results, best_result, model_type)
        
        return best_result
    
    def _print_summary(self, results: list, best: dict, model_type: str):
        """Imprime resumen de resultados."""
        valid = [r for r in results if 'error' not in r]
        errors = len(results) - len(valid)
        
        # Resumen de resultados
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN {model_type.upper()}")
        print(f"{'='*70}")
        
        if errors > 0:
            print(f"⚠️ Errores: {errors}/{len(results)}")
        
        # Ordena los resultados por test_accuracy descendente
        sorted_results = sorted(valid, key=lambda x: x['test_accuracy'], reverse=True)
        
        # Imprimir top 10 resultados
        print(f"\n🏆 TOP 10:")
        print(f"{'-'*70}")
        for i, r in enumerate(sorted_results[:10], 1):
            print(f"{i}. Test: {r['test_accuracy']:.4f} | Train: {r['train_accuracy']:.4f} | Overfit: {r['overfit_diff']:.4f}")
        
        # Imprimir mejor configuración
        if valid:
            print(f"\n{'='*70}")
            print(f"🎯 MEJOR CONFIGURACIÓN")
            print(f"{'='*70}")
            for k, v in best.items():
                if k not in ['test_accuracy', 'train_accuracy', 'overfit_diff', 'precision', 'recall', 'f1_score']:
                    print(f"  {k}: {v}")
            print(f"\n{'-'*70}")
            print(f"  Métricas:\n")
            print(f"  Train Acc: {best['train_accuracy']:.4f}")
            print(f"  Test Acc:  {best['test_accuracy']:.4f}")
            print(f"  overfit:  {best['overfit_diff']:.4f}")
            print (f"  Precision: {best['precision']:.4f}")
            print(f"  Recall:    {best['recall']:.4f}")
            print(f"  F1-Score:  {best['f1_score']:.4f}")
            print(f"{'='*70}\n")


# ==================== PARAM GRIDS ====================

PARAM_GRIDS = {
    'svm': {
        'C': [
                        # Rango muy bajo (0.1 - 1.0) - MUY DENSO por mejores resultados
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.0,
            # Rango bajo-medio (3 - 10)
            4, 5, 6, 7, 8, 9, 10,
            # Rango medio (12 - 50)
            12, 15, 18, 20, 25, 30, 35, 40, 44, 45, 50,
            # Rango alto (60 - 150)
            60, 70, 80, 90, 100],
        'gamma': [
            # Rango muy bajo (0.00001 - 0.001)
            0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001,
            # Rango bajo (0.001 - 0.01) 
            0.001664, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.01,
            # Rango medio-bajo (0.01 - 0.1) - MUY DENSO por mejores resultados
            0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.1,
            # Rango medio (0.1 - 0.3) - CRÍTICO: contiene γ=0.16 y γ=0.2
            0.11, 0.12, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.18, 0.19, 0.2, 0.25, 0.26, 0.3,
            # Rango medio-alto (0.3 - 0.6)
            0.4, 0.4664, 0.5, 0.55, 0.6,
            # Rango alto (0.6 - 1.0)
            0.65, 0.7, 0.8, 0.9, 1.0,
            # Auto
            'scale'
        ],
        'tol': [
            # Rango muy bajo (0.00001 - 0.0001) - CRÍTICO: contiene tol=0.0001
            0.00001, 0.00002, 0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.001,
            # Rango bajo (0.001 - 0.01)
            0.00166, 0.002, 0.003, 0.005, 0.007, 0.008, 0.01,
            # Rango medio (0.01 - 0.1)
            0.015, 0.02, 0.03, 0.05, 0.07, 0.08, 0.1,
            # Rango medio-alto (0.1 - 0.5) - CRÍTICO: contiene tol=0.5
            0.12, 0.15, 0.16, 0.16446, 0.18, 0.2, 0.25, 0.28, 0.3, 0.35, 0.4, 0.45, 0.5,
            # Rango alto (0.5 - 1.0)
            0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0
        ]
    },
    'xgboost': {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.01, 0.02, 0.046664, 0.05, 0.1, 0.3],
        'max_depth': [3],
        'subsample': [0.5, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
        'gamma': [0.1, 0.2, 0.4, 0.5, 0.6, 1.0],
        'min_child_weight': [3],
        'reg_alpha': [0.1, 0.2, 0.4, 0.5, 0.6, 1.0],
        'reg_lambda': [0.1, 0.2, 0.4, 0.5, 0.6, 1.0, 0.01, 1.0]
    },
    'adaboost': {
        'n_estimators': [10, 100, 300, 400],
        'learning_rate': [0.01, 0.02, 0.046664, 0.05, 0.1, 0.3, 1.0, 1.2, 1.3, 1.4, 1.6],
        'estimator__max_depth': [1, 2, 3, 4, 5],
        'estimator__criterion': ['entropy', 'gini'],
        'estimator__min_samples_split': [2, 3, 4, 5],
        'estimator__min_samples_leaf': [1, 2, 3, 4, 5],
        'estimator__max_features': [None, 'sqrt'],
        'estimator__min_impurity_decrease': [0.0]
    },
    'logistic_lasso': {
        'C': [
            # Rango ultra bajo (0.001 - 0.1) - NUEVO: valores muy restrictivos
            0.001, 0.002, 0.005
        ],
        'tol': [
            # Rango ultra bajo (0.000001 - 0.00001) - NUEVO: convergencia muy estricta
            0.000001, 0.000002
        ]
    }
}


# ==================== MAIN ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Búsqueda de hiperparámetros')
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--data', required=True, help='Ruta al CSV')
    parser.add_argument('--model', required=True, choices=['svm', 'xgboost', 'adaboost', 'logistic_lasso'])
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()
    
    # Crear buscador
    searcher = HyperparameterSearcher(args.config, args.data)
    searcher.prepare_data()
    
    # Obtener param_grid
    param_grid = PARAM_GRIDS.get(args.model)
    if not param_grid:
        print(f"Modelo no soportado: {args.model}")
        return
    
    # Ejecutar búsqueda
    best = searcher.search(args.model, param_grid, n_jobs=args.n_jobs)
    
    print("\n🎉 Búsqueda completada!")
    return best


if __name__ == '__main__':
    main()