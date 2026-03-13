"""
Modelo XGBoost con GridSearch usando YAML
"""
import yaml
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def create_model():
    """
    Crear modelo XGBoost con GridSearch usando configuración YAML.
    """
    # Cargar configuración desde la misma carpeta del modelo
    config_path = Path(__file__).parent / "gridsearch_xgboost_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    general = config['gridsearch_xgboost_settings']['general']
    params = config['gridsearch_xgboost_settings']['param_grid']
    
    # Convertir YAML a param_grid
    param_grid = {
        param: [None if v in ['null', None] else v for v in values]
        for param, values in params.items()
    }
    
    # Crear modelo base
    # XGBClassifier detecta automáticamente el número de clases
    xgb = XGBClassifier(
        random_state=general['random_state'],
        eval_metric='logloss',  # Funciona para binario y multiclase
        verbosity=0,
        tree_method='hist'  # Fijamos 'hist' para CPU rápido (fuera del grid)
    )
    
    # Crear validación cruzada estratificada
    cv = StratifiedKFold(
        n_splits=general['cv_folds'], 
        shuffle=True, 
        random_state=general['random_state']
    )
    
    # Crear GridSearchCV
    return GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=cv,
        scoring=general['scoring'],
        n_jobs=general['n_jobs'],
        verbose=general['verbose'],
        return_train_score=general['return_train_score']
    )
