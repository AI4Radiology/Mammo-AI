"""
Modelo AdaBoost con GridSearch usando configuración YAML
"""
import yaml
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def create_model():
    """
    Crear modelo AdaBoost con GridSearch usando configuración YAML.
    """
    # Cargar configuración desde la misma carpeta del modelo
    config_path = Path(__file__).parent / "gridsearch_adaboost_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    general = config['gridsearch_adaboost_settings']['general']
    params = config['gridsearch_adaboost_settings']['balanced']
    
    # Convertir YAML a param_grid
    param_grid = {}
    for param, values in params.items():
        if param.startswith('estimator__'):
            # Parámetros para el estimador base
            param_grid[param] = [None if v in ['null', None] else v for v in values]
        else:
            # Parámetros del AdaBoost
            param_grid[param] = [None if v in ['null', None] else v for v in values]
    
    # Crear modelo AdaBoost con DecisionTreeClassifier como estimador base
    base_estimator = DecisionTreeClassifier(random_state=general['random_state'])
    adaboost = AdaBoostClassifier(
        estimator=base_estimator,
        random_state=general['random_state']
    )
    
    # Configurar validación cruzada
    cv = StratifiedKFold(
        n_splits=general['cv_folds'], 
        shuffle=True, 
        random_state=general['random_state']
    )
    
    return GridSearchCV(
        estimator=adaboost,
        param_grid=param_grid,
        cv=cv,
        scoring=general['scoring'],
        n_jobs=general['n_jobs'],
        verbose=general['verbose'],
        return_train_score=general['return_train_score']
    )