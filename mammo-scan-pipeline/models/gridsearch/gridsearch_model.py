"""
Modelo SVM con GridSearch - Ultra simplificado usando YAML
"""
import yaml
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def create_model():
    """
    Crear modelo SVM con GridSearch usando configuración YAML.
    """
    # Cargar configuración desde la misma carpeta del modelo
    config_path = Path(__file__).parent / "gridsearch_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    general = config['gridsearch_settings']['general']
    
    params = config['gridsearch_settings']['balanced']
    
    # Convertir YAML a param_grid
    param_grid = {
        param: [None if v in ['null', None] else v for v in values]
        for param, values in params.items()
    }
    
    # Crear modelo
    svm = SVC(random_state=general['random_state'])
    cv = StratifiedKFold(n_splits=general['cv_folds'], shuffle=True, random_state=general['random_state'])
    
    return GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=cv,
        scoring=general['scoring'],
        n_jobs=general['n_jobs'],
        verbose=general['verbose'],
        return_train_score=general['return_train_score']
    )


