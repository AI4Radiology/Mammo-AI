"""
Modelo AdaBoost básico usando configuración YAML
"""
import yaml
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def create_model():
    """
    Crear modelo AdaBoost usando configuración YAML.
    """
    # Cargar configuración desde la misma carpeta del modelo
    config_path = Path(__file__).parent / "adaboost_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    general = config['adaboost_settings']['general']
    model_params = config['adaboost_settings']['model']
    base_params = config['adaboost_settings']['base_estimator']
    
    # CAMBIO CRÍTICO: Crear estimador base con sus parámetros del YAML
    base_estimator = DecisionTreeClassifier(
        max_depth=base_params['max_depth'],
        criterion=base_params['criterion'],
        min_samples_split=base_params['min_samples_split'],
        min_samples_leaf=base_params['min_samples_leaf'],
        random_state=general['random_state'],
        max_features=base_params['max_features'],
        min_impurity_decrease=base_params['min_impurity_decrease']
    )
    
    # Crear AdaBoost con el estimador base configurado
    return AdaBoostClassifier(
        estimator=base_estimator,  # ✅ AHORA USA EL ESTIMADOR BASE
        n_estimators=model_params['n_estimators'],
        learning_rate=model_params['learning_rate'],
        random_state=general['random_state']
    )