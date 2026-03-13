"""
Modelo Logistic Regression con regularización Lasso usando configuración YAML
"""
import yaml
from pathlib import Path
from sklearn.linear_model import LogisticRegression


def create_model():
    """
    Crear modelo Logistic Regression con Lasso usando configuración YAML.
    """
    # Cargar configuración desde la misma carpeta del modelo
    config_path = Path(__file__).parent / "logistic_lasso_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    general = config['logistic_lasso_settings']['general']
    model_params = config['logistic_lasso_settings']['model']
    
    # Combinar configuración general con parámetros del modelo
    logistic_params = {
        'random_state': general['random_state'],
        'max_iter': general['max_iter'],
        **model_params
    }
    
    # Convertir valores None del YAML
    for key, value in logistic_params.items():
        if value in ['null', None]:
            logistic_params[key] = None
    
    return LogisticRegression(**logistic_params)