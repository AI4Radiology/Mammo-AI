"""
Modelo SVM básico usando configuración YAML
"""
import yaml
from pathlib import Path
from sklearn.svm import SVC


def create_model():
    """
    Crear modelo SVM usando configuración YAML.
    """
    # Cargar configuración desde la misma carpeta del modelo
    config_path = Path(__file__).parent / "svm_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    general = config['svm_settings']['general']
    model_params = config['svm_settings']['model']
    
    # Combinar configuración general con parámetros del modelo
    svm_params = {
        'random_state': general['random_state'],
        'probability': general['probability'],
        'verbose': general['verbose'],
        **model_params
    }
    
    # Convertir valores None del YAML
    for key, value in svm_params.items():
        if value in ['null', None]:
            svm_params[key] = None
    
    return SVC(**svm_params)
