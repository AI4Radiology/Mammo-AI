"""
Modelo XGBoost básico usando configuración YAML
"""
import yaml
from pathlib import Path
from xgboost import XGBClassifier


def create_model():
    """
    Crear modelo XGBoost usando configuración YAML.
    """
    # Cargar configuración desde la misma carpeta del modelo
    config_path = Path(__file__).parent / "xgboost_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    general = config['xgboost_settings']['general']
    model_params = config['xgboost_settings']['model']

    xgb_args = {**general, **model_params}

    # Convertir valores 'null' o None a None real
    for k, v in xgb_args.items():
        if v in ['null', None]:
            xgb_args[k] = None

    return XGBClassifier(**xgb_args)
