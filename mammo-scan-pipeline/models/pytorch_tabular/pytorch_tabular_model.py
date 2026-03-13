"""
Modelo PyTorch Tabular usando configuración YAML
"""
import yaml
from pathlib import Path
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)


def create_model(continuous_cols=None, categorical_cols=None):
    """
    Crear modelo PyTorch Tabular usando configuración YAML.
    
    Args:
        continuous_cols: Lista de columnas continuas. Si es None, se usará configuración del YAML.
        categorical_cols: Lista de columnas categóricas. Si es None, se usará configuración del YAML.
    
    Returns:
        TabularModel: Modelo configurado listo para entrenar
    """
    # Cargar configuración desde la misma carpeta del modelo
    config_path = Path(__file__).parent / "pytorch_tabular_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['pytorch_tabular_settings']['data']
    trainer_cfg = config['pytorch_tabular_settings']['trainer']
    optimizer_cfg = config['pytorch_tabular_settings']['optimizer']
    model_cfg = config['pytorch_tabular_settings']['model']
    
    # Normalizar target a List[str]
    t = data_cfg.get('target')
    if isinstance(t, str):
        target_list = [t]
    elif isinstance(t, list):
        target_list = t
    else:
        raise ValueError(f"Unsupported type for 'target' in config: {type(t)}")

    # Configuración de datos
    data_config = DataConfig(
        target=target_list,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )
    
    # Configuración del entrenador
    trainer_params = {
        'auto_lr_find': trainer_cfg['auto_lr_find'],
        'batch_size': trainer_cfg['batch_size'],
        'max_epochs': trainer_cfg['max_epochs'],
    }
    
    # Agregar parámetros opcionales si existen
    optional_trainer = ['early_stopping', 'early_stopping_patience', 'checkpoints', 'load_best', 'progress_bar']
    for param in optional_trainer:
        if param in trainer_cfg and trainer_cfg[param]:
            trainer_params[param] = trainer_cfg[param]
    
    trainer_config = TrainerConfig(**trainer_params)
    
    # Configuración del optimizador
    optimizer_params = {
        'optimizer': optimizer_cfg.get('optimizer', 'Adam'),
    }
    if optimizer_cfg.get('optimizer_params'):
        optimizer_params['optimizer_params'] = optimizer_cfg['optimizer_params']
    
    optimizer_config = OptimizerConfig(**optimizer_params)
    
    # Configuración del modelo
    model_params = {
        'task': model_cfg['task'],
        'layers': model_cfg['layers'],
        'activation': model_cfg['activation'],
        'learning_rate': model_cfg['learning_rate'],
    }
    
    # Agregar parámetros opcionales si existen
    optional_model = ['dropout', 'use_batch_norm', 'initialization']
    for param in optional_model:
        if param in model_cfg:
            model_params[param] = model_cfg[param]
    
    model_config = CategoryEmbeddingModelConfig(**model_params)
    

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    
    return tabular_model


def load_model(model_path):
    return TabularModel.load_model(model_path)
