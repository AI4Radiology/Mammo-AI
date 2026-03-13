"""
Factory minimal para construir modelos de PyTorch Tabular con Optuna
usando configuración YAML.
"""
from pathlib import Path
import yaml
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig


def create_model_for_optuna(trial, continuous_cols):
    """
    Crea un modelo de PyTorch Tabular para Optuna.
    Carga la configuración del YAML y sugiere hiperparámetros desde el trial.
    
    Args:
        trial: Trial de Optuna
        continuous_cols: Lista de columnas continuas
    
    Returns:
        TabularModel configurado con hiperparámetros del trial
    """
    # Cargar config
    p = Path(__file__).parent / "pytorch_tabular_optuna_config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f).get("pytorch_tabular_optuna_settings", {})
    
    data_cfg = c.get("data", {})
    trainer_cfg = c.get("trainer", {})
    model_cfg = c.get("model", {})
    space = c.get("search_space", {})

    # Sugerir hiperparámetros desde el search_space del YAML
    suggested = {}
    for name, spec in space.items():
        t = spec.get("type")
        if t == "float":
            low, high = float(spec["low"]), float(spec["high"])
            log = bool(spec.get("log", False))
            step = spec.get("step")
            if step and not log:
                suggested[name] = trial.suggest_float(name, low, high, step=float(step), log=False)
            else:
                suggested[name] = trial.suggest_float(name, low, high, log=log)
        elif t == "int":
            low, high = int(spec["low"]), int(spec["high"])
            step = spec.get("step")
            suggested[name] = trial.suggest_int(name, low, high, step=int(step)) if step else trial.suggest_int(name, low, high)
        elif t == "categorical":
            suggested[name] = trial.suggest_categorical(name, spec["choices"])

    # Obtener valores por defecto del YAML
    max_epochs = int(trainer_cfg.get("max_epochs", 50))
    use_batch_norm = model_cfg.get("use_batch_norm", True)
    initialization = model_cfg.get("initialization", "kaiming")
    early_stopping = trainer_cfg.get("early_stopping", "valid_loss")
    early_stopping_patience = trainer_cfg.get("early_stopping_patience", 5)
    progress_bar = trainer_cfg.get("progress_bar", "none")

    # Normalizar target
    t = data_cfg.get("target")
    target_list = t if isinstance(t, list) else [t]

    # Construir configuraciones
    data_config = DataConfig(
        target=target_list,
        continuous_cols=continuous_cols,
        categorical_cols=[],
        num_workers=0,
        pin_memory=False,
    )

    trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=suggested['batch_size'],
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        checkpoints=None,
        load_best=False,
        progress_bar=progress_bar,
    )

    optimizer_config = OptimizerConfig(
        optimizer=suggested['optimizer'],
        optimizer_params={"weight_decay": suggested['weight_decay']},
    )

    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers=suggested['layers'],
        activation=suggested['activation'],
        learning_rate=suggested['learning_rate'],
        dropout=suggested['dropout'],
        use_batch_norm=use_batch_norm,
        initialization=initialization,
    )

    return TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
