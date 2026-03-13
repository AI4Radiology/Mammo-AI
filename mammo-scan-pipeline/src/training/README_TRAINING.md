
### Script completo clasificación binaria
```bash
python -m src.training.train_generic --config configs/train_config.yaml --data data/features/MG_Tipo_Tejido_4Cl_Evento.csv --features data/features/best_features_20250907_202035/kbest.txt 
```

```bash
python -m src.training.train_generic --config configs/train_config.yaml --data data/features/best_features_20251003_113727/pca_transformed.csv
```

## Con Eda binario
```bash
python -m src.training.train_generic --config configs/train_config.yaml --data data/features/best_EDA/radiomics_bin_kbest.csv --features data/features/best_EDA/kbest_bin.txt
```
## Con Eda multiclase
```bash
python -m src.training.train_generic --config configs/train_config.yaml --data data/features/best_EDA/radiomics_mult_kbest.csv --features data/features/best_EDA/kbest_mult.txt
```
## Pythorch tabular
```bash
python -m src.training.train_pytorch_tabular.py --config configs/train_config.yaml --data data/features/best_EDA/radiomics_bin_kbest.csv --optimize --n-trials 10
```

## Guardar modelo
```bash
python -m src.training.train_generic --config configs/train_config.yaml --data data/features/best_EDA/radiomics_mult_kbest.csv --features data/features/best_EDA/kbest_mult.txt --save-model
```

## Entrenar con combinatoria de parámetros
```bash
python src/training/test.py --data data/features/df_sel_bin.csv --model logistic_lasso
```

## Entrenar con ensamble de modelos
```bash
python -m src.training.train_ensemble --config configs/ensemble_config.yaml 
```
