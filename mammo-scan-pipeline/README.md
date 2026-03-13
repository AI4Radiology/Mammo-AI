# PROYECTO GRADO - Sistema de Análisis Radiómico para Clasificación de Densidad Mamaria

Sistema completo para el procesamiento de imágenes DICOM mamográficas, extracción de características radiómicas y clasificación mediante modelos de Machine Learning.

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Configuración](#configuración)
- [Pipeline de Ejecución](#pipeline-de-ejecución)
  - [Procesamiento de Imágenes DICOM](#procesamiento-de-imágenes-dicom)
  - [Extracción de Características](#extracción-de-características)
  - [Entrenamiento de Modelos](#entrenamiento-de-modelos)
- [Comandos de Ejecución](#comandos-de-ejecución)
- [Archivos de Configuración](#archivos-de-configuración)

---

## Descripción General

Este proyecto implementa un pipeline completo para:

1. Procesamiento de imágenes DICOM mamográficas
2. Adición y validación de etiquetas de densidad mamaria
3. Extracción de características radiómicas
4. Selección de características óptimas
5. Entrenamiento y evaluación de modelos de clasificación
6. Búsqueda de hiperparámetros óptimos

---

## Estructura del Proyecto

```
DENSITY_CLASSIFICATION/
├── configs/                          # Archivos de configuración YAML
│   ├── train_config.yaml             # Configuración principal de entrenamiento
│   ├── ensemble_config.yaml          # Configuración para ensambles de modelos
│   └── radiomics_config.yaml         # Configuración de extracción radiómicas
│
├── data/                             # Datos del proyecto (NO incluido en repositorio)
│   ├── dicom/                        # Imágenes DICOM originales
│   ├── features/                     # Características extraídas (CSV)
│   │   └── best_EDA/                 # Mejores features seleccionadas
│   └── processed/                    # Datos procesados intermedios
│
├── models/                           # Definición de modelos ML
│   ├── adaboost/                     # Modelo AdaBoost
│   │   ├── adaboost_config.yaml
│   │   └── adaboost_model.py
│   ├── svm/                          # Modelo SVM
│   │   ├── svm_config.yaml
│   │   └── svm_model.py
│   ├── xgboost/                      # Modelo XGBoost
│   │   ├── xgboost_config.yaml
│   │   └── xgboost_model.py
│   ├── logistic_lasso/               # Regresión Logística con L1
│   │   ├── logistic_lasso_config.yaml
│   │   └── logistic_lasso_model.py
│   ├── pytorch_tabular/              # Modelos de PyTorch Tabular
│   │   ├── pytorch_tabular_config.yaml
│   │   └── pytorch_tabular_model.py
│   ├── gridsearch/                   # Configuración GridSearch genérico
│   ├── gridsearch_adaboost/          # GridSearch específico AdaBoost
│   ├── gridsearch_logistic_lasso/    # GridSearch específico Logistic Lasso
│   └── gridsearch_xgboost/           # GridSearch específico XGBoost
│
├── notebooks/                        # Jupyter notebooks para exploración
│
├── outputs/                          # Resultados y modelos guardados
│   └── saved_models/                 # Modelos entrenados con métricas
│
├── src/                              # Código fuente principal
│   ├── scripts/                      # Scripts de procesamiento
│   │   ├── addDensity.py             # Agregar tag de densidad a DICOM
│   │   ├── test_density_mapping.py   # Validar tags de densidad
│   │   ├── dicom_image_processing.py # Procesamiento de imágenes DICOM
│   │   ├── radiomic_extraction.py    # Extracción de características
│   │   └── feature_selection.py      # Selección de características
│   │
│   └── training/                     # Scripts de entrenamiento
│       ├── base_trainer.py           # Clase base abstracta para trainers
│       ├── train_generic.py          # Entrenamiento genérico de modelos
│       ├── train_pytorch_tabular.py  # Entrenamiento con PyTorch Tabular
│       ├── evaluate_ensemble.py      # Evaluación de ensambles
│       └── test.py                   # Búsqueda exhaustiva de hiperparámetros
│
└── README.md                         
```

### Nota sobre la carpeta `data/`

La carpeta `data/` no se incluye en el repositorio por restricciones de espacio. Al clonar el proyecto, debe recrearse con la siguiente estructura:

```
data/
├── dicom/              # Colocar imágenes DICOM originales aquí
├── features/           # Se generan automáticamente por los scripts
│   └── best_EDA/       # Crear manualmente o mediante feature_selection.py
└── processed/          # Se genera automáticamente
```

---

## Requisitos

### Dependencias principales

```
python >= 3.9
scikit-learn
xgboost
imbalanced-learn
pandas
numpy
pydicom
pyradiomics
pytorch-tabular
optuna
tqdm
pyyaml
```

### Instalación

```bash
# Crear entorno virtual
python -m venv env

# Activar entorno (Windows)
env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## Configuración

### Archivos de Configuración

Todos los archivos de configuración se encuentran en la carpeta `configs/`:

#### train_config.yaml

Archivo principal de configuración para entrenamiento. Define:

- **model**: Tipo de modelo a usar y mapeo de módulos
  - `type`: Nombre del modelo (svm, xgboost, adaboost, logistic_lasso, pytorch_tabular)
  - `module_map`: Mapeo de tipos a rutas de módulos
  
- **preprocessing**: Configuración de preprocesamiento
  - `target_variable`: Nombre de la columna objetivo
  - `classification_type`: Tipo de clasificación (b_binary, multiclass, etc.)
  - `class_mappings`: Mapeo de clases para cada tipo de clasificación
  - `scaler_type`: Tipo de escalador (standard, minmax)
  - `use_smote`: Activar/desactivar balanceo de clases
  - `smote_params`: Parámetros de SMOTE
  
- **data_split**: Configuración de división de datos
  - `test_size`: Proporción de datos para test
  - `random_state`: Semilla para reproducibilidad
  - `stratify`: Estratificación por clase
  
- **metrics**: Configuración de métricas de evaluación

Modificar este archivo cuando:
- Se cambie el tipo de modelo a entrenar
- Se modifique la clasificación (binaria vs multiclase)
- Se ajusten parámetros de preprocesamiento
- Se cambien configuraciones de división de datos

#### ensemble_config.yaml

Configuración para entrenamiento de ensambles de modelos. Define:

- Lista de modelos a combinar
- Método de votación (hard/soft)
- Pesos de cada modelo

Modificar cuando se requiera entrenar ensambles de múltiples modelos.

#### radiomics_config.yaml

Configuración para extracción de características radiómicas. Define:

- Tipos de características a extraer
- Parámetros de filtros de imagen
- Configuración de discretización

Modificar cuando se requieran diferentes características radiómicas.

### Configuración de Modelos Individuales

Cada modelo tiene su propio archivo de configuración en `models/<modelo>/<modelo>_config.yaml`:

- `models/svm/svm_config.yaml`: Parámetros C, gamma, kernel, etc.
- `models/xgboost/xgboost_config.yaml`: n_estimators, learning_rate, max_depth, etc.
- `models/adaboost/adaboost_config.yaml`: n_estimators, learning_rate, base_estimator
- `models/logistic_lasso/logistic_lasso_config.yaml`: C, tol, max_iter

Modificar estos archivos para ajustar hiperparámetros específicos de cada modelo.

---

## Pipeline de Ejecución

### Procesamiento de Imágenes DICOM

El pipeline de procesamiento depende de si las imágenes DICOM ya contienen el tag de densidad mamaria.

#### Caso 1: Imágenes SIN tag de densidad

Si las imágenes DICOM no tienen el tag de densidad integrado:

1. **Agregar tag de densidad**
   ```bash
   python src/scripts/addDensity.py
   ```

2. **Validar tags agregados**
   ```bash
   python src/scripts/test_density_mapping.py
   ```

3. **Continuar con procesamiento normal** (ver Caso 2)

#### Caso 2: Imágenes CON tag de densidad

1. **Procesamiento de imágenes DICOM**
   ```bash
   python src/scripts/dicom_image_processing.py
   ```

2. **Extracción de características radiómicas**
   ```bash
   python src/scripts/radiomic_extraction.py -r data/features
   ```

3. **Selección de características**
   ```bash
   python src/scripts/feature_selection.py -r MG_Tipo_Tejido_4Cl_Evento.csv
   ```
   
   Si no se especifica archivo, usa el CSV más reciente:
   ```bash
   python src/scripts/feature_selection.py
   ```

### Extracción de Características

El script `feature_selection.py` realiza:

- Carga de datos desde CSV
- Codificación de variables categóricas
- Aplicación de SMOTE para balanceo de clases
- Selección de características con SelectKBest
- Reducción de dimensionalidad con PCA
- Guardado de resultados

### Entrenamiento de Modelos

#### Preparación

1. Configurar el tipo de modelo en `configs/train_config.yaml`:
   ```yaml
   model:
     type: svm  # Opciones: svm, xgboost, adaboost, logistic_lasso, pytorch_tabular
   ```

2. Ajustar parámetros específicos del modelo en su archivo de configuración correspondiente

#### Entrenamiento Básico

```bash
python -m src.training.train_generic --config configs/train_config.yaml --data <ruta_csv>
```

#### Entrenamiento con Guardado de Modelo

```bash
python -m src.training.train_generic --config configs/train_config.yaml --data <ruta_csv> --save-model
```

#### Búsqueda de Hiperparámetros

Para encontrar los mejores hiperparámetros de un modelo:

```bash
python src/training/test.py --data <ruta_csv> --model <tipo_modelo>
```

Modelos disponibles: `svm`, `xgboost`, `adaboost`, `logistic_lasso`

---

## Comandos de Ejecución

### Scripts de Procesamiento

| Comando | Descripción |
|---------|-------------|
| `python src/scripts/addDensity.py` | Agrega tag de densidad a imágenes DICOM |
| `python src/scripts/test_density_mapping.py` | Valida tags de densidad en imágenes DICOM |
| `python src/scripts/dicom_image_processing.py` | Procesa imágenes DICOM |
| `python src/scripts/radiomic_extraction.py -r data/features` | Extrae características radiómicas |
| `python src/scripts/feature_selection.py -r <archivo.csv>` | Selección de características |

### Scripts de Entrenamiento

| Comando | Descripción |
|---------|-------------|
| `python -m src.training.train_generic --config configs/train_config.yaml --data <csv>` | Entrenamiento básico |
| `python -m src.training.train_generic --config configs/train_config.yaml --data <csv> --save-model` | Entrenamiento con guardado |
| `python -m src.training.train_generic --config configs/train_config.yaml --data <csv> --features <features.txt>` | Entrenamiento con features específicas |
| `python -m src.training.train_pytorch_tabular --config configs/train_config.yaml --data <csv> --optimize --n-trials 10` | Entrenamiento PyTorch Tabular con Optuna |
| `python -m src.training.train_ensemble --config configs/ensemble_config.yaml` | Entrenamiento de ensamble |
| `python src/training/test.py --data <csv> --model <modelo>` | Búsqueda exhaustiva de hiperparámetros |

### Ejemplos Completos

#### Clasificación Binaria

```bash
python -m src.training.train_generic --config configs/train_config.yaml --data data/features/best_EDA/radiomics_bin_kbest.csv --features data/features/best_EDA/kbest_bin.txt
```

#### Clasificación Multiclase

```bash
python -m src.training.train_generic --config configs/train_config.yaml --data data/features/best_EDA/radiomics_mult_kbest.csv --features data/features/best_EDA/kbest_mult.txt
```

#### PyTorch Tabular con Optimización

```bash
python -m src.training.train_pytorch_tabular --config configs/train_config.yaml --data data/features/best_EDA/radiomics_bin_kbest.csv --optimize --n-trials 10
```

#### Búsqueda de Hiperparámetros

```bash
python src/training/test.py --data data/features/df_sel_bin.csv --model logistic_lasso
```

---

## Resultados

Los modelos entrenados y sus métricas se guardan en `outputs/saved_models/` con la siguiente estructura:

```
outputs/saved_models/
└── modelo_<tipo>_<timestamp>/
    ├── modelo_<tipo>.pkl          # Modelo serializado
    └── modelo_<tipo>_results.json # Métricas y configuración
```

El archivo JSON de resultados contiene:
- Métricas de entrenamiento y test
- Matriz de confusión
- Parámetros del modelo
- Timestamp de entrenamiento

---

## Notas Adicionales

### Reproducibilidad

Para garantizar reproducibilidad:
- Configurar `random_state` en `train_config.yaml`
- Usar la misma semilla en todos los experimentos

### Multiprocessing

El script `test.py` utiliza todos los cores disponibles menos uno para la búsqueda de hiperparámetros. Ajustar el parámetro `--n-jobs` si es necesario.

### Memoria

Para datasets grandes:
- Reducir el batch size en PyTorch Tabular
- Limitar el número de combinaciones en GridSearch
- Usar muestreo estratificado para validación
