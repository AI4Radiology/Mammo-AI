### Script completo dicom image processing
```bash
python src/scripts/dicom_image_processing.py
```

### Script completo feature selection
```bash
python src/scripts/feature_selection.py -r MG_Tipo_Tejido_4Cl_Evento.csv
```

### Script completo radiomic extraction
```bash
python src/scripts/radiomic_extraction.py -r data/features
```

## Script para añadir tag de densidad a imagenes DICOM
```bash
python src/scripts/addDensity.py
```

## Script para validar tag de densidad en imagenes DICOM
```bash
python src/scripts/test_density_mapping.py
```