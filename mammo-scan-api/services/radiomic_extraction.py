import argparse
import json
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from radiomics import featureextractor
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    yaml = None
    YAML_AVAILABLE = False

class RadiomicsMLTrainer:
    
    def __init__(self, results_folder="results"):
        self.results_folder = Path(results_folder)
        # Intentar cargar configuración externa de radiomics
        self.radiomics_config = {}
        self.rad_cfg = {}
        try:
            if YAML_AVAILABLE:
                cfg_path = Path('config/radiomics_config.yaml')
                if cfg_path.exists():
                    with open(cfg_path, 'r', encoding='utf-8') as f:
                        loaded = yaml.safe_load(f)
                        if loaded:
                            self.radiomics_config = loaded
                            self.rad_cfg = loaded.get('radiomic_extraction_settings', {}) or {}
                            print(f"Configuración radiomics cargada desde: {cfg_path}")
                else:
                    # no hay archivo, usar valores por defecto
                    self.rad_cfg = {}
            else:
                print('Advertencia: PyYAML no disponible; usando configuración por defecto en el script')
                self.rad_cfg = {}
        except Exception as e:
            print(f"Advertencia: error cargando radiomics_config.yaml: {e}")
            self.rad_cfg = {}
    
    def extract_radiomics_features(self, image, mask):
        """
        Extraer características radiómicas usando PyRadiomics.
        
        IMPORTANTE: No se debe normalizar la imagen ni modificar sus intensidades.
        Los valores originales del DICOM son CRÍTICOS para obtener radiomics precisos.
        
        Args:
            image: numpy array float64 con valores originales del DICOM
            mask: numpy array uint8 binario (0 y 1)
            
        Returns:
            dict con características radiómicas extraídas
        """
        try:
            # Validar que las imágenes no estén vacías
            if image is None or mask is None:
                print("Error: Imagen o máscara es None")
                return None
            
            # Validar que tengan las mismas dimensiones
            if image.shape != mask.shape:
                print(f"Error: Dimensiones no coinciden - Imagen: {image.shape}, Máscara: {mask.shape}")
                return None
            
            # Asegurar que sean arrays numpy
            image = np.asarray(image, dtype=np.float64)
            mask = np.asarray(mask, dtype=np.uint8)
            
            # NO normalizar la imagen - mantener valores originales del DICOM
            # Esta es la configuración crítica para obtener radiomics precisos
            
            # Asegurar que la máscara sea binaria (0 y 1 solamente)
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)
            
            # Verificar que la máscara tenga al menos algunos píxeles activos
            if np.sum(mask) == 0:
                print("Advertencia: Máscara vacía (todos los píxeles son 0)")
                return None
            
            # Configurar extractor de características
            # CRÍTICO: normalize debe ser False para mantener valores originales
            settings = {
                'binWidth': 25,
                'normalize': False,  # CRÍTICO: NO normalizar
                'force2D': True,
                'force2Ddimension': 0
            }
            
            # Cargar configuración adicional si existe
            cfg = getattr(self, 'rad_cfg', {}) or {}
            
            # Solo aplicar enableImageTypes si está en config
            if 'enableImageTypes' in cfg:
                settings['enableImageTypes'] = cfg['enableImageTypes']
            
            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            
            # Habilitar clases de características
            extractor.enableImageTypeByName('Original')
            extractor.enableFeatureClassByName('firstorder')
            extractor.enableFeatureClassByName('glcm')
            extractor.enableFeatureClassByName('glrlm')
            extractor.enableFeatureClassByName('glszm')
            extractor.enableFeatureClassByName('gldm')
            extractor.enableFeatureClassByName('ngtdm')
            extractor.enableFeatureClassByName('shape2D')
            
            # Convertir a SimpleITK
            image_sitk = sitk.GetImageFromArray(image)
            mask_sitk = sitk.GetImageFromArray(mask)

            # Extraer características
            result = extractor.execute(image_sitk, mask_sitk)

            # Convertir a diccionario de características
            features = {}
            for key, value in result.items():
                if not key.startswith('diagnostics'):
                    try:
                        features[key] = float(value)
                    except:
                        pass

            return features
            
        except Exception as e:
            print(f"Error extrayendo características radiómicas: {e}")
            import traceback
            traceback.print_exc()
            return None