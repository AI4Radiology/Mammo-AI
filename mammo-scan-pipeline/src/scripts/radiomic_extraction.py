import argparse
import json
import os
import random
from typing import Optional
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from radiomics import featureextractor
import gc  # Para liberación explícita de memoria
import multiprocessing as mp
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    yaml = None
    YAML_AVAILABLE = False

# =============================
# Backend de workers (globales)
# =============================
_RAD_CFG = None
_EXTRACTOR = None
_SOURCE_FOLDER = 'unknown'

def _build_extractor_from_cfg(cfg: dict):
    settings = {}
    cfg = cfg or {}
    # enableImageTypes
    if cfg.get('enableImageTypes'):
        settings['enableImageTypes'] = cfg.get('enableImageTypes')
    # normalize
    if cfg.get('normalize') is not None:
        settings['normalize'] = cfg.get('normalize')
    if cfg.get('normalizeScale'):
        settings['normalizeScale'] = cfg.get('normalizeScale')
    # pixel spacing
    if cfg.get('pixelSpacing'):
        settings['pixelSpacing'] = cfg.get('pixelSpacing')
    if cfg.get('resampledPixelSpacing'):
        settings['resampledPixelSpacing'] = cfg.get('resampledPixelSpacing')
    # force2D
    if cfg.get('force2D') is not None:
        settings['force2D'] = cfg.get('force2D')
    # binWidth
    if cfg.get('binWidth'):
        settings['binWidth'] = cfg.get('binWidth')
    # padDistance
    if cfg.get('padDistance'):
        settings['padDistance'] = cfg.get('padDistance')
    # preCrop
    if cfg.get('preCrop') is not None:
        settings['preCrop'] = cfg.get('preCrop')

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Configurar tipos de imagen correctamente (Wavelet/LoG/LBP2D/etc.)
    try:
        extractor.disableAllImageTypes()
        img_types = cfg.get('enableImageTypes') or { 'Original': {} }
        for itype, iparams in img_types.items():
            iparams = iparams or {}
            try:
                extractor.enableImageTypeByName(itype, **iparams)
            except Exception:
                # Si un tipo no es soportado (p.ej., LBP2D en versiones antiguas), lo ignoramos
                pass
    except Exception:
        # Si la API difiere, al menos garantizamos Original
        try:
            extractor.enableImageTypeByName('Original')
        except Exception:
            pass

    # Configurar clases de features
    enabled_classes = cfg.get('enableFeatureClasses', []) or []
    extractor.disableAllFeatures()
    for feature_class in enabled_classes:
        try:
            extractor.enableFeatureClassByName(feature_class)
        except Exception:
            pass
    if cfg.get('force2D', False):
        try:
            extractor.disableFeatureClassByName('shape')
        except Exception:
            pass
    return extractor

def _worker_process_init(rad_cfg: dict, source_folder: str):
    global _RAD_CFG, _EXTRACTOR, _SOURCE_FOLDER
    _RAD_CFG = rad_cfg or {}
    _SOURCE_FOLDER = source_folder or 'unknown'
    # Limitar threads internos de ITK/SimpleITK dentro de cada proceso
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    except Exception:
        pass
    _EXTRACTOR = _build_extractor_from_cfg(_RAD_CFG)

def _worker_thread_init(rad_cfg: dict, source_folder: str):
    # Versión ligera para hilos (ThreadPool)
    global _RAD_CFG, _EXTRACTOR, _SOURCE_FOLDER
    _RAD_CFG = rad_cfg or {}
    _SOURCE_FOLDER = source_folder or 'unknown'
    _EXTRACTOR = _build_extractor_from_cfg(_RAD_CFG)

def _process_image_worker(path_dict: dict):
    """Worker top-level compatible con ProcessPool (Windows spawn).
    Espera un dict con keys: index, original_path, mask_path, density_path (todos serializables).
    """
    try:
        # Cargar datos desde disco
        image = np.load(path_dict['original_path'])
        mask = np.load(path_dict['mask_path'])
        density_obj = np.load(path_dict['density_path'], allow_pickle=True)

        # Extraer density numérica
        density_value = None
        try:
            density_element = density_obj.item() if hasattr(density_obj, 'item') else density_obj
            if hasattr(density_element, 'value'):
                density_value = int(float(density_element.value))
        except Exception:
            density_value = None

        # Validaciones y preprocesamiento
        if image is None or mask is None:
            return None
        if image.shape != mask.shape:
            return None
        image = np.asarray(image)
        mask = np.asarray(mask)
        if image.max() > 255 or image.min() < 0:
            image = np.clip(image, 0, 255)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        if np.sum(mask) == 0:
            return None

        # Convertir a SimpleITK
        image_sitk = sitk.GetImageFromArray(image.astype(np.float64))
        mask_sitk = sitk.GetImageFromArray(mask)

        # Crear extractor si no existe (caché por worker)
        global _EXTRACTOR
        if _EXTRACTOR is None:
            _EXTRACTOR = _build_extractor_from_cfg(_RAD_CFG or {})

        # Ejecutar extracción
        result = _EXTRACTOR.execute(image_sitk, mask_sitk)
        features = {}
        for key, value in result.items():
            if not str(key).startswith('diagnostics'):
                try:
                    features[key] = float(value)
                except Exception:
                    pass

        # Agregar metadatos
        feats_meta = features
        feats_meta['image_index'] = int(path_dict.get('index', -1))
        feats_meta['original_path'] = str(path_dict.get('original_path'))
        feats_meta['mask_path'] = str(path_dict.get('mask_path'))
        feats_meta['source_folder'] = _SOURCE_FOLDER
        feats_meta['Tipo_Tejido'] = density_value if density_value is not None else None

        # Liberar objetos locales
        del image_sitk
        del mask_sitk
        del image
        del mask
        del density_obj

        return feats_meta

    except Exception:
        return None

class RadiomicsMLTrainer:
    
    def __init__(self, results_folder="results"):
        self.results_folder = Path(results_folder)
        # Intentar cargar configuración externa de radiomics
        self.radiomics_config = {}
        self.rad_cfg = {}
        try:
            if YAML_AVAILABLE:
                cfg_path = Path('configs/radiomics_config.yaml')
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
    
    def get_processed_file_paths(self, input_folder=None):
        """Obtener SOLO las rutas de archivos, sin cargar imágenes en memoria
        
        Args:
            input_folder: Ruta específica a la carpeta de datos procesados (opcional).
                         Si es None, busca automáticamente la carpeta más reciente en data/processed/
        """
        
        if input_folder:
            # Usar carpeta especificada por el usuario
            latest_folder = Path(input_folder)
            if not latest_folder.exists():
                print(f"Error: Carpeta especificada no encontrada: {latest_folder}")
                return None
            print(f"Usando carpeta especificada: {latest_folder.name}")
        else:
            # Verificar si existe la carpeta base data/processed
            data_processed_path = Path("data/processed")
            if not data_processed_path.exists():
                print(f"Error: Carpeta base no encontrada: {data_processed_path}")
                return None
            
            # Buscar la carpeta con timestamp más reciente
            timestamp_folders = [f for f in data_processed_path.iterdir() if f.is_dir()]
            if not timestamp_folders:
                print("Error: No se encontraron carpetas con timestamp")
                return None
            
            # Ordenar por nombre (timestamp) y tomar la más reciente
            latest_folder = sorted(timestamp_folders, key=lambda x: x.name)[-1]
            print(f"Usando carpeta más reciente: {latest_folder.name}")
        
        self.latest_folder = latest_folder.name
        self.latest_folder_path = latest_folder

        # Verificar estructura de subcarpetas (originals o cleaned como alternativa)
        originals_path = latest_folder / "originals"
        if not originals_path.exists():
            # Intentar usar 'cleaned' como alternativa
            cleaned_path = latest_folder / "cleaned"
            if cleaned_path.exists():
                # Verificar si hay un subdirectorio 'cleaned/cleaned/'
                cleaned_subdir = cleaned_path / "cleaned"
                if cleaned_subdir.exists() and list(cleaned_subdir.glob("*.npy")):
                    originals_path = cleaned_subdir
                    print(f"Usando 'cleaned/cleaned/' para imágenes")
                else:
                    originals_path = cleaned_path
                    print(f"Usando 'cleaned/' para imágenes")
            else:
                print(f"Error: No se encontró 'originals/' ni 'cleaned/' en {latest_folder}")
                return None
        
        masks_path = latest_folder / "masks"
        densities_path = latest_folder / "densities"
        
        if not masks_path.exists() or not densities_path.exists():
            print(f"Error: Faltan subcarpetas en {latest_folder}")
            print(f"  masks/: {'OK' if masks_path.exists() else 'FALTA'}")
            print(f"  densities/: {'OK' if densities_path.exists() else 'FALTA'}")
            return None
        
        # Cargar metadatos si existen
        metadata_file = latest_folder / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            total_images = metadata.get('total_images', 0)
            print(f"Metadatos encontrados: {total_images} imágenes")
        
        # Obtener SOLO las rutas de archivos (no cargar contenido)
        # Intentar primero con patrón 'original_*.npy', luego 'cleaned_*.npy'
        original_files = sorted(list(originals_path.glob("original_*.npy")))
        if not original_files:
            # Si no hay archivos 'original_*.npy', buscar 'cleaned_*.npy'
            original_files = sorted(list(originals_path.glob("cleaned_*.npy")))
            if original_files:
                print(f"Usando archivos 'cleaned_*.npy' como imágenes originales")
        
        mask_files = sorted(list(masks_path.glob("mask_*.npy")))
        density_files = sorted(list(densities_path.glob("density_*.npy")))

        if len(original_files) != len(mask_files) or len(original_files) != len(density_files):
            print(f"Advertencia: Número diferente de archivos - imágenes ({len(original_files)}), máscaras ({len(mask_files)}), densidades ({len(density_files)})")

        # Crear lista de tuplas (original_path, mask_path, density_path, index)
        file_paths = []
        for i, (original_file, mask_file, density_file) in enumerate(zip(original_files, mask_files, density_files)):
            file_paths.append({
                'index': i,
                'original_path': original_file,
                'mask_path': mask_file,
                'density_path': density_file
            })
        
        print(f"Encontrados {len(file_paths)} conjuntos de archivos")
        print(f"Uso de memoria: MÍNIMO (solo rutas, no imágenes cargadas)")
        
        return file_paths
    
    def load_single_image(self, path_dict):
        """Cargar UNA SOLA imagen desde sus paths (optimizado para paralelización)"""
        try:
            # Cargar imagen, máscara y densidad
            image = np.load(path_dict['original_path'])
            mask = np.load(path_dict['mask_path'])
            density_obj = np.load(path_dict['density_path'], allow_pickle=True)
            
            # Extraer el valor numérico de density correctamente
            # density_obj es un numpy array que contiene un DataElement de pydicom
            try:
                # Obtener el item del array (DataElement)
                if hasattr(density_obj, 'item'):
                    density_element = density_obj.item()
                else:
                    density_element = density_obj
                
                # Extraer el valor numérico del DataElement y convertir a entero
                if hasattr(density_element, 'value'):
                    density_value = int(float(density_element.value))
                else:
                    density_value = None
            except Exception as e:
                print(f"  ⚠️  Error extrayendo density {path_dict['index']}: {e}")
                density_value = None
            
            return {
                'index': path_dict['index'],
                'image': image,
                'mask': mask,
                'density_value': density_value,  # Valor numérico directo
                'original_path': str(path_dict['original_path']),
                'mask_path': str(path_dict['mask_path']),
                'density_path': str(path_dict['density_path'])
            }
        except Exception as e:
            print(f"  ⚠️  Error cargando imagen {path_dict['index']}: {e}")
            return None
    
    def extract_radiomics_features(self, image, mask):
        """Extraer características radiómicas usando PyRadiomics"""
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
            image = np.asarray(image)
            mask = np.asarray(mask)
            
            # Normalizar imagen si es necesario (0-255 rango típico)
            if image.max() > 255 or image.min() < 0:
                image = np.clip(image, 0, 255)
            
            # Asegurar que la máscara sea binaria (0 y 1 solamente)
            mask = np.where(mask > 0, 1, 0)
            
            # Verificar que la máscara tenga al menos algunos píxeles activos
            if np.sum(mask) == 0:
                print("Advertencia: Máscara vacía (todos los píxeles son 0)")
                return None
            # Configurar extractor de características: preferir configuración externa
            settings = {}
            cfg = getattr(self, 'rad_cfg', {}) or {}

            # enableImageTypes
            if cfg.get('enableImageTypes'):
                settings['enableImageTypes'] = cfg.get('enableImageTypes')
            # normalize
            if cfg.get('normalize') is not None:
                settings['normalize'] = cfg.get('normalize')
            if cfg.get('normalizeScale'):
                settings['normalizeScale'] = cfg.get('normalizeScale')
            # pixel spacing
            if cfg.get('pixelSpacing'):
                settings['pixelSpacing'] = cfg.get('pixelSpacing')
            if cfg.get('resampledPixelSpacing'):
                settings['resampledPixelSpacing'] = cfg.get('resampledPixelSpacing')
            # force2D
            if cfg.get('force2D') is not None:
                settings['force2D'] = cfg.get('force2D')
            # binWidth para discretización
            if cfg.get('binWidth'):
                settings['binWidth'] = cfg.get('binWidth')
            # padDistance para filtros LoG
            if cfg.get('padDistance'):
                settings['padDistance'] = cfg.get('padDistance')
            # preCrop para acelerar
            if cfg.get('preCrop') is not None:
                settings['preCrop'] = cfg.get('preCrop')

            # Crear el extractor
            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            
            # Configurar clases de features (habilitar/deshabilitar específicamente)
            enabled_classes = cfg.get('enableFeatureClasses', [])
            
            # Desactivar TODAS las clases primero
            extractor.disableAllFeatures()
            
            # Activar solo las clases especificadas
            for feature_class in enabled_classes:
                extractor.enableFeatureClassByName(feature_class)
            
            # Si force2D está activado, asegurar que 'shape' (3D) esté desactivado
            if cfg.get('force2D', False):
                try:
                    extractor.disableFeatureClassByName('shape')
                except Exception:
                    pass  # 'shape' ya estaba desactivado
            
            # Convertir a SimpleITK
            image_sitk = sitk.GetImageFromArray(image.astype(np.float64))
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
            
            # Extraer características
            result = extractor.execute(image_sitk, mask_sitk)
            
            # Convertir a diccionario de características
            features = {}
            for key, value in result.items():
                if not key.startswith('diagnostics'):
                    features[key] = float(value)
            
            # Liberar objetos SimpleITK
            del image_sitk
            del mask_sitk
            del result
            
            return features
            
        except Exception as e:
            return None
    
    def process_radiomics_features(self, all_features):
        """Procesar características radiómicas y eliminar columnas diagnósticas"""
        if not all_features:
            return None
        
        # Convertir a DataFrame
        features_df = pd.DataFrame(all_features)

        cfg_cols = (getattr(self, 'rad_cfg', {}) or {}).get('columns_to_eliminate')
        columns_to_eliminate = cfg_cols
        
        # Eliminar columnas que existen en el DataFrame, PERO NO eliminar 'Tipo_Tejido'
        columns_to_remove = [col for col in columns_to_eliminate 
                            if col in features_df.columns and col != 'Tipo_Tejido']
        # Crear DataFrame limpio
        features_df_clean = features_df.drop(columns=columns_to_remove)
        
        print(f"Características radiómicas procesadas: {features_df_clean.shape}")
        return features_df_clean
    
    def append_to_csv(self, features_list, csv_path, is_first_batch=False):
        """Agregar features al CSV de forma incremental"""
        if not features_list:
            print("  ⚠️  No hay features para guardar en este lote")
            return
        
        try:
            df = pd.DataFrame(features_list)
            
            if is_first_batch:
                # Primera vez: crear archivo con headers
                df.to_csv(csv_path, mode='w', index=False, header=True)
                print(f"  ✅ Creado CSV con headers: {csv_path}")
            else:
                # Lotes siguientes: append sin headers
                df.to_csv(csv_path, mode='a', index=False, header=False)
                print(f"  ✅ Agregadas {len(df)} filas al CSV")
                
        except Exception as e:
            print(f"  ❌ Error guardando CSV: {e}")
    
    def clean_csv_columns(self, csv_path):
        """Aplicar limpieza de columnas al CSV final completo"""
        try:
            print(f"\n🧹 Limpiando columnas del CSV final...")
            
            # Leer CSV completo
            df = pd.read_csv(csv_path)
            print(f"  CSV original: {df.shape}")
            
            # Obtener columnas a eliminar desde config
            cfg_cols = (getattr(self, 'rad_cfg', {}) or {}).get('columns_to_eliminate', [])
            
            if cfg_cols:
                # Eliminar columnas que existen, EXCEPTO 'Tipo_Tejido'
                columns_to_remove = [col for col in cfg_cols 
                                    if col in df.columns and col != 'Tipo_Tejido']
                
                if columns_to_remove:
                    df_clean = df.drop(columns=columns_to_remove)
                    print(f"  Columnas eliminadas: {len(columns_to_remove)}")
                    print(f"  CSV limpio: {df_clean.shape}")
                    
                    # Sobrescribir el CSV con la versión limpia
                    df_clean.to_csv(csv_path, index=False)
                    print(f"  ✅ CSV actualizado con columnas limpias")
                else:
                    print(f"  ℹ️  No hay columnas para eliminar")
            else:
                print(f"  ℹ️  No hay configuración de columnas a eliminar")
                
        except Exception as e:
            print(f"  ❌ Error limpiando columnas: {e}")

    def process_batch_incremental(self, file_paths, batch_size=100, max_workers=16, append_to_existing=False, backend: str = 'threads'):
        """Procesar en lotes PARALELOS y guardar incrementalmente en CSV
        
        Args:
            file_paths: Lista de rutas de archivos
            batch_size: Tamaño del lote
            max_workers: Número de workers (procesos o hilos)
            append_to_existing: Continuar CSV existente
            backend: 'threads' para ThreadPoolExecutor o 'processes' para ProcessPoolExecutor
        """
        backend = (backend or 'threads').lower()
        if backend not in ('threads', 'processes'):
            backend = 'threads'
        
        total = len(file_paths)
        print(f"\n📦 Procesando {total} imágenes en lotes de {batch_size}")
        if backend == 'threads':
            print(f"🧵 Paralelización: {max_workers} workers (hilos)")
        else:
            print(f"🧪 Paralelización: {max_workers} workers (procesos)")
        print(f"�💾 Guardado incremental: ACTIVADO (CSV se actualiza cada lote)\n")
        
        # Preparar ruta del CSV
        features_folder = Path("data/features")
        features_folder.mkdir(parents=True, exist_ok=True)
        
        source_ts = getattr(self, 'latest_folder', 'unknown')

        # Si se pide append_to_existing, buscar el último CSV en data/features que contenga el source_ts
        csv_path = None
        if append_to_existing and source_ts != 'unknown':
            candidates = sorted(list(features_folder.glob(f"*_{source_ts}.csv")))
            if candidates:
                csv_path = candidates[-1]
                print(f"🔁 Se encontró CSV existente para source '{source_ts}': {csv_path} — se continuará agregando")

        if csv_path is None:
            now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_name = f"{now_ts}_{source_ts}.csv"
            csv_path = features_folder / csv_name
        
        print(f"📄 Archivo CSV: {csv_path}")
        print(f"📊 Estructura: Cada lote se GUARDA inmediatamente al terminar\n")
        
        # Si se detectó un CSV existente y se solicitó append, intentar determinar
        # cuántas imágenes ya fueron procesadas para reanudar desde allí.
        start_from = 0
        if append_to_existing and csv_path.exists():
            try:
                # Preferir leer columna image_index si está presente
                existing_df = pd.read_csv(csv_path, usecols=['image_index'])
                if not existing_df.empty:
                    max_index = int(existing_df['image_index'].dropna().astype(int).max())
                    start_from = max_index + 1
                else:
                    start_from = 0
                print(f"🔁 Reanudando desde índice {start_from} basado en '{csv_path.name}' (image_index)")
            except Exception:
                try:
                    # Fallback: contar filas (restar header)
                    with open(csv_path, 'r', encoding='utf-8') as fh:
                        lines = sum(1 for _ in fh)
                    start_from = max(0, lines - 1)
                    print(f"🔁 Reanudando desde índice {start_from} (fallback por conteo de filas)")
                except Exception:
                    start_from = 0

        # Si start_from > 0, saltar las primeras rutas ya procesadas
        if start_from > 0:
            if start_from >= total:
                print(f"ℹ️  Todas las {total} imágenes ya parecen estar procesadas según el CSV. No hay nada que hacer.")
                return csv_path
            print(f"🔀 Saltando {start_from} rutas ya procesadas; procesaremos desde {start_from} hasta {total-1}")
            file_paths = file_paths[start_from:]
            total = len(file_paths)

        # Preparar parámetros comunes para procesos (evitar pasar objetos no serializables)
        source_folder = getattr(self, 'latest_folder', 'unknown')
        
        # Procesar por lotes con paralelización REAL
        total_extracted = 0
        
        # Determinar si estamos en modo append (CSV ya existe y se solicitó append)
        append_mode = append_to_existing and csv_path.exists()

        for batch_num, start in enumerate(range(0, total, batch_size)):
            end = min(start + batch_size, total)
            batch_paths = file_paths[start:end]
            
            print(f"{'='*70}")
            print(f"🔄 LOTE {batch_num + 1}/{(total + batch_size - 1) // batch_size}")
            print(f"📸 Procesando imágenes {start} → {end-1} ({len(batch_paths)} imágenes)")
            print(f"{'='*70}")
            
            # Procesar TODAS las imágenes del lote en PARALELO (ThreadPool real)
            batch_features = []
            
            import time
            start_time = time.time()
            
            # Verifica que backend usar
            if backend == 'threads':
                # Worker local (hilos) reutiliza el worker global para consistencia
                # Inicializar contexto de hilo (ligero)
                _worker_thread_init(self.rad_cfg or {}, source_folder)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # futures es una lista de resultados en el mismo orden que batch_paths
                    futures = list(executor.map(_process_image_worker, batch_paths))
                    # Recopilar resultados en orden e informar progreso
                    for idx, result in enumerate(futures):
                        # Agregar solo resultados válidos
                        if result:
                            batch_features.append(result)
                        # Informar progreso cada 5 imágenes o al final
                        if (idx + 1) % 5 == 0 or (idx + 1) == len(batch_paths):
                            global_idx = start + idx + 1
                            elapsed = time.time() - start_time
                            speed = (idx + 1) / elapsed if elapsed > 0 else 0
                            print(f"  ⚡ {global_idx}/{total} procesadas | Lote: {len(batch_features)}/{idx+1} exitosas | Velocidad: {speed:.1f} img/s")
            # Si no es threads, usar procesos
            else:
                # Procesos: convertir Paths a str para evitar problemas de pickle en Windows
                send_batch = []
                # Convertir cada Path a str
                for pdict in batch_paths:
                    send_batch.append({
                        'index': int(pdict['index']),
                        'original_path': str(pdict['original_path']),
                        'mask_path': str(pdict['mask_path']),
                        'density_path': str(pdict['density_path'])
                    })
                # chunksize para reducir overhead
                chunksize = max(1, len(send_batch) // (max_workers * 4) or 1)
                with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_process_init, initargs=(self.rad_cfg or {}, source_folder)) as executor:
                    # futures es una lista de resultados en el mismo orden que send_batch
                    futures = list(executor.map(_process_image_worker, send_batch, chunksize=chunksize))
                    for idx, result in enumerate(futures):
                        if result:
                            batch_features.append(result)
                        if (idx + 1) % 5 == 0 or (idx + 1) == len(send_batch):
                            global_idx = start + idx + 1
                            elapsed = time.time() - start_time
                            speed = (idx + 1) / elapsed if elapsed > 0 else 0
                            print(f"  ⚡ {global_idx}/{total} procesadas | Lote: {len(batch_features)}/{idx+1} exitosas | Velocidad: {speed:.1f} img/s")
            
            elapsed_batch = time.time() - start_time
            
            # Guardar lote en CSV INMEDIATAMENTE
            print(f"\n💾 Guardando lote {batch_num + 1} en CSV...")
            # Si estamos en append_mode no debemos crear/escribir el header ni truncar el CSV
            is_first = (batch_num == 0 and not append_mode)
            self.append_to_csv(batch_features, csv_path, is_first_batch=is_first)
            
            total_extracted += len(batch_features)
            
            # Liberar memoria del lote AGRESIVAMENTE
            del batch_features
            del batch_paths
            del futures
            gc.collect()  # Forzar recolección de basura
            
            print(f"🗑️  Memoria liberada (gc.collect())")
            print(f"✅ Lote {batch_num + 1} guardado: {total_extracted}/{total} totales | Tiempo lote: {elapsed_batch:.1f}s\n")
        
        # Aplicar limpieza de columnas al CSV final
        if total_extracted > 0:
            self.clean_csv_columns(csv_path)
        
        print(f"\n{'='*70}")
        print(f"🎉 PROCESAMIENTO COMPLETO")
        print(f"{'='*70}")
        print(f"✅ Total extraídas: {total_extracted}/{total}")
        print(f"📄 CSV final: {csv_path}")
        print(f"📊 El CSV se guardó incrementalmente en cada lote")
        print(f"{'='*70}\n")
        
        return csv_path

    def run_complete_pipeline(self, batch_size=100, max_workers=16, append_to_existing=False, input_folder=None, backend: str = 'threads', max_images: Optional[int] = None):
        """Pipeline completo de extracción de características con procesamiento incremental PARALELO
        
        Args:
            batch_size: Tamaño del lote para procesamiento
            max_workers: Número de workers paralelos
            append_to_existing: Si es True, continúa agregando al CSV existente
            input_folder: Ruta específica a carpeta con datos procesados (opcional)
            backend: 'threads' o 'processes'
        """
        try:
            print("\n" + "="*70)
            print("=== PIPELINE DE EXTRACCIÓN DE CARACTERÍSTICAS RADIÓMICAS ===")
            print("="*70 + "\n")
            
            # 1. Obtener rutas de archivos (NO cargar imágenes todavía)
            print("📂 Paso 1: Obteniendo rutas de archivos...")
            file_paths = self.get_processed_file_paths(input_folder=input_folder)

            if file_paths is None or len(file_paths) == 0:
                print("❌ Error: No se pudieron obtener las rutas de archivos")
                return None
            
            print(f"✅ Encontradas {len(file_paths)} imágenes para procesar\n")

            # 1.1 Limitar cantidad de imágenes si se solicita (pruebas rápidas)
            if max_images is not None and max_images > 0:
                prev_total = len(file_paths)
                file_paths = file_paths[:max_images]
                print(f"🔬 Modo prueba: tomando solo {len(file_paths)} de {prev_total} imágenes\n")
            
            # 2. Procesar en lotes incrementales CON PARALELIZACIÓN
            print(f"🚀 Paso 2: Procesamiento incremental PARALELO")
            print(f"   • Lotes de {batch_size} imágenes")
            if (backend or 'threads') == 'processes':
                print(f"   • {max_workers} workers paralelos (Multiprocessing)")
            else:
                print(f"   • {max_workers} workers paralelos (Multithreading)")
            print(f"   • Guardado automático después de cada lote\n")
            # Ejecutar procesamiento incremental
            csv_path = self.process_batch_incremental(
                file_paths, 
                batch_size=batch_size,
                max_workers=max_workers,
                append_to_existing=append_to_existing,
                backend=backend
            )
            
            # 3. Finalización
            if csv_path:
                print("="*70)
                print("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
                print("="*70)
                return csv_path
            else:
                print("❌ Error: No se pudo completar el pipeline")
                return None
            
        except Exception as e:
            print(f"\n❌ Error en pipeline completo: {e}")
            import traceback
            traceback.print_exc()
            return None
    
def main():
    """Función principal para ejecución independiente"""

    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Extracción de Características Radiómicas con Procesamiento Incremental PARALELO')
    parser.add_argument('--results-folder', '-r', default='./data/features', 
                       help='Ruta de la carpeta de resultados')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                       help='Tamaño del lote para procesamiento incremental (default: 100)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Número de workers paralelos (default: número de CPUs * 2 para I/O bound)')
    parser.add_argument('--backend', choices=['auto','threads','processes'], default='auto',
                        help="Backend de paralelización: 'threads', 'processes' o 'auto' (Windows: processes; Linux/macOS: threads)")
    parser.add_argument('--append-last-csv', action='store_true', default=False,
                        help='Si existe un CSV previo para el mismo source timestamp, anexar en lugar de crear uno nuevo')
    parser.add_argument('--input-folder', '-i', type=str, default=None,
                        help='Carpeta específica con datos procesados (ej: data/processed/20251022_012338). Si no se especifica, usa la carpeta más reciente')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Limitar el número de imágenes a procesar (útil para pruebas rápidas, p.ej., 10)')
    
    args = parser.parse_args()
    
    # Selección de backend por plataforma si 'auto'
    if args.backend == 'auto':
        args.backend = 'processes' if os.name == 'nt' else 'threads'

    # Determinar número de workers
    if args.workers is None:
        if args.backend == 'processes':
            args.workers = max(1, mp.cpu_count())
        else:
            args.workers = max(1, mp.cpu_count() * 2)  # I/O bound: más hilos que CPUs

    # Evitar sobre-subscription de BLAS/OpenMP cuando se usan múltiples procesos
    if args.backend == 'processes':
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    
    # Crear y ejecutar pipeline
    ml_trainer = RadiomicsMLTrainer(args.results_folder)
    
    print(f"\n{'='*70}")
    print(f"🚀 EXTRACCIÓN DE CARACTERÍSTICAS RADIÓMICAS - MODO PARALELO")
    print(f"{'='*70}")
    print(f"📦 Tamaño de lote: {args.batch_size} imágenes")

    # Detalles de configuración
    if args.backend == 'processes':
        print(f"🚀 Workers: {args.workers} (Multiprocessing para CPU-bound)")
    else:
        print(f"🚀 Workers: {args.workers} (Multithreading optimizado para I/O)")
    print(f"💾 Guardado incremental: ACTIVADO (CSV se actualiza cada lote)")
    print(f"🧠 Optimización de memoria: ACTIVADA (gc.collect() cada lote)")
    if args.backend == 'processes':
        print(f"⚡ Paralelización: ProcessPoolExecutor (óptimo para CPU-bound, evita GIL)")
    else:
        print(f"⚡ Paralelización: ThreadPoolExecutor (óptimo para PyRadiomics + I/O)")
    if args.input_folder:
        print(f"📂 Carpeta de entrada: {args.input_folder}")
    else:
        print(f"📂 Carpeta de entrada: AUTO (más reciente en data/processed/)")
    print(f"{'='*70}\n")
    
    results = ml_trainer.run_complete_pipeline(
        batch_size=args.batch_size,
        max_workers=args.workers,
        append_to_existing=args.append_last_csv,
        input_folder=args.input_folder,
        backend=args.backend,
        max_images=args.max_images
    )


if __name__ == "__main__":
    exit(main())
