"""
DICOM ETL Workflow with Watchdog
=================================

Sistema de procesamiento automático de archivos DICOM que:
1. Monitorea carpeta de entrada para archivos .dcm, .dicom, carpetas y zips
2. Procesa imágenes DICOM y extrae máscaras
3. Extrae características radiómicas
4. Almacena features en CSV para reentrenamiento
5. Clasifica con modelos ML
6. Genera mensajes HL7/ORM con resultados

Carpetas del sistema:
- input/: Recepción de imágenes DICOM
- output/: Mensajes HL7 generados
- data/: CSV con features extraídos
- logs/: Registros de procesamiento
- error/: Archivos que fallaron en procesamiento
"""

import os
import sys
import time
import json
import logging
import shutil
import zipfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from queue import Queue
from threading import Thread, Lock

import pydicom
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Importar servicios locales
from services.dicom_image_processing import DICOMImageProcessor
from services.radiomic_extraction import RadiomicsMLTrainer
from services.classify import classify_dataframe
import cv2


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración centralizada del sistema"""
    
    # Carpetas del sistema
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "input"
    OUTPUT_DIR = BASE_DIR / "output"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    ERROR_DIR = BASE_DIR / "error"
    
    # Archivo CSV para almacenar features
    FEATURES_CSV = DATA_DIR / "radiomics_features.csv"
    
    # Configuración de logging
    LOG_FILE = LOGS_DIR / f"dicom_etl_{datetime.now().strftime('%Y%m%d')}.log"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = logging.INFO
    
    # Extensiones soportadas
    DICOM_EXTENSIONS = ['.dcm', '.dicom']
    ZIP_EXTENSIONS = ['.zip']
    
    # Configuración de procesamiento
    PROCESS_QUEUE_TIMEOUT = 2  # segundos
    MAX_RETRIES = 3
    
    @classmethod
    def setup_directories(cls):
        """Crear todas las carpetas necesarias"""
        for directory in [cls.INPUT_DIR, cls.OUTPUT_DIR, cls.DATA_DIR, 
                         cls.LOGS_DIR, cls.ERROR_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_logging(cls):
        """Configurar sistema de logging"""
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configurar logger raíz
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format=cls.LOG_FORMAT,
            handlers=[
                logging.FileHandler(cls.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )


# ============================================================================
# GENERADOR DE MENSAJES HL7
# ============================================================================

class HL7MessageGenerator:
    """Generador de mensajes HL7 ORM para resultados de clasificación"""
    
    @staticmethod
    def generate_orm_message(
        dicom_metadata: Dict[str, Any],
        classification_result: Dict[str, Any],
        radiomics_features: Dict[str, float]
    ) -> str:
        """
        Generar mensaje HL7 ORM con resultados de clasificación
        
        Args:
            dicom_metadata: Metadatos extraídos del DICOM
            classification_result: Resultado de la clasificación
            radiomics_features: Features radiómicos extraídos
            
        Returns:
            str: Mensaje HL7 ORM formateado
        """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Extraer información del DICOM
        patient_id = dicom_metadata.get('PatientID', 'UNKNOWN')
        patient_name = dicom_metadata.get('PatientName', 'UNKNOWN')
        study_instance_uid = dicom_metadata.get('StudyInstanceUID', 'UNKNOWN')
        accession_number = dicom_metadata.get('AccessionNumber', 'UNKNOWN')
        modality = dicom_metadata.get('Modality', 'MG')
        study_date = dicom_metadata.get('StudyDate', timestamp[:8])
        study_time = dicom_metadata.get('StudyTime', timestamp[8:])
        
        # Extraer clasificaciones
        binary_pred = classification_result['binary']['prediction']
        binary_prob = classification_result['binary']['probabilities'].get(binary_pred, 0.0)
        multiclass_pred = classification_result['multiclass']['prediction']
        multiclass_prob = classification_result['multiclass']['probabilities'].get(multiclass_pred, 0.0)
        
        # Construir mensaje HL7 ORM
        message_lines = []
        
        # MSH - Message Header
        message_lines.append(
            f"MSH|^~\\&|MAMMO_SCAN_ETL|RADIOLOGY|RIS|HOSPITAL|{timestamp}||ORM^O01|"
            f"{timestamp}_{patient_id}|P|2.5|||AL|AL|USA"
        )
        
        # PID - Patient Identification
        message_lines.append(
            f"PID|1||{patient_id}||{patient_name}||||||||||||||{accession_number}"
        )
        
        # PV1 - Patient Visit
        message_lines.append(
            f"PV1|1|O|RAD||||||||RAD|||||||||{accession_number}|||||||||||||||||||||||||{study_date}"
        )
        
        # ORC - Common Order
        message_lines.append(
            f"ORC|NW|{accession_number}|{study_instance_uid}||CM||^^^^^R||"
            f"{study_date}{study_time}|||||||||||RADIOLOGY"
        )
        
        # OBR - Observation Request
        message_lines.append(
            f"OBR|1|{accession_number}|{study_instance_uid}|{modality}^MAMMOGRAPHY^L|||"
            f"{study_date}{study_time}|||||||||||||||{timestamp}|||F"
        )
        
        # OBX - Observation Result (Clasificación Binaria)
        message_lines.append(
            f"OBX|1|ST|DENSITY_CLASS^Density Classification^L||{binary_pred}||||||F|||"
            f"{timestamp}||MAMMO_SCAN_AI"
        )
        
        # OBX - Observation Result (Probabilidad Binaria)
        message_lines.append(
            f"OBX|2|NM|DENSITY_PROB^Density Probability^L||{binary_prob:.4f}||||||F|||"
            f"{timestamp}||MAMMO_SCAN_AI"
        )
        
        # OBX - Observation Result (Clasificación BI-RADS)
        message_lines.append(
            f"OBX|3|ST|BIRADS_CLASS^BI-RADS Classification^L||{multiclass_pred}||||||F|||"
            f"{timestamp}||MAMMO_SCAN_AI"
        )
        
        # OBX - Observation Result (Probabilidad BI-RADS)
        message_lines.append(
            f"OBX|4|NM|BIRADS_PROB^BI-RADS Probability^L||{multiclass_prob:.4f}||||||F|||"
            f"{timestamp}||MAMMO_SCAN_AI"
        )
        
        # OBX - Observation Result (Número de features radiómicos)
        message_lines.append(
            f"OBX|5|NM|RADIOMICS_COUNT^Radiomics Features Count^L||{len(radiomics_features)}||||||F|||"
            f"{timestamp}||MAMMO_SCAN_AI"
        )
        
        # NTE - Notes and Comments
        message_lines.append(
            f"NTE|1||Automated mammography density and BI-RADS classification"
        )
        message_lines.append(
            f"NTE|2||Binary: {binary_pred} ({binary_prob:.2%}), "
            f"BI-RADS: {multiclass_pred} ({multiclass_prob:.2%})"
        )
        
        # Unir líneas con \r\n (terminadores HL7)
        return '\r\n'.join(message_lines) + '\r\n'
    
    @staticmethod
    def save_hl7_message(message: str, output_path: Path) -> None:
        """Guardar mensaje HL7 en archivo"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(message)


# ============================================================================
# PROCESADOR DE ARCHIVOS DICOM
# ============================================================================

class DICOMProcessor:
    """Procesador de archivos DICOM con flujo ETL completo"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dicom_processor = DICOMImageProcessor(workers=4)
        self.radiomics_trainer = RadiomicsMLTrainer()
        self.hl7_generator = HL7MessageGenerator()
        self.csv_lock = Lock()
        
        self.logger.info("DICOMProcessor inicializado")
    
    def extract_dicom_metadata(self, dicom_file) -> Dict[str, Any]:
        """Extraer metadatos relevantes del archivo DICOM"""
        metadata = {}
        
        # Metadatos del paciente
        metadata['PatientID'] = str(getattr(dicom_file, 'PatientID', 'UNKNOWN'))
        metadata['PatientName'] = str(getattr(dicom_file, 'PatientName', 'UNKNOWN'))
        metadata['PatientBirthDate'] = str(getattr(dicom_file, 'PatientBirthDate', ''))
        metadata['PatientSex'] = str(getattr(dicom_file, 'PatientSex', ''))
        
        # Metadatos del estudio
        metadata['StudyInstanceUID'] = str(getattr(dicom_file, 'StudyInstanceUID', 'UNKNOWN'))
        metadata['StudyDate'] = str(getattr(dicom_file, 'StudyDate', ''))
        metadata['StudyTime'] = str(getattr(dicom_file, 'StudyTime', ''))
        metadata['AccessionNumber'] = str(getattr(dicom_file, 'AccessionNumber', 'UNKNOWN'))
        metadata['StudyDescription'] = str(getattr(dicom_file, 'StudyDescription', ''))
        
        # Metadatos de la serie
        metadata['SeriesInstanceUID'] = str(getattr(dicom_file, 'SeriesInstanceUID', ''))
        metadata['SeriesNumber'] = str(getattr(dicom_file, 'SeriesNumber', ''))
        metadata['Modality'] = str(getattr(dicom_file, 'Modality', 'MG'))
        
        # Metadatos de la imagen
        metadata['SOPInstanceUID'] = str(getattr(dicom_file, 'SOPInstanceUID', ''))
        metadata['InstanceNumber'] = str(getattr(dicom_file, 'InstanceNumber', ''))
        metadata['ViewPosition'] = str(getattr(dicom_file, 'ViewPosition', ''))
        metadata['ImageLaterality'] = str(getattr(dicom_file, 'ImageLaterality', ''))
        
        return metadata
    
    def process_dicom_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Procesar un archivo DICOM completo: carga, segmentación, radiomics, clasificación
        
        Args:
            file_path: Ruta al archivo DICOM
            
        Returns:
            Dict con resultados del procesamiento
        """
        self.logger.info(f"Procesando archivo DICOM: {file_path}")
        
        try:
            # 1. Leer archivo DICOM
            with open(file_path, 'rb') as f:
                dicom_bytes = f.read()
            
            dicom_file, image = self.dicom_processor.read_dicom_bytes(dicom_bytes)
            self.logger.info(f"DICOM leído - Shape: {image.shape}")
            
            # 2. Extraer metadatos
            metadata = self.extract_dicom_metadata(dicom_file)
            self.logger.info(f"Metadatos extraídos - PatientID: {metadata['PatientID']}, "
                           f"StudyUID: {metadata['StudyInstanceUID']}")
            
            # Define image views
            view_result = self.dicom_processor.get_image_view_position(dicom_file)

            if view_result.get('view_position') == 'MLO':
                raise Exception('MLO view not supported - Only CC view is supported')
            
            # Flip image if right laterality (BEFORE any processing)
            if view_result.get('image_laterality') == 'R':
                image = cv2.flip(image, 1)
                self.logger.info("Imagen volteada - Lateralidad Derecha detectada")
            
            # 3. Limpiar imagen y obtener máscara
            clean_result = self.dicom_processor.clean_single_image(image)
            
            if not clean_result.get('success', False):
                raise Exception(f"Error en segmentación: {clean_result.get('error', 'Unknown')}")
            
            cleaned_image = clean_result['cleaned']
            mask = clean_result['mask']
            self.logger.info(f"Segmentación exitosa - Píxeles en máscara: {mask.sum()}")
            
            # 4. Extraer características radiómicas
            radiomics_features = self.radiomics_trainer.extract_radiomics_features(
                cleaned_image, mask
            )
            
            if radiomics_features is None:
                raise Exception("Error extrayendo características radiómicas")
            
            self.logger.info(f"Features radiómicos extraídos: {len(radiomics_features)}")
            
            # 5. Procesar features (eliminar columnas diagnósticas)
            features_df = self.radiomics_trainer.process_radiomics_features(radiomics_features)
            
            if features_df is None:
                raise Exception("Error procesando características radiómicas")
            
            # 6. Preparar payload para clasificación
            payload = {
                'columns': list(features_df.columns),
                'records': [
                    {'característica': col, 'valor': features_df[col].iloc[0]}
                    for col in features_df.columns
                ]
            }
            
            # 7. Clasificar
            classification_result = classify_dataframe(payload)
            self.logger.info(f"Clasificación - Binaria: {classification_result['binary']['prediction']}, "
                           f"BI-RADS: {classification_result['multiclass']['prediction']}")
            
            # 8. Almacenar features en CSV
            self._save_features_to_csv(features_df, metadata, classification_result)
            
            # 9. Generar mensaje HL7
            hl7_message = self.hl7_generator.generate_orm_message(
                metadata, classification_result, radiomics_features
            )
            
            # 10. Guardar mensaje HL7
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            patient_id = metadata['PatientID']
            hl7_filename = f"HL7_ORM_{patient_id}_{timestamp}.hl7"
            hl7_path = Config.OUTPUT_DIR / hl7_filename
            
            self.hl7_generator.save_hl7_message(hl7_message, hl7_path)
            self.logger.info(f"Mensaje HL7 guardado: {hl7_path}")
            
            return {
                'success': True,
                'metadata': metadata,
                'radiomics_count': len(radiomics_features),
                'classification': classification_result,
                'hl7_file': str(hl7_path),
                'original_file': str(file_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error procesando {file_path}: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'original_file': str(file_path)
            }
    
    def _save_features_to_csv(
        self, 
        features_df: pd.DataFrame, 
        metadata: Dict[str, Any],
        classification_result: Dict[str, Any]
    ) -> None:
        """Guardar features radiómicos en CSV para reentrenamiento futuro"""
        
        with self.csv_lock:
            try:
                # Preparar fila con metadatos y features
                row_data = {
                    'timestamp': datetime.now().isoformat(),
                    'patient_id': metadata['PatientID'],
                    'study_uid': metadata['StudyInstanceUID'],
                    'accession_number': metadata['AccessionNumber'],
                    'binary_prediction': classification_result['binary']['prediction'],
                    'multiclass_prediction': classification_result['multiclass']['prediction'],
                }
                
                # Agregar features radiómicos
                for col in features_df.columns:
                    row_data[col] = features_df[col].iloc[0]
                
                # Crear DataFrame con la nueva fila
                new_row_df = pd.DataFrame([row_data])
                
                # Agregar o crear CSV
                if Config.FEATURES_CSV.exists():
                    existing_df = pd.read_csv(Config.FEATURES_CSV)
                    combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                else:
                    combined_df = new_row_df
                
                # Guardar CSV
                combined_df.to_csv(Config.FEATURES_CSV, index=False)
                self.logger.info(f"Features guardados en CSV: {Config.FEATURES_CSV}")
                
            except Exception as e:
                self.logger.error(f"Error guardando features en CSV: {e}")


# ============================================================================
# MANEJADOR DE EVENTOS DE WATCHDOG
# ============================================================================

class DICOMFileHandler(FileSystemEventHandler):
    """Manejador de eventos de sistema de archivos para Watchdog"""
    
    def __init__(self, process_queue: Queue):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.process_queue = process_queue
        self.processing_files = set()
        self.processing_lock = Lock()
    
    def on_created(self, event):
        """Evento cuando se crea un nuevo archivo o carpeta"""
        if event.is_directory:
            self.logger.info(f"Nueva carpeta detectada: {event.src_path}")
            self._handle_directory(Path(event.src_path))
        else:
            self.logger.info(f"Nuevo archivo detectado: {event.src_path}")
            self._handle_file(Path(event.src_path))
    
    def on_modified(self, event):
        """Evento cuando se modifica un archivo (útil para archivos grandes en copia)"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            # Solo procesar si es un archivo completo (no está siendo copiado)
            if self._is_file_complete(file_path):
                self._handle_file(file_path)
    
    def _is_file_complete(self, file_path: Path) -> bool:
        """Verificar si un archivo está completo (no siendo copiado)"""
        try:
            # Esperar un momento para asegurar que el archivo no está siendo escrito
            time.sleep(0.5)
            
            # Intentar abrir el archivo en modo exclusivo
            if file_path.exists():
                size1 = file_path.stat().st_size
                time.sleep(0.5)
                size2 = file_path.stat().st_size
                return size1 == size2 and size1 > 0
            return False
        except Exception:
            return False
    
    def _handle_file(self, file_path: Path) -> None:
        """Manejar un archivo individual"""
        
        # Evitar procesar el mismo archivo múltiples veces
        with self.processing_lock:
            if str(file_path) in self.processing_files:
                return
            self.processing_files.add(str(file_path))
        
        try:
            # Verificar que el archivo está completo
            if not self._is_file_complete(file_path):
                self.logger.warning(f"Archivo no completo, ignorando: {file_path}")
                return
            
            # Procesar según tipo de archivo
            suffix = file_path.suffix.lower()
            
            if suffix in Config.DICOM_EXTENSIONS:
                self.logger.info(f"Archivo DICOM detectado: {file_path}")
                self.process_queue.put(('dicom', file_path))
                
            elif suffix in Config.ZIP_EXTENSIONS:
                self.logger.info(f"Archivo ZIP detectado: {file_path}")
                self.process_queue.put(('zip', file_path))
                
            else:
                self.logger.warning(f"Tipo de archivo no soportado: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error manejando archivo {file_path}: {e}")
        finally:
            # Remover de lista de archivos en procesamiento después de un tiempo
            Thread(target=self._remove_from_processing, args=(str(file_path),)).start()
    
    def _remove_from_processing(self, file_path: str) -> None:
        """Remover archivo de lista de procesamiento después de un delay"""
        time.sleep(10)
        with self.processing_lock:
            self.processing_files.discard(file_path)
    
    def _handle_directory(self, dir_path: Path) -> None:
        """Manejar una carpeta (buscar archivos DICOM dentro)"""
        try:
            # Esperar a que la carpeta esté completa
            time.sleep(2)
            
            # Buscar archivos DICOM en la carpeta
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in Config.DICOM_EXTENSIONS:
                    self._handle_file(file_path)
                    
        except Exception as e:
            self.logger.error(f"Error manejando carpeta {dir_path}: {e}")


# ============================================================================
# PROCESADOR DE COLA
# ============================================================================

class QueueProcessor:
    """Procesador de cola de archivos DICOM"""
    
    def __init__(self, process_queue: Queue):
        self.logger = logging.getLogger(__name__)
        self.process_queue = process_queue
        self.dicom_processor = DICOMProcessor()
        self.running = False
    
    def start(self):
        """Iniciar procesamiento de cola"""
        self.running = True
        self.logger.info("QueueProcessor iniciado")
        
        while self.running:
            try:
                # Obtener item de la cola (timeout para permitir shutdown)
                item = self.process_queue.get(timeout=Config.PROCESS_QUEUE_TIMEOUT)
                
                if item is None:  # Señal de shutdown
                    break
                
                file_type, file_path = item
                self.logger.info(f"Procesando de cola: {file_type} - {file_path}")
                
                if file_type == 'dicom':
                    self._process_dicom(file_path)
                elif file_type == 'zip':
                    self._process_zip(file_path)
                
                self.process_queue.task_done()
                
            except Exception as e:
                if not isinstance(e, Exception) or "Empty" not in str(type(e).__name__):
                    self.logger.error(f"Error en QueueProcessor: {e}")
    
    def stop(self):
        """Detener procesamiento de cola"""
        self.running = False
        self.process_queue.put(None)  # Señal de shutdown
        self.logger.info("QueueProcessor detenido")
    
    def _process_dicom(self, file_path: Path) -> None:
        """Procesar un archivo DICOM"""
        try:
            result = self.dicom_processor.process_dicom_file(file_path)
            
            if result['success']:
                self.logger.info(f"✓ Procesamiento exitoso: {file_path}")
                self.logger.info(f"  - HL7 generado: {result['hl7_file']}")
                self.logger.info(f"  - Clasificación Binaria: {result['classification']['binary']['prediction']}")
                self.logger.info(f"  - Clasificación BI-RADS: {result['classification']['multiclass']['prediction']}")
                
                # Eliminar archivo original después de procesamiento exitoso
                file_path.unlink()
                self.logger.info(f"  - Archivo original eliminado: {file_path}")
                
            else:
                self.logger.error(f"✗ Error procesando: {file_path}")
                self._move_to_error(file_path, result['error'])
                
        except Exception as e:
            self.logger.error(f"Error procesando DICOM {file_path}: {e}")
            self._move_to_error(file_path, str(e))
    
    def _process_zip(self, zip_path: Path) -> None:
        """Procesar un archivo ZIP con DICOMs"""
        temp_extract_dir = None
        
        try:
            self.logger.info(f"Extrayendo ZIP: {zip_path}")
            
            # Crear carpeta temporal para extracción
            temp_extract_dir = Config.INPUT_DIR / f"temp_{zip_path.stem}_{int(time.time())}"
            temp_extract_dir.mkdir(exist_ok=True)
            
            # Extraer ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            
            self.logger.info(f"ZIP extraído en: {temp_extract_dir}")
            
            # Buscar y procesar todos los archivos DICOM
            dicom_files = list(temp_extract_dir.rglob('*'))
            dicom_files = [f for f in dicom_files if f.is_file() and 
                          f.suffix.lower() in Config.DICOM_EXTENSIONS]
            
            self.logger.info(f"Encontrados {len(dicom_files)} archivos DICOM en ZIP")
            
            for dicom_file in dicom_files:
                self._process_dicom(dicom_file)
            
            # Eliminar ZIP original después de procesamiento exitoso
            zip_path.unlink()
            self.logger.info(f"ZIP procesado y eliminado: {zip_path}")
            
        except Exception as e:
            self.logger.error(f"Error procesando ZIP {zip_path}: {e}")
            self._move_to_error(zip_path, str(e))
            
        finally:
            # Limpiar carpeta temporal
            if temp_extract_dir and temp_extract_dir.exists():
                try:
                    shutil.rmtree(temp_extract_dir)
                    self.logger.info(f"Carpeta temporal eliminada: {temp_extract_dir}")
                except Exception as e:
                    self.logger.error(f"Error eliminando carpeta temporal: {e}")
    
    def _move_to_error(self, file_path: Path, error_message: str) -> None:
        """Mover archivo a carpeta de errores"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            error_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            error_path = Config.ERROR_DIR / error_filename
            
            shutil.move(str(file_path), str(error_path))
            
            # Guardar información del error
            error_log_path = error_path.with_suffix('.error.txt')
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Error procesando: {file_path}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error: {error_message}\n")
            
            self.logger.info(f"Archivo movido a error: {error_path}")
            
        except Exception as e:
            self.logger.error(f"Error moviendo archivo a carpeta error: {e}")


# ============================================================================
# APLICACIÓN PRINCIPAL
# ============================================================================

class DICOMWatchdogApp:
    """Aplicación principal de Watchdog para procesamiento DICOM"""
    
    def __init__(self):
        # Configurar sistema
        Config.setup_directories()
        Config.setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*80)
        self.logger.info("DICOM ETL Watchdog iniciando...")
        self.logger.info("="*80)
        
        # Cola de procesamiento
        self.process_queue = Queue()
        
        # Componentes
        self.event_handler = DICOMFileHandler(self.process_queue)
        self.queue_processor = QueueProcessor(self.process_queue)
        self.observer = Observer()
        
        # Thread para procesar cola
        self.processor_thread = None
    
    def start(self):
        """Iniciar la aplicación"""
        try:
            self.logger.info(f"Monitoreando carpeta: {Config.INPUT_DIR}")
            self.logger.info(f"Salida HL7: {Config.OUTPUT_DIR}")
            self.logger.info(f"CSV de features: {Config.FEATURES_CSV}")
            self.logger.info(f"Logs: {Config.LOG_FILE}")
            
            # Iniciar procesador de cola en thread separado
            self.processor_thread = Thread(target=self.queue_processor.start, daemon=True)
            self.processor_thread.start()
            
            # Configurar observer de watchdog
            self.observer.schedule(
                self.event_handler,
                str(Config.INPUT_DIR),
                recursive=True
            )
            
            # Iniciar observer
            self.observer.start()
            self.logger.info("Watchdog iniciado - Esperando archivos DICOM...")
            
            # Procesar archivos existentes en input
            self._process_existing_files()
            
            # Mantener aplicación corriendo
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
                
        except Exception as e:
            self.logger.error(f"Error iniciando aplicación: {e}")
            self.logger.error(traceback.format_exc())
            self.stop()
    
    def stop(self):
        """Detener la aplicación"""
        self.logger.info("Deteniendo aplicación...")
        
        # Detener observer
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        
        # Detener procesador de cola
        self.queue_processor.stop()
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
        
        self.logger.info("Aplicación detenida")
    
    def _process_existing_files(self):
        """Procesar archivos que ya existen en la carpeta de input"""
        self.logger.info("Buscando archivos existentes en input...")
        
        try:
            for file_path in Config.INPUT_DIR.rglob('*'):
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    
                    if suffix in Config.DICOM_EXTENSIONS:
                        self.logger.info(f"Archivo DICOM existente: {file_path}")
                        self.process_queue.put(('dicom', file_path))
                        
                    elif suffix in Config.ZIP_EXTENSIONS:
                        self.logger.info(f"Archivo ZIP existente: {file_path}")
                        self.process_queue.put(('zip', file_path))
                        
        except Exception as e:
            self.logger.error(f"Error procesando archivos existentes: {e}")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

def main():
    """Función principal"""
    app = DICOMWatchdogApp()
    app.start()


if __name__ == "__main__":
    main()
