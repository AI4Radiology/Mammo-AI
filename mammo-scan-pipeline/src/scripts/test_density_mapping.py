"""
Script de prueba para verificar el procesamiento DICOM con mapeo de densidad
Procesa solo 3 imágenes para validar el funcionamiento
"""

import sys
sys.path.append('.')

from src.scripts.dicom_image_processing import DICOMImageProcessor
import pydicom
from pathlib import Path as _Path

def test_processing():
    """Prueba el procesamiento DICOM y muestra metadata de la primera imagen encontrada.
    """

    print("\n" + "="*80)
    print("PRUEBA DE PROCESAMIENTO DICOM CON MAPEO DE DENSIDAD")
    print("="*80 + "\n")
    
    # Buscar imagenes DICOM en la carpeta del base_path de forma recursiva
    project_root = _Path(__file__).parent.parent.parent # src/scripts/ -> src/ -> proyecto_raíz/
    base_path = project_root / "cc"
    print(f"Buscando una imagen DICOM en: {base_path} (recursivo)")
    all_dicom = list(_Path(base_path).rglob("*.dcm"))
    if not all_dicom:
        print(f"❌ No se encontraron archivos DICOM en la carpeta {base_path}")
        return

    # Toma la primera o la imagen a selección de las encontradas para la prueba
    first_dicom = all_dicom[0]
    print(f"✅ Imagen encontrada: {first_dicom}\n")

    # Mostrar metadata completa de la imagen para identificar que el tag de densidad este correcto
    try:
        print(f"  Cargando DICOM: {first_dicom}")
        # Leer el archivo DICOM
        dic = pydicom.dcmread(str(first_dicom), force=True)
        print("\n" + "="*80)
        print("METADATA COMPLETA DE LA PRIMERA IMAGEN EN cc")
        print("="*80 + "\n")
        for i, elem in enumerate(dic, 1):
            # Intentar mostrar cada elemento de metadata
            try:
                tag = elem.tag
                name = elem.name
                vr = elem.VR
                val = elem.value
                val_str = str(val)
                # Mostrar valor truncado si es muy largo
                if isinstance(val, bytes):
                    val_str = f"<bytes len={len(val)}>"
                if len(val_str) > 200:
                    val_str = val_str[:197] + '...'
                print(elem)
            except Exception as e:
                print(f"[{i:03d}] Error leyendo elemento: {e}")

        print("\n" + "="*80)
    except Exception as e:
        print(f"❌ Error al leer DICOM: {e}")
        return
    print("="*80 + "\n")

if __name__ == "__main__":
    test_processing()
