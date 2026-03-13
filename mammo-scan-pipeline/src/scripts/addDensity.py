"""
Script para agregar tag de densidad (4010,1018) a imágenes DICOM
Lee la densidad del nombre de la carpeta (ejemplo: Calc-Test_P_00038_LEFT_CC_D2)
donde D2 indica Densidad 2, y agrega el tag (4010,1018) FL con valor 2.0
"""

import pydicom
from pathlib import Path
import re
from pydicom.dataset import Dataset


def extract_density_from_folder(file_name):
    """
    Extrae la densidad del nombre de carpeta.
    Ejemplo: 'Calc-Test_P_00038_LEFT_CC_D2' -> 1.0
    
    Mapeo: D1 -> 0.0, D2 -> 1.0, D3 -> 2.0, D4 -> 3.0
    """
    # Buscar patrón D seguido de un número (0-3) al final del nombre
    match = re.search(r'_([ABCD])\.dcm$', file_name)
    if match:
        density_letter = match.group(1)

        density_map = {'A': 0.0, 'B': 1.0, 'C': 2.0, 'D': 3.0}
        return density_map.get(density_letter)
    return None


def add_density_tag_to_dicom(dicom_path, density_value):
    """
    Agrega el tag (4010,1018) FL Density a un archivo DICOM.
    
    Args:
        dicom_path: Path al archivo DICOM
        density_value: Valor float de densidad (1.0, 2.0, 3.0, 4.0)
    
    Returns:
        True si se agregó exitosamente, False en caso contrario
    """
    try:
        # Leer archivo DICOM
        ds = pydicom.dcmread(str(dicom_path), force=True)
        
        # Crear y agregar el tag de densidad
        # Tag (4010,1018) con VR=FL (Floating Point Single)
        density_tag = (0x4010, 0x1018)
        
        # Agregar el elemento al dataset
        ds.add_new(density_tag, 'FL', density_value)
        
        # Guardar el archivo modificado
        ds.save_as(str(dicom_path))
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error procesando {dicom_path.name}: {e}")
        return False


def process_all_folders():
    """
    Procesa todas las carpetas y agrega tag de densidad a cada imagen.
    """
    project_root = Path(__file__).parent.parent.parent # src/scripts/ -> src/ -> proyecto_raíz/
    base_path = project_root / "cc"
    
    if not base_path.exists():
        print(f"❌ La carpeta {base_path} no existe")
        return
    
    print("\n" + "="*80)
    print("AGREGANDO TAG DE DENSIDAD A IMÁGENES DICOM")
    print("="*80)
    print(f"\n📁 Carpeta base: {base_path}\n")
    
    # Obtener todas las subcarpetas en la carpeta base
    folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    if not folders:
        print(f"❌ No se encontraron carpetas en {base_path}")
        folders = [base_path]
    
    print(f"📂 Total de carpetas encontradas: {len(folders)}\n")
    
    total_images = 0
    successful = 0
    failed = 0
    no_density = 0
    
    # Mapeo de densidad para estadísticas
    density_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    density_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print("⏳ Procesando carpetas...\n")
    
    # Procesar cada carpeta
    for idx, folder in enumerate(folders, 1):
        folder_name = folder.name
        
        # Buscar archivos DICOM recursivamente en esta carpeta
        dicom_files = list(folder.rglob("*.dcm"))
        
        if not dicom_files:
            print(f"⚠️  [{idx:3d}/{len(folders)}] {folder_name}: Sin archivos DICOM")
            continue
        
        # Procesar cada imagen DICOM en esta carpeta
        for dicom_file in dicom_files:
            
            fileName = dicom_file.name
            # Extraer densidad del nombre de la carpeta
            density_value = extract_density_from_folder(fileName)
            
            if density_value is None:
                print(f"⚠️  [{idx:3d}/{len(folders)}] {folder_name}: Sin densidad en nombre")
                no_density += 1
                continue

            total_images += 1
            
            # Agregar el tag de densidad al archivo DICOM
            if add_density_tag_to_dicom(dicom_file, density_value):
                successful += 1
                density_counts[density_value] += 1
            else:
                failed += 1
        
        # Resumen por carpeta
        density_letter = density_map.get(density_value, '?')
        print(f"✅ [{idx:3d}/{len(folders)}] {folder_name}: Densidad {density_value} ({density_letter}) -> {len(dicom_files)} imagen(es)")
        
        # Mostrar progreso cada 50 carpetas
        if idx % 50 == 0:
            print(f"\n   📊 Progreso: {idx}/{len(folders)} carpetas procesadas\n")
    
    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL")
    print("="*80)
    print(f"\nTotal de carpetas procesadas: {len(folders)}")
    print(f"Total de imágenes DICOM procesadas: {total_images}")
    print(f"  ✅ Exitosas: {successful}")
    print(f"  ❌ Fallidas: {failed}")
    print(f"  ⚠️  Carpetas sin densidad: {no_density}")
    
    print(f"\n📈 Distribución de densidad:")
    for dens_val, dens_letter in density_map.items():
        count = density_counts[dens_val]
        percentage = (count / successful * 100) if successful > 0 else 0
        print(f"  Densidad {int(dens_val)} ({dens_letter}): {count:4d} imágenes ({percentage:5.2f}%)")
    
    print("\n" + "="*80)
    print("✅ PROCESO COMPLETADO")
    print("="*80 + "\n")


if __name__ == "__main__":
    process_all_folders()
