"""
Script de prueba para el sistema DICOM ETL
==========================================

Ejecutar para verificar que todos los componentes funcionan correctamente.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Probar que todos los imports funcionen"""
    print("="*60)
    print("TEST 1: Verificando imports...")
    print("="*60)
    
    tests = [
        ("watchdog", "Watchdog para monitoreo de archivos"),
        ("pydicom", "Lectura de archivos DICOM"),
        ("pandas", "Manejo de DataFrames"),
        ("numpy", "Operaciones numéricas"),
        ("cv2", "OpenCV para procesamiento de imágenes"),
        ("SimpleITK", "SimpleITK para radiomics"),
        ("radiomics", "PyRadiomics para extracción"),
        ("yaml", "YAML para configuración"),
        ("joblib", "Carga de modelos ML"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in tests:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name:20s} - {description}")
            passed += 1
        except ImportError as e:
            print(f"  ✗ {module_name:20s} - ERROR: {e}")
            failed += 1
    
    print(f"\nResultado: {passed}/{len(tests)} imports exitosos")
    return failed == 0


def test_services():
    """Probar que los servicios se puedan importar"""
    print("\n" + "="*60)
    print("TEST 2: Verificando servicios...")
    print("="*60)
    
    services = [
        ("services.dicom_image_processing", "DICOMImageProcessor"),
        ("services.radiomic_extraction", "RadiomicsMLTrainer"),
        ("services.classify", "classify_dataframe"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, class_or_func in services:
        try:
            module = __import__(module_name, fromlist=[class_or_func])
            obj = getattr(module, class_or_func)
            print(f"  ✓ {module_name:40s} [{class_or_func}]")
            passed += 1
        except Exception as e:
            print(f"  ✗ {module_name:40s} - ERROR: {e}")
            failed += 1
    
    print(f"\nResultado: {passed}/{len(services)} servicios disponibles")
    return failed == 0


def test_config():
    """Probar que los archivos de configuración existan"""
    print("\n" + "="*60)
    print("TEST 3: Verificando configuración...")
    print("="*60)
    
    configs = [
        ("config/kbest.txt", "Lista de features K-best"),
        ("config/radiomics_config.yaml", "Configuración de radiomics"),
    ]
    
    passed = 0
    failed = 0
    
    for file_path, description in configs:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {file_path:40s} ({size} bytes) - {description}")
            passed += 1
        else:
            print(f"  ✗ {file_path:40s} - NO ENCONTRADO")
            failed += 1
    
    print(f"\nResultado: {passed}/{len(configs)} archivos de configuración encontrados")
    return failed == 0


def test_models():
    """Probar que los modelos ML existan"""
    print("\n" + "="*60)
    print("TEST 4: Verificando modelos ML...")
    print("="*60)
    
    models = [
        ("models/modelo_binario.pkl", "Clasificador binario (denso/no denso)"),
        ("models/modelo_multiclase.pkl", "Clasificador BI-RADS (A, B, C, D)"),
    ]
    
    passed = 0
    failed = 0
    
    for file_path, description in models:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {file_path:40s} ({size} bytes) - {description}")
            passed += 1
        else:
            print(f"  ✗ {file_path:40s} - NO ENCONTRADO")
            failed += 1
    
    print(f"\nResultado: {passed}/{len(models)} modelos encontrados")
    
    if failed > 0:
        print("\n  ADVERTENCIA: Faltan modelos ML. El sistema no funcionará sin ellos.")
    
    return failed == 0


def test_directories():
    """Verificar que las carpetas del sistema se puedan crear"""
    print("\n" + "="*60)
    print("TEST 5: Verificando estructura de carpetas...")
    print("="*60)
    
    dirs = ["input", "output", "data", "logs", "error"]
    
    passed = 0
    failed = 0
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        try:
            dir_path.mkdir(exist_ok=True)
            if dir_path.exists() and dir_path.is_dir():
                print(f"  ✓ {dir_name:15s} - OK")
                passed += 1
            else:
                print(f"  ✗ {dir_name:15s} - No se pudo crear")
                failed += 1
        except Exception as e:
            print(f"  ✗ {dir_name:15s} - ERROR: {e}")
            failed += 1
    
    print(f"\nResultado: {passed}/{len(dirs)} carpetas disponibles")
    return failed == 0


def test_watchdog_module():
    """Probar que el módulo principal de watchdog se pueda importar"""
    print("\n" + "="*60)
    print("TEST 6: Verificando módulo principal...")
    print("="*60)
    
    try:
        import dicom_watchdog
        print("  ✓ dicom_watchdog.py - Módulo importado correctamente")
        
        # Verificar clases principales
        classes = [
            "Config",
            "HL7MessageGenerator",
            "DICOMProcessor",
            "DICOMFileHandler",
            "QueueProcessor",
            "DICOMWatchdogApp"
        ]
        
        for class_name in classes:
            if hasattr(dicom_watchdog, class_name):
                print(f"  ✓ Clase {class_name} disponible")
            else:
                print(f"  ✗ Clase {class_name} NO encontrada")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error importando dicom_watchdog: {e}")
        traceback.print_exc()
        return False


def main():
    """Ejecutar todos los tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "MAMMO-SCAN ETL - TEST SUITE" + " "*20 + "║")
    print("╚" + "="*58 + "╝")
    print()
    
    results = []
    
    # Ejecutar tests
    results.append(("Imports", test_imports()))
    results.append(("Servicios", test_services()))
    results.append(("Configuración", test_config()))
    results.append(("Modelos ML", test_models()))
    results.append(("Carpetas", test_directories()))
    results.append(("Módulo principal", test_watchdog_module()))
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE TESTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:10s} - {test_name}")
    
    print(f"\nTotal: {passed}/{len(results)} tests pasados")
    
    # Conclusión
    print("\n" + "="*60)
    if failed == 0:
        print("✓ TODOS LOS TESTS PASARON")
        print("="*60)
        print("\nEl sistema está listo para usar.")
        print("Ejecutar: python dicom_watchdog.py")
        return 0
    else:
        print("✗ ALGUNOS TESTS FALLARON")
        print("="*60)
        print(f"\n{failed} problema(s) encontrado(s).")
        print("Por favor resolver los errores antes de usar el sistema.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
