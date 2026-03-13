import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import sys


# ============================================================
# 1. FUNCIÓN PARA SEGMENTACIÓN Y MÁSCARA
# ============================================================

def clean_single_image(image):
    """
    Segmentación mínima para extracción radiómica.
    No altera intensidades del DICOM.
    """
    try:
        img_original = image.astype(np.float64)

        img_norm_temp = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        _, binary = cv2.threshold(img_norm_temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 1, -1)

        else:
            mask = np.ones(image.shape, dtype=np.uint8)

        return {
            'original': img_original,
            'mask': mask,
            'cleaned': img_original,
            'success': True
        }

    except Exception as e:
        print(f"Error en clean_single_image: {e}")
        return {'success': False}


# ============================================================
# 2. FUNCIÓN PARA EXTRAER RADÓMICOS
# ============================================================

def extract_radiomics_features(image, mask):
    """
    Extraer características radiómicas sin modificar intensidades.
    """
    try:
        if image is None or mask is None:
            return None

        if image.shape != mask.shape:
            print("Dimensiones imagen/máscara no coinciden")
            return None

        image = np.asarray(image, dtype=np.float64)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        if np.sum(mask) == 0:
            print("Máscara vacía")
            return None

        settings = {
            'binWidth': 25,
            'normalize': False,
            'force2D': True,
            'force2Ddimension': 0
        }

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

        extractor.enableImageTypeByName('Original')
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('gldm')
        extractor.enableFeatureClassByName('ngtdm')
        extractor.enableFeatureClassByName('shape2D')

        image_sitk = sitk.GetImageFromArray(image)
        mask_sitk = sitk.GetImageFromArray(mask)

        result = extractor.execute(image_sitk, mask_sitk)

        features = {}

        for key, value in result.items():
            if not key.startswith("diagnostics"):
                try:
                    features[key] = float(value)
                except:
                    pass

        return features

    except Exception as e:
        print(f"Error en extract_radiomics_features: {e}")
        return None


# ============================================================
# 3. LECTURA DE CARPETA COMPLETA + PARSING DEL NOMBRE
# ============================================================

def parse_filename(filename):
    """
    Formato esperado:
       N Densidad_VISTA_LADO
    Ejemplo: 123A_CC_R.dcm
    """
    name = os.path.splitext(filename)[0]

    # Separar número + resto (ej: "123A_CC_R" o "68491_C")
    prefix = ''.join([c for c in name if c.isdigit()])
    rest = name[len(prefix):].lstrip('_')

    # Normalizar y dividir por guiones bajos
    parts = [p for p in rest.split('_') if p != '']

    # Casos posibles:
    # - ['A','CC','R'] -> densidad, vista, lado
    # - ['A','CC'] -> densidad, vista (lado faltante)
    # - ['C'] -> solo densidad (vista/lado faltantes)
    if len(parts) == 3:
        densidad, vista, lado = parts
    elif len(parts) == 2:
        densidad, vista = parts
        lado = ''
    elif len(parts) == 1 and parts[0]:
        densidad = parts[0]
        vista = ''
        lado = ''
    else:
        raise ValueError(f"Nombre no válido o sin información esperada: {filename}")

    return densidad, vista, lado


def densidad_to_labels(densidad):
    """
    binario: A,B → 0   |   C,D → 1
    multiclase: A=0, B=1, C=2, D=3
    """
    dens = densidad.strip().upper()
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    if dens not in mapping:
        raise ValueError(f"Valor de densidad desconocido: {densidad}")

    multiclase = mapping[dens]
    binario = 0 if dens in ["A", "B"] else 1
    return binario, multiclase


# ============================================================
# 4. PIPELINE COMPLETO
# ============================================================

def procesar_carpeta_dicoms(ruta_folder):

    registros = []

    def _print_missing_pixel_handlers_instructions():
        msg = (
            "To decode compressed DICOM pixel data you need additional handlers.\n"
            "Install these packages into your environment (run inside your venv if you use one):\n"
            "  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm\n"
            "If you already have those, ensure the Python interpreter running this script is the one with the packages installed.\n"
            "After installing, restart the Python process and run the script again."
        )
        print(msg)

    for archivo in os.listdir(ruta_folder):
        if not archivo.lower().endswith(".dcm"):
            continue

        try:
            densidad, vista, lado = parse_filename(archivo)

            if vista.upper() == "MLO":
                print(f"Saltando MLO: {archivo}")
                continue

            ruta_archivo = os.path.join(ruta_folder, archivo)
            ds = pydicom.dcmread(ruta_archivo)

            try:
                imagen = ds.pixel_array.astype(np.float64)
            except Exception as e:
                err_str = str(e)
                # Common pydicom message when no handlers are available references GDCM/pylibjpeg
                if 'handlers are available' in err_str or 'GDCM' in err_str or 'pylibjpeg' in err_str:
                    print(f"Falló decodificación de {archivo}: {err_str}")
                    _print_missing_pixel_handlers_instructions()
                else:
                    print(f"Error al obtener pixel_array de {archivo}: {e}")
                continue

            segment = clean_single_image(imagen)

            if not segment["success"]:
                print(f"Falló segmentación: {archivo}")
                continue

            features = extract_radiomics_features(
                segment["original"],
                segment["mask"]
            )

            if features is None:
                print(f"Falló extracción radiómica: {archivo}")
                continue

            binario, multiclase = densidad_to_labels(densidad)

            fila = {
                "archivo": archivo,
                "densidad": densidad,
                "vista": vista,
                "lado": lado,
                "binario": binario,
                "multiclase": multiclase
            }

            fila.update(features)

            registros.append(fila)

        except Exception as e:
            print(f"Error procesando {archivo}: {e}")

    df = pd.DataFrame(registros)
    return df


# ============================================================
# 5. EJECUCIÓN
# ============================================================

images = ["images\\mamografia\\new-mammo"]

#images = ["images\\mamografia\\paciente_1_C", "images\\mamografia\\paciente_2_B",
#          "images\\mamografia\\paciente_3_C", "images\\mamografia\\paciente_4_D",
#          "images\\mamografia\\paciente_5_B"]  # Reemplazar con rutas reales

for img_folder in images:
    print(f"Procesando carpeta: {img_folder}")
    df_radiomics = procesar_carpeta_dicoms(img_folder)
    output_csv = os.path.join(img_folder, "radiomics_features.csv")
    df_radiomics.to_csv(output_csv, index=False)
    print(f"Características guardadas en: {output_csv}")
