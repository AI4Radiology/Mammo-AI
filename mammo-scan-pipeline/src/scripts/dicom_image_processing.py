import os
import argparse
import json
import datetime
import numpy as np
import cv2
import pydicom as dicom
from skimage import measure
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gc  # Para liberación explícita de memoria


# OpenCV y NumPy para multithread
cv2.setNumThreads(0)
cv2.setUseOptimized(True)

class DICOMImageProcessor:

    def __init__(self, workers=16):
        self.workers = workers
        print(f"Procesador DICOM inicializado con {workers} workers")
    
    def load_dicom_image(self, dicom_path):
        """Cargar y normalizar imagen DICOM - Método del notebook"""

        try:
            #dicom_path = Path(dicom_path) if isinstance(dicom_path, str) else dicom_path
            dicom_file = dicom.dcmread(str(dicom_path), force=True)

            # DESCOMPRIMIR SI ES NECESARIO 
            #transfer_syntax = dicom_file.file_meta.TransferSyntaxUID

            # Obtener matriz de píxeles
            image = dicom_file.pixel_array
            
            return {
                'dicom_file': dicom_file,
                'image': image
            }
        except Exception as e:
            return print(f"  ❌ Error cargando: {e}")

    def get_image_view_position(self, dicom_file):
        """Obtener view_position directamente del archivo DICOM"""
        try:

            # Extraer View Position, Image Laterality y Density Tag de la imagen DICOM
            view_position = dicom_file.get('ViewPosition', None)
            image_laterality = dicom_file.get('ImageLaterality', None)
            density_tag = dicom_file.get((0x4010, 0x1018), None)  # Tag privado de Density
            
            return {
                'view_position': view_position,
                'image_laterality': image_laterality,
                'density': density_tag
            }
            
        except Exception as e:
            return None
    
    def denoise_and_enhance(self, image):
        """Aplicar denoising y mejora de contraste a la imagen"""

        try:
            # Aplicar Non-local Means Denoising para reducción de ruido
            denoised_image = cv2.fastNlMeansDenoising(
                image, 
                None, 
                h=10, 
                templateWindowSize=7, 
                searchWindowSize=21
            )
            
            # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization) para mejora de contraste
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            enhanced_image = clahe.apply(denoised_image)

            img_smooth = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=75, sigmaSpace=75)
            
            return img_smooth
        
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
 
    def clean_single_image(self, image):
        """Limpiar imagen individual aplicando normalización, máscara y mejora"""

        try:
            
            # Normalizar
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            image = image.astype(np.uint8)
            
            # Threshold para el fondo
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
            
            # Buscar el contorno más grande
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Crear máscara basada en el contorno más grande
            if contours:
                # Calcular áreas de todos los contornos
                contour_areas = [cv2.contourArea(c) for c in contours]
                # Seleccionar el contorno con el área más grande
                main_contour = contours[np.argmax(contour_areas)]
                
                # Crear la máscara en base al contorno
                mask = np.zeros_like(image)
                cv2.drawContours(mask, [main_contour], -1, color=255, thickness=-1)
                
                # Aplicar la máscara
                cleaned_image = cv2.bitwise_and(image, image, mask=mask)
            else:
                # Si no se encuentran contornos, usar imagen original
                mask = np.ones_like(image) * 255
                cleaned_image = image
            

            # Denoising y mejora de contraste
            img_smooth = self.denoise_and_enhance(cleaned_image)
            
            return {
                'original': image,
                'mask': mask,
                'cleaned': img_smooth,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def process_binary_and_edges(self, image):
        """Procesar imagen para obtener binarización y detección de bordes"""

        try:
            # Conversión a binario (threshold 105)
            ret, thresh1 = cv2.threshold(image, 90,255,cv2.THRESH_BINARY)
            
            # Convertir a uint8 para Canny
            thresh1_uint8 = thresh1.astype(np.uint8)
            
            # Detección de bordes con Canny
            edges = cv2.Canny(thresh1_uint8, 0, 255)

            # Dilatación de bordes para mejorar conectividad
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)) ## Mejor (4,4) 40 iter.
            dilation = cv2.dilate(edges,kernel,iterations = 40)
            # Erosión para cerrar huecos y obtener contorno sólido
            eroded_and_closed_edges = cv2.erode(dilation, kernel, iterations=20)
            # Encontrar contornos en la imagen procesada
            contours, _ = cv2.findContours(eroded_and_closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Crear nueva máscara basada en el contorno más grande
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                new_mask = np.zeros_like(eroded_and_closed_edges)
                cv2.drawContours(new_mask, [main_contour], -1, color=255, thickness=-1)
            
            return {
                'original': image,
                'binary': thresh1,
                'edges': edges,
                'dilation': dilation,
                'mask': new_mask,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    
    def apply_final_mask(self, image_mask):
        """Aplicar la máscara final a la imagen limpia"""
    
        try:
            # Aplicar la máscara morfológica a la imagen limpia
            final_cleaned = cv2.bitwise_and(image_mask['original'], image_mask['original'], mask=image_mask['mask'])

            return final_cleaned
 
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def crop_image_parallel(self, image_with_index):
        """Recortar imagen usando contornos en paralelo"""

        # Desempaquetar tupla
        image, index = image_with_index
        
        try:
            # Buscar contornos en la imagen para recorte
            # Aplicar threshold para encontrar regiones
            contours = measure.find_contours(image, level=50)
            
            if contours:
                # Encontrar el contorno más grande
                main_contour = max(contours, key=len)
                
                # Convertir puntos del contorno a enteros para cálculo de bounding box
                main_contour_int = np.array(main_contour).astype(np.int32)

                # Calcular bounding box para el contorno principal
                x, y, w, h = cv2.boundingRect(main_contour_int)
                
                # Recortar imagen al bounding box
                cropped_image = image[x:x+w,y:y+h]
                
                return {
                    'index': index,
                    'original': image,
                    'cropped': cropped_image,
                    'bbox': (x, y, w, h),
                    'contour_found': True,
                    'success': True
                }
            else:
                # Si no se encuentran contornos, usar imagen original
                return {
                    'index': index,
                    'original': image,
                    'cropped': image,
                    'bbox': None,
                    'contour_found': False,
                    'success': True
                }
                
        except Exception as e:
            return {
                'index': index,
                'error': str(e),
                'success': False
            }
    

    
    def save_results_unified(self, original_images, masks, image_type="processed", output_folder="results"):
        """Guardar resultados unificados para CC y pipeline completo"""
        
        # Crear carpetas
        results_path = Path(output_folder)
        results_path.mkdir(exist_ok=True)
        
        images_path = results_path / "processed_images"
        masks_path = results_path / "masks"
        images_path.mkdir(exist_ok=True)
        masks_path.mkdir(exist_ok=True)
        
        saved_count = 0
        
        # Guardar imágenes y máscaras en las carpetas correspondientes
        for i, (original, mask) in enumerate(zip(original_images, masks)):
            try:
                # Guardar imagen original
                image_file = images_path / f"{image_type}_image_{i:03d}.npy"
                np.save(image_file, original)
                
                # Guardar máscara correspondiente
                mask_file = masks_path / f"{image_type}_mask_{i:03d}.npy"
                np.save(mask_file, mask)
                
                saved_count += 1
                
            except Exception as e:
                print(f"Error guardando {image_type} {i}: {e}")
        
        print(f"Guardados {image_type}: {saved_count} pares en {output_folder}")
        return saved_count
        

    def find_dicom_files(self, base_path):
        """Buscar archivos DICOM en estructura de subcarpetas"""
        
        # Asegurar que base_path es Path para evitar errores
        base_path = Path(base_path)
        
        if not base_path.exists():
            print(f"❌ La ruta {base_path} no existe")
            return []
        
        # Obtener todas las subcarpetas
        folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
        
        dicom_files = []

        # Validar si se encontraron carpetas
        if not folders:

            print(f"❌ No se encontraron carpetas en {base_path}")
            
            files_found = list(base_path.rglob("*.dicom")) + list(base_path.rglob("*.dcm"))
            dicom_files = [str(f) for f in files_found]
            return dicom_files
        
        print(f"📁 Escaneando {len(folders)} carpetas...")
        
        
        
        # Iterar por cada carpeta y buscar archivos DICOM
        for folder in folders:
            # Buscar .dicom y .dcm en cada carpeta
            files_in_folder = list(folder.rglob("*.dicom")) + list(folder.rglob("*.dcm"))
            
            # Convertir Path a string y agregar a la lista
            for file in files_in_folder:
                dicom_files.append(str(file))
        
        return dicom_files


    def define_image_views(self, dicom_file):
        """Definir las vistas de imagen basadas en el view_position"""
        
        # Obtener view_position e image_laterality
        view_result = self.get_image_view_position(dicom_file)
        print(f"View Position: {view_result['view_position']}, Image Laterality: {view_result['image_laterality']}")
       

        return view_result
    

    def process_one(self, path):
        """Procesar una sola imagen DICOM y retornar resultados según vista"""

        # Extraer el nombre del archivo (image_id) sin extensión
        image_id = Path(path).stem
        
        # Procesar la imagen DICOM
        try:
            # Cargar imagen DICOM
            image_dicom = self.load_dicom_image(path)
            
            if image_dicom is None:
                print(f"❌ Error cargando imagen {image_id}")
                return [None, None, None, None, image_id]
            
            # Obtener información de la imagen (densidad, vista, lateralidad)
            view_result = self.get_image_view_position(image_dicom['dicom_file'])

            # Ajustar orientación si es imagen derecha
            if view_result['image_laterality'] == 'R':
                image_dicom['image'] = cv2.flip(image_dicom['image'], 1)

            # Limpiar la imagen DICOM (normalización, máscara, mejora)
            cleaning_results = self.clean_single_image(image_dicom['image'])
            
            # Verificar si hubo un error en la limpieza de la imagen
            if not cleaning_results.get('success', True):
                print(f"❌ Error procesando imagen {image_id}: {cleaning_results.get('error', 'Error desconocido')}")
                # Liberar memoria
                del image_dicom, cleaning_results
                return [None, None, None, view_result['density'], image_id]
            
            # Procesar según la vista de la imagen
            if view_result['view_position'] == 'MLO':
                # Procesar binarización y bordes para MLO
                binary_results = self.process_binary_and_edges(cleaning_results['cleaned'])
                apply_final_mask = self.apply_final_mask(binary_results)

                mask = binary_results['mask'].copy()
                image = cleaning_results['original'].copy()
                density = view_result['density']

                # Liberar memoria de variables intermedias ANTES de retornar
                del image_dicom, cleaning_results, binary_results, apply_final_mask, view_result
                
                # Fix: Ahora MLO también retorna densidad e image_id
                return [image, mask, apply_final_mask, density, image_id]

            # Liberar memoria de variables intermedias para CC
            original = cleaning_results['original'].copy()
            mask = cleaning_results['mask'].copy()
            cleaned = cleaning_results['cleaned'].copy()
            density = view_result['density']
            
            del image_dicom, cleaning_results, view_result
            
            # Retornar también el image_id para CC
            return [original, mask, cleaned, density, image_id]
            
        except Exception as e:
            print(f"❌ Error crítico procesando {image_id}: {e}")
            return [None, None, None, None, image_id]
        
    def process_dicom_folder(self, dicom_paths, batch_size=500, resume_folder=None): 
        """Procesar una lista de archivos DICOM en lotes y guardar resultados en carpetas organizadas."""

        all_results = []
        total = len(dicom_paths)
        
        # Si se proporciona resume_folder, usar esa carpeta; sino crear nueva
        if resume_folder:
            output_folder = resume_folder
            base_path = Path(output_folder)
            
            if not base_path.exists():
                print(f"❌ La carpeta {resume_folder} no existe. Creando nueva...")
                resume_folder = None
        
        if not resume_folder:
            # Crear timestamp único para toda la sesión
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = f"./data/processed/{timestamp}"
            base_path = Path(output_folder)
            base_path.mkdir(parents=True, exist_ok=True)
        
        # Crear subcarpetas UNA SOLA VEZ
        originals_path = base_path / "originals"
        masks_path = base_path / "masks"
        cleaned_path = base_path / "cleaned"
        densities_path = base_path / "densities"
        
        originals_path.mkdir(exist_ok=True)
        masks_path.mkdir(exist_ok=True)
        cleaned_path.mkdir(exist_ok=True)
        densities_path.mkdir(exist_ok=True)
        
        # DETECTAR ÚLTIMO ÍNDICE PROCESADO (para reanudar en caso de ser necesario)
        existing_files = sorted(list(originals_path.glob('original_*.npy')))
        if existing_files:
            # Obtener el último archivo procesado para continuar la numeración
            last_file = existing_files[-1]
            # Extraer el índice del nombre del archivo
            last_index = int(last_file.stem.split('_')[1])
            global_index = last_index + 1
            skip_count = global_index
            
            print(f"🔄 REANUDANDO desde imagen {global_index}")
            print(f"📁 Carpeta existente: {output_folder}")
            print(f"✅ Ya procesadas: {len(existing_files)} imágenes")
            
            # Saltar las imágenes ya procesadas
            dicom_paths = dicom_paths[skip_count:]
            total = len(dicom_paths)
            
            if total == 0:
                print(f"✅ ¡Todas las imágenes ya están procesadas!")
                return all_results
            
            print(f"📦 Imágenes restantes: {total}")
        else:
            global_index = 0
            skip_count = 0
            print(f"📁 Carpeta de salida: {output_folder}")
            print(f"📦 Procesando en lotes de {batch_size} imágenes (guardando en carpeta única)")
        
        # Variables para numeración continua
        all_density_mapping = {}
        total_successful = 0
        total_failed = 0
        
        # Procesar en lotes
        for batch_num, batch_start in enumerate(range(0, total, batch_size), 1):
            # Calcular el rango del lote actual sacando el mínimo entre el tamaño del lote y el total
            batch_end = min(batch_start + batch_size, total)
            # Obtener los paths del lote actual
            batch_paths = dicom_paths[batch_start:batch_end]
            
            print(f"\n{'='*60}")
            print(f"🔄 LOTE {batch_num}: Procesando imágenes {batch_start+1} a {batch_end}")
            print(f"{'='*60}")
            
            originals = []
            masks = []
            cleaned_images = []
            densities = []
            image_ids = []
            
            # Procesar imágenes en paralelo usando ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=16) as executor:

                # Procesar imágenes en paralelo y recopilar resultados
                for idx, result in enumerate(executor.map(self.process_one, batch_paths)):
                    # Filtrar resultados nulos
                    if result[0] is not None:
                        originals.append(result[0])
                        masks.append(result[1])
                        cleaned_images.append(result[2])
                        densities.append(result[3])
                        image_ids.append(result[4])
                    
                    # Actualizar índice global para seguimiento
                    global_idx = batch_start + idx + 1
                    if (idx+1) % 50 == 0 or (idx+1) == len(batch_paths):
                        successful = len(originals)
                        failed = (idx+1) - successful
                        print(f"  Procesadas {global_idx}/{total} imágenes ({successful} exitosas en lote, {failed} fallidas en lote)")
            
            # Guardar lote actual en la MISMA carpeta con numeración continua
            print(f"💾 Guardando lote {batch_num} en {output_folder}...")
            saved, density_mapping = self.save_batch_continuous(
                originals, masks, cleaned_images, densities, image_ids,
                originals_path, masks_path, cleaned_path, densities_path,
                global_index
            )
            
            # Actualizar mapeo global
            all_density_mapping.update(density_mapping)
            
            # Actualizar contadores
            global_index += saved
            total_successful += saved
            total_failed += (len(batch_paths) - saved)
            
            # Agregar resultados del lote a la lista completa
            all_results.extend([(o, m, c, d, i) for o, m, c, d, i in zip(originals, masks, cleaned_images, densities, image_ids)])
            
            # Liberar memoria del lote MÁS AGRESIVAMENTE
            del originals, masks, cleaned_images, densities, image_ids, density_mapping
            gc.collect()
            gc.collect()  # Segunda pasada para asegurar liberación
            
            print(f"✅ Lote {batch_num} guardado: {saved} imágenes (índices {global_index-saved} a {global_index-1})")
            print(f"🗑️  Memoria liberada agresivamente")
        
        # Guardar mapeo consolidado y metadata al final
        print(f"\n💾 Guardando archivos finales...")
        timestamp = base_path.name if not resume_folder else base_path.name
        self.save_final_metadata(base_path, all_density_mapping, total_successful, total_failed, timestamp)
        
        print(f"\n{'='*60}")
        print(f"🎉 PROCESAMIENTO COMPLETO")
        print(f"{'='*60}")
        print(f"✅ Exitosas: {total_successful}/{total}")
        print(f"❌ Fallidas: {total_failed}/{total}")
        print(f"📁 Todo guardado en: {output_folder}")
        print(f"{'='*60}")
        
        return all_results

    def save_batch_continuous(self, originals, masks, cleaned_images, densities, image_ids,
                             originals_path, masks_path, cleaned_path, densities_path, start_index):
        """Guardar un lote con numeración continua"""
        
        saved_count = 0
        density_mapping = {}
        
        # Mapeo de valores float a letras
        density_letter_map = {0.0: 'A', 1.0: 'B', 2.0: 'C', 3.0: 'D'}
        
        for i, (original, mask, cleaned, density, image_id) in enumerate(zip(originals, masks, cleaned_images, densities, image_ids)):
            try:
                # Índice global continuo
                global_idx = start_index + i
                
                # Nombre del archivo procesado con índice global
                processed_filename = f"original_{global_idx:05d}.npy"
                
                # Guardar como archivos .npy
                np.save(originals_path / processed_filename, original)
                np.save(masks_path / f"mask_{global_idx:05d}.npy", mask)
                np.save(cleaned_path / f"cleaned_{global_idx:05d}.npy", cleaned)
                np.save(densities_path / f"density_{global_idx:05d}.npy", density)
                
                # Convertir density a float
                density_value = float(density.value) if density is not None else None
                density_class = density_letter_map.get(density_value, 'Unknown') if density_value else 'Unknown'
                
                # Agregar al mapeo
                density_mapping[processed_filename] = {
                    'image_id': image_id,
                    'density_value': density_value,
                    'density_class': density_class,
                    'index': global_idx
                }
                
                saved_count += 1
                
            except Exception as e:
                print(f"Error guardando imagen {global_idx}: {e}")
        
        return saved_count, density_mapping

    def save_final_metadata(self, base_path, density_mapping, total_successful, total_failed, timestamp):
        """Guardar archivos finales consolidados"""
        
        # Guardar el mapeo de densidad completo
        density_mapping_file = base_path / "density_mapping.json"
        with open(density_mapping_file, 'w') as f:
            json.dump(density_mapping, f, indent=2)
        
        print(f"✅ Guardado mapeo de densidad: {density_mapping_file}")
        
        # Crear archivo de metadatos con información completa
        metadata = {
            'timestamp': timestamp,
            'total_images_processed': total_successful + total_failed,
            'total_images_successful': total_successful,
            'total_images_failed': total_failed,
            'success_rate': f"{(total_successful/(total_successful+total_failed)*100):.2f}%",
            'density_mapping_file': str(density_mapping_file),
            'folder_structure': {
                'originals': str(base_path / "originals"),
                'masks': str(base_path / "masks"),
                'cleaned': str(base_path / "cleaned"),
                'densities': str(base_path / "densities")
            },
            'density_distribution': {
                'A': sum(1 for v in density_mapping.values() if v['density_class'] == 'A'),
                'B': sum(1 for v in density_mapping.values() if v['density_class'] == 'B'),
                'C': sum(1 for v in density_mapping.values() if v['density_class'] == 'C'),
                'D': sum(1 for v in density_mapping.values() if v['density_class'] == 'D'),
                'Unknown': sum(1 for v in density_mapping.values() if v['density_class'] == 'Unknown')
            }
        }
        
        # Guardar metadata como JSON
        metadata_file = base_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Guardado metadata: {metadata_file}")
        print(f"📊 Distribución de densidad: A={metadata['density_distribution']['A']}, B={metadata['density_distribution']['B']}, C={metadata['density_distribution']['C']}, D={metadata['density_distribution']['D']}, Unknown={metadata['density_distribution']['Unknown']}")

    def save_training_data(self, originals, masks, cleaned_images, densities, image_ids, output_folder, batch_num=None):
        """Guardar datos para entrenamiento de modelos con estructura timestamp"""
        
        # Crear carpetas con estructura data/processed/timestamp
        base_path = Path(output_folder)
        base_path.mkdir(parents=True, exist_ok=True)
        
        originals_path = base_path / "originals"
        masks_path = base_path / "masks"
        cleaned_path = base_path / "cleaned"
        densities_path = base_path / "densities"

        
        originals_path.mkdir(exist_ok=True)
        masks_path.mkdir(exist_ok=True)
        cleaned_path.mkdir(exist_ok=True)
        densities_path.mkdir(exist_ok=True)
        
        saved_count = 0
        density_mapping = {}  # Diccionario para el mapeo de densidad
        
        # Mapeo de valores float a letras
        density_letter_map = {0.0: 'A', 1.0: 'B', 2.0: 'C', 3.0: 'D'}
        
        # Guardar imágenes y máscaras en las carpetas correspondientes en orden
        for i, (original, mask, cleaned, density, image_id) in enumerate(zip(originals, masks, cleaned_images, densities, image_ids)):
            try:
                # Nombre del archivo procesado
                processed_filename = f"original_{i:04d}.npy"
                
                # Guardar como archivos .npy (formato numpy nativo)
                np.save(originals_path / processed_filename, original)
                np.save(masks_path / f"mask_{i:04d}.npy", mask)
                np.save(cleaned_path / f"cleaned_{i:04d}.npy", cleaned)
                np.save(densities_path / f"density_{i:04d}.npy", density)
                
                # Convertir density a float si es posible, manejar None
                density_value = float(density.value) if density is not None else None
                density_class = density_letter_map.get(density_value, 'Unknown') if density_value else 'Unknown'
                
                # Agregar al mapeo de densidad
                density_mapping[processed_filename] = {
                    'image_id': image_id,
                    'density_value': density_value,
                    'density_class': density_class,
                    'index': i
                }
                
                saved_count += 1
                
            except Exception as e:
                print(f"Error guardando imagen {i}: {e}")
        
        batch_info = f" (Lote {batch_num})" if batch_num else ""
        print(f"Guardadas {saved_count} imágenes en {output_folder}{batch_info}")
        print(f"Estructura: {output_folder}/{{originals,masks,cleaned,densities}}/archivo_XXXX.npy")
        
        # Guardar el mapeo de densidad como JSON
        density_mapping_file = base_path / "density_mapping.json"
        with open(density_mapping_file, 'w') as f:
            json.dump(density_mapping, f, indent=2)
        
        print(f"✅ Guardado mapeo de densidad: {density_mapping_file}")
        
        # Crear archivo de metadatos con información del procesamiento
        metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'batch_number': batch_num,
            'total_images': saved_count,
            'density_mapping_file': str(density_mapping_file),
            'folder_structure': {
                'originals': str(originals_path),
                'masks': str(masks_path),
                'cleaned': str(cleaned_path),
                'densities': str(densities_path)
            },
            'density_distribution': {
                'A': sum(1 for v in density_mapping.values() if v['density_class'] == 'A'),
                'B': sum(1 for v in density_mapping.values() if v['density_class'] == 'B'),
                'C': sum(1 for v in density_mapping.values() if v['density_class'] == 'C'),
                'D': sum(1 for v in density_mapping.values() if v['density_class'] == 'D'),
                'Unknown': sum(1 for v in density_mapping.values() if v['density_class'] == 'Unknown')
            }
        }
        
        metadata_file = base_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Guardado metadata: {metadata_file}")
        print(f"📊 Distribución de densidad: A={metadata['density_distribution']['A']}, B={metadata['density_distribution']['B']}, C={metadata['density_distribution']['C']}, D={metadata['density_distribution']['D']}, Unknown={metadata['density_distribution']['Unknown']}")
        
        return saved_count
                
def main():
    """Función principal para ejecución independiente"""

    # Configuración de argumentos
    parser = argparse.ArgumentParser(description='Procesamiento de Imágenes DICOM')
    parser.add_argument('--workers', '-w', type=int, default=16, help='Número de workers')
    parser.add_argument('--input', '-i', type=str, default="./cc",
                       help='Carpeta base con subcarpetas de imágenes DICOM o archivos DICOM individuales')
    parser.add_argument('--resume', '-r', type=str, default=None,
                       help='Carpeta existente para reanudar procesamiento (ej: ./data/processed/20251011_174757)')
    parser.add_argument('--batch-size', '-b', type=int, default=500,
                       help='Tamaño del lote para procesamiento (default: 500)')
    
    args = parser.parse_args()
    
    # Si se especifica --resume, verificar que exista
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"❌ La carpeta {args.resume} no existe")
            print(f"💡 Usa --resume con una carpeta válida o no uses este argumento para comenzar nuevo")
            return []
        print(f"🔄 Modo REANUDAR activado")
    
    processor = DICOMImageProcessor(workers=args.workers)
    
    # Obtener todos los archivos DICOM de las subcarpetas
    dicom_files = processor.find_dicom_files(args.input)
    
    if not dicom_files:
        print("❌ No se encontraron archivos DICOM")
        return []
    
    print(f"📊 Total de archivos DICOM encontrados: {len(dicom_files)}")
    
    # Procesar todas las imágenes en paralelo
    all_results = processor.process_dicom_folder(dicom_files, batch_size=args.batch_size, resume_folder=args.resume)
    print(f"\n✅ Procesadas {len(all_results)} imágenes en total")

    return all_results

if __name__ == "__main__":
    exit(main())


