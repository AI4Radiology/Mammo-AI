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


# OpenCV y NumPy para multithread
cv2.setNumThreads(0)
cv2.setUseOptimized(True)

class DICOMImageProcessor:

    def __init__(self, workers=12):
        self.workers = workers
        print(f"Procesador DICOM inicializado con {workers} workers")

    def read_dicom_bytes(self, file_stream: bytes) -> np.ndarray:
        """Read a DICOM file from bytes and return the raw DICOM pixel array.

        Returns the image without normalization to preserve original intensity values
        required for accurate radiomics extraction.
        """
        from io import BytesIO

        if file_stream is None:
            raise ValueError("file_stream is None")

        bio = BytesIO(file_stream)
        dicom_file = dicom.dcmread(bio)
        image = dicom_file.pixel_array
        
        # Return image as float64 without normalization
        # This preserves the original DICOM intensity values
        img = image.astype(np.float64)
        
        return dicom_file, img

    def get_image_view_position(self, dicom_file):
        """Obtener view_position directamente del archivo DICOM"""
        try:

            # Extraer View Position y Image Laterality
            view_position = dicom_file.get('ViewPosition', None)
            image_laterality = dicom_file.get('ImageLaterality', None)
            
            return {
                'view_position': view_position,
                'image_laterality': image_laterality
            }
            
        except Exception as e:
            return None
 
    def clean_single_image(self, image):
        """
        Minimal preprocessing for radiomics extraction.
        
        Only performs segmentation to create a binary mask without modifying
        the original image intensities. This approach preserves the DICOM 
        intensity values required for accurate radiomics feature extraction.
        
        Args:
            image: Raw DICOM pixel array (float64, original intensity values)
            
        Returns:
            dict with:
                - original: Original image (float64)
                - mask: Binary mask (uint8, 0 or 1)
                - cleaned: Original image (float64, same as input)
                - success: bool
        """
        try:
            # Keep original image as float64
            img_original = image.astype(np.float64)
            
            # Create temporary normalized image ONLY for threshold/segmentation
            # This is used only to find contours, not for radiomics extraction
            img_norm_temp = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            # Threshold using Otsu's method for automatic threshold selection
            _, binary = cv2.threshold(img_norm_temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (the breast)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create binary mask (0 and 1, not 0 and 255)
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, 1, -1)
            else:
                # If no contours found, use full image
                mask = np.ones(image.shape, dtype=np.uint8)
            
            # Return original image without any intensity modifications
            return {
                'original': img_original,
                'mask': mask,
                'cleaned': img_original,  # Same as original - no processing applied
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }