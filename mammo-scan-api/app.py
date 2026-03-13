from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import base64
import csv
import os
from typing import List, Tuple, Dict, Any
import numpy as np
import pydicom as dicom
from services.dicom_image_processing import DICOMImageProcessor
import cv2
from services.radiomic_extraction import RadiomicsMLTrainer
from services.classify import classify_dataframe

app = Flask(__name__)
CORS(app)

dicom_processor = DICOMImageProcessor(workers=12)
rad_trainer = RadiomicsMLTrainer()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/radiomics', methods=['POST'])
def radiomics():
    """POST /radiomics

    Expects form-data with a file field named 'dicom'. read it and proccess it in order
    to extract radiomics features. Returns a JSON with the radiomics features.
    """

    # Expect file field named 'dicom'
    if 'dicom' not in request.files:
        return jsonify({'error': "Missing file field 'dicom'"}), 400

    f = request.files['dicom']
    
    # Read file stream
    try:
        content = f.read()
    except Exception as e:
        return jsonify({'error': f'Failed to read uploaded file: {e}'}), 500
    
    # Turn to dicom
    try:
        dicom_file, image = dicom_processor.read_dicom_bytes(content)
    except Exception as e:
        return jsonify({'error': f'Failed to parse DICOM: {e}'}), 400

    # Define image views
    view_result = dicom_processor.get_image_view_position(dicom_file)

    if view_result.get('view_position') == 'MLO':
        return jsonify({'error': 'MLO view not supported', 'detail': 'Only CC view is supported'}), 400

    # Flip image if right laterality (BEFORE any processing)
    if view_result.get('image_laterality') == 'R':
        image = cv2.flip(image, 1)

    # Use the processor to segment the image and get a mask
    # IMPORTANT: This only creates a mask without modifying image intensities
    cleaning = dicom_processor.clean_single_image(image)
    if not cleaning.get('success'):
        return jsonify({'error': 'Image segmentation failed', 'detail': cleaning.get('error')}), 500

    # Extract original image and mask
    # The 'cleaned' image is the same as original - no intensity modifications applied
    cleaned = cleaning.get('cleaned')  # Same as original, preserves DICOM values
    mask = cleaning.get('mask')  # Binary mask (0 and 1)

    # Extract radiomics features from the original image with mask
    # CRITICAL: The image maintains original DICOM intensity values
    features = rad_trainer.extract_radiomics_features(cleaned, mask)
    if features is None or len(features) == 0:
        return jsonify({'error': 'Radiomics extraction failed or returned no features'}), 500

    # Transform features dict into records
    records = []
    for k, v in features.items():
        # Only include scalar numeric values
        try:
            val = float(v)
            records.append({
                'característica': k,
                'valor': val,
            })
        except Exception:
            continue
    
    if len(records) == 0:
        return jsonify({'error': 'No valid radiomics features extracted'}), 500

    # try:
    #     result = classify_dataframe({'columns': ['característica', 'valor'], 'records': records})
    # except Exception as e:
    #     return jsonify({'error': 'Classification failed', 'detail': str(e)}), 500

    # print("Classification result:", result)

    # Return columns without descripcion as requested
    return jsonify({'columns': ['característica', 'valor'], 'records': records})


@app.route('/classify', methods=['POST'])
def classify():
    """POST /classify

    Expects form-data with a file field named 'csv'. Returns JSON with the
    requested structure: binary and multi probabilities. This implementation
    uses simple deterministic math over CSV contents so it's testable.
    """

    # Accept JSON containing a dataframe-like structure: {'columns': [...], 'records': [{..}, ...]}
    if request.is_json:
        payload = request.get_json()
    else:
        return jsonify({'error': 'Expected application/json with dataframe payload'}), 400

    try:
        result = classify_dataframe(payload)
    except Exception as e:
        return jsonify({'error': str(e), 'detail': str(e)}), 500

    return jsonify(result)


if __name__ == '__main__':
    # Use 0.0.0.0 so it's reachable from other hosts if needed
    app.run(debug=True, host='0.0.0.0')


     