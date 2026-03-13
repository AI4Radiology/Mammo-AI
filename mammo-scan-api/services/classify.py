import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def load_feature_selection() -> List[str]:
    """
    Load the list of selected features from the config file.
    
    Returns:
        List[str]: List of feature names to use for classification
    """

    bin_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'sel_bin.txt')

    mult_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'sel_mult.txt')

    with open(bin_config_path, 'r') as f:
        bin_features = [line.strip() for line in f if line.strip()]

    with open(mult_config_path, 'r') as f:
        mult_features = [line.strip() for line in f if line.strip()]

    return bin_features, mult_features


def load_models() -> tuple:
    """
    Load the binary and multiclass models from disk.
    
    Returns:
        (binary_model, binary_scaler, multiclass_model, multiclass_scaler)
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    binary_model_path = os.path.join(models_dir, 'modelo_b_binary_adaboost_final.pkl')
    multiclass_model_path = os.path.join(models_dir, 'modelo_m_xgboost_final.pkl')
    
    # Load
    binary = joblib.load(binary_model_path)

    # Model
    binary_model = binary['model']

    # Scaler
    binary_scaler = binary['scaler']

    # Same for multiclass
    multiclass = joblib.load(multiclass_model_path)
    multiclass_model = multiclass['model']
    multiclass_scaler = multiclass['scaler']

    return binary_model, binary_scaler, multiclass_model, multiclass_scaler


def payload_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert the payload to a pandas DataFrame.
    
    Args:
        payload: Dictionary with 'columns' and 'records' keys
        
    Returns:
        pd.DataFrame: DataFrame with features as columns
    """
    records = payload.get('records', [])
    
    # Create a dictionary mapping feature names to values
    feature_dict = {}
    for record in records:
        feature_name = record.get('característica')
        feature_value = record.get('valor')
        if feature_name and feature_value is not None:
            feature_dict[feature_name] = feature_value
    
    # Create DataFrame with a single row
    df = pd.DataFrame([feature_dict])
    
    return df


def filter_selected_features(df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """
    Filter the DataFrame to keep only selected features.
    
    Args:
        df: DataFrame with all extracted features
        selected_features: List of feature names to keep
        
    Returns:
        pd.DataFrame: DataFrame with only selected features, in the correct order
    """
    # Keep only selected features that exist in the dataframe
    available_features = [feat for feat in selected_features if feat in df.columns]
    
    # Select and reorder columns
    df_filtered = df[available_features]
    
    # If some selected features are missing, fill them with 0 or raise an error
    missing_features = set(selected_features) - set(available_features)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with 0 values
        for feat in missing_features:
            df_filtered[feat] = 0
    
    # Ensure correct order
    df_filtered = df_filtered[selected_features]
    
    return df_filtered


def classify_dataframe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main classification function that processes the payload and returns predictions.
    
    Args:
        payload: Dictionary with 'columns' and 'records' keys containing radiomics features
        
    Returns:
        Dict[str, Any]: Dictionary with binary and multiclass predictions and probabilities
    """
    # Load kbest features
    bin_features, mult_features = load_feature_selection()
    
    # Convert payload to DataFrame
    df = payload_to_dataframe(payload)

    df_filtered_binary = df.copy()
    df_filtered_multiclass = df.copy()

    df_binary = filter_selected_features(df_filtered_binary, bin_features)
    df_multiclass = filter_selected_features(df_filtered_multiclass, mult_features)
    
    # Load models
    binary_model, binary_scaler, multiclass_model, multiclass_scaler = load_models()
    
    # Scale features binary
    df_filtered_binary = pd.DataFrame(binary_scaler.transform(df_binary), columns=df_binary.columns)

    # Binary classification (denso/no denso)
    binary_prediction = binary_model.predict(df_filtered_binary)[0]
    binary_proba = binary_model.predict_proba(df_filtered_binary)[0]

    # Scale features multiclass
    df_filtered_multiclass = pd.DataFrame(multiclass_scaler.transform(df_multiclass), columns=df_multiclass.columns)
    
    # Multiclass classification (A, B, C, D)
    multiclass_prediction = multiclass_model.predict(df_filtered_multiclass)[0]
    multiclass_proba = multiclass_model.predict_proba(df_filtered_multiclass)[0]

    # CRITICAL: Decode class labels
    # Models were trained with encoding: {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # We need to decode predictions back to letters
    MULTICLASS_DECODING = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    BINARY_DECODING = {0: 'no_denso', 1: 'denso'}
    
    # Decode binary prediction
    binary_prediction_decoded = BINARY_DECODING.get(int(binary_prediction), str(binary_prediction))
    
    # Decode multiclass prediction
    multiclass_prediction_decoded = MULTICLASS_DECODING.get(int(multiclass_prediction), str(multiclass_prediction))
    
    # Get class labels (decoded)
    binary_classes_decoded = [BINARY_DECODING.get(int(cls), str(cls)) for cls in binary_model.classes_]
    multiclass_classes_decoded = [MULTICLASS_DECODING.get(int(cls), str(cls)) for cls in multiclass_model.classes_]
    
    # Prepare result with DECODED labels
    result = {
        'binary': {
            'prediction': binary_prediction_decoded,
            'probabilities': {
                cls_decoded: float(prob) 
                for cls_decoded, prob in zip(binary_classes_decoded, binary_proba)
            }
        },
        'multiclass': {
            'prediction': multiclass_prediction_decoded,
            'probabilities': {
                cls_decoded: float(prob) 
                for cls_decoded, prob in zip(multiclass_classes_decoded, multiclass_proba)
            }
        }
    }
    
    return result
