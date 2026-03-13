import os
import time
import joblib
import pandas as pd
import numpy as np


# =========================================================
# Utilidades para cargar modelos y características
# =========================================================

def load_kbest_features(path):
    with open(path, "r") as f:
        feats = [line.strip() for line in f if line.strip()]
    return feats


def load_all_models(models_dir):
    models_info = []

    for file in os.listdir(models_dir):
        if not file.endswith(".pkl"):
            continue
        
        model_path = os.path.join(models_dir, file)
        model_data = joblib.load(model_path)
        model_name = file.replace(".pkl", "")

        # Tipo de modelo
        if "_b_" in file:
            model_type = "binario"
            kbest_path = "config\\sel_bin.txt"
        elif "_m_" in file:
            model_type = "multiclase"
            kbest_path = "config\\sel_mult.txt"
        else:
            continue

        if not os.path.exists(kbest_path):
            print(f"⚠ No existe K-best para {model_name}")
            continue

        feats = load_kbest_features(kbest_path)

        models_info.append({
            "name": model_name,
            "type": model_type,
            "model": model_data["model"],
            "scaler": model_data["scaler"],
            "features": feats
        })

    return models_info


# =========================================================
# Subset de features
# =========================================================

def filter_features(df, feature_list):
    df_filtered = df.copy()

    available = [f for f in feature_list if f in df.columns]
    df_filtered = df_filtered[available]

    missing = set(feature_list) - set(available)
    for m in missing:
        df_filtered[m] = 0.0

    df_filtered = df_filtered[feature_list]
    return df_filtered


# =========================================================
# Evaluación por modelo
# =========================================================

def evaluate_model(model_info, df):
    model = model_info["model"]
    scaler = model_info["scaler"]
    feats = model_info["features"]

    df_feats = filter_features(df, feats)
    X = scaler.transform(df_feats)

    true_col = "binario" if model_info["type"] == "binario" else "multiclase"
    y_true = df[true_col].values

    predictions = []
    times = []
    errores = []

    print("\n📌 Errores detectados:")

    for i in range(len(df)):
        x_row = X[i].reshape(1, -1)

        start = time.time()
        pred = model.predict(x_row)[0]
        end = time.time()

        predictions.append(pred)
        times.append(end - start)

        if pred != y_true[i]:
            err_info = {
                "imagen": df.iloc[i]["archivo"],
                "real": int(y_true[i]),
                "pred": int(pred)
            }
            errores.append(err_info)

            # 🔥 IMPRIME CADA ERROR AQUÍ
            print(f"  archivo: {err_info['imagen']}  |  real: {err_info['real']}  |  predicción: {err_info['pred']}")

    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y_true)

    return accuracy, np.mean(times), errores


# =========================================================
# Pipeline principal
# =========================================================

def evaluate_all_models(models_dir, csv_path):
    df = pd.read_csv(csv_path)

    if "archivo" not in df.columns:
        raise ValueError("El CSV debe contener una columna 'archivo' con el nombre original.")

    models = load_all_models(models_dir)
    results = []

    for m in models:
        print(f"\n🔍 Evaluando modelo: {m['name']} ({m['type']})")

        acc, avg_time, errors = evaluate_model(m, df)

        results.append({
            "modelo": m["name"],
            "tipo": m["type"],
            "accuracy": acc,
            "tiempo_promedio": avg_time,
            "errores": errors
        })

        print(f"\n📊 RESULTADO MODELO {m['name']}")
        print(f"   → Accuracy: {acc:.4f}")
        print(f"   → Tiempo promedio: {avg_time:.6f} seg")
        print(f"   → Errores totales: {len(errors)}")

    return pd.DataFrame(results)


# =========================================================
# EJEMPLO DE USO
# =========================================================

models_dir = "models"
csv_path = "images\\mamografia\\new-mammo\\radiomics_features.csv"

resultados = evaluate_all_models(models_dir, csv_path)

print("\n===============================")
print("📌 RESUMEN GENERAL")
print("===============================")
print(resultados)