import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA, KernelPCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


class RadiomicsSelector:
    
    def __init__(self, result_csv=None):
        self.result_csv = result_csv
    

    def load_processed_data(self):
        """
        Carga el archivo CSV de características desde data/features/{results_folder} si existe,
        o el archivo más reciente de la carpeta si no se pasa el parámetro.
        Devuelve un DataFrame de pandas.
        """

        project_root = Path(__file__).parent.parent.parent # src/scripts/ -> src/ -> proyecto_raíz/

        features_folder = project_root / "data" / "features"

        # Validar si se especificó un archivo CSV
        if self.result_csv is not None:
            file_path = features_folder / self.result_csv

        # Si no se especificó, usar el más reciente
        else:
            # Buscar el archivo CSV más reciente en la carpeta de features
            csv_files = list(features_folder.glob("*.csv"))
            if not csv_files:
                print(f"No se encontraron archivos CSV en {features_folder}")
                return None
            # Ordenar por fecha de modificación y tomar el más reciente
            csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            file_path = csv_files[0]
        
        try:
            df = pd.read_csv(file_path, sep=',', on_bad_lines='skip')
            print(f"Cargado archivo de características: {file_path}")
            return df
        
        except Exception as e:
            print(f"Error cargando CSV: {e}")
            return None
        
    
    def correlation_select(self, df, threshold=None):
        """
        Selecciona las mejores features correlacionadas con la variable 'Tipo_Tejido'.
        Devuelve una lista de nombres de columnas seleccionadas.
        threshold: valor mínimo absoluto de correlación para seleccionar la feature (por defecto percentil 75).
        """
        
        # Separar X y y
        X = df.drop(columns=['Tipo_Tejido'])
        y = df['Tipo_Tejido']
        
        # SOLUCIÓN: Codificar variable categórica a numérica
        
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)
        
        # Calcular correlación de cada feature con la variable objetivo numérica
        corrs = []
        for col in X.columns:
            try:
                corr = np.abs(np.corrcoef(X[col], y_numeric)[0,1])
                if np.isnan(corr):
                    corr = 0
            except Exception:
                corr = 0
            corrs.append(corr)
        
        # Calcular threshold (percentil 75)
        if threshold is None:
            valid_corrs = [c for c in corrs if not np.isnan(c) and c > 0]
            threshold = np.percentile(valid_corrs, 75) if valid_corrs else 0.5
        
        # Seleccionar features con correlación >= threshold
        selected_features_mask = np.array(corrs) >= threshold
        selected_features = X.columns[selected_features_mask].tolist()
        
        print(f"Features seleccionadas (correlación >= {threshold:.3f}): {len(selected_features)} de {len(X.columns)}")
        return selected_features

    def select_kbest_features(self, df, threshold=None):
        """
        Selecciona las mejores features usando SelectKBest respecto a 'Tipo_Tejido'.
        Devuelve una lista de nombres de columnas seleccionadas cuyo score sea >= threshold (por defecto la media), ordenadas por score.
        """

        # Separar X y y
        X = df.drop(columns=['Tipo_Tejido'])
        y = df['Tipo_Tejido']

        # Aplicar SelectKBest
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        scores = selector.scores_
        # Calcular threshold por defecto (media)
        if threshold is None:
            valid_scores = [s for s in scores if s is not None]
            threshold = np.mean(valid_scores) if valid_scores else 0
        
        # Filtrar por threshold y ordenar por score descendente
        selected = [(score, col) for score, col in zip(scores, X.columns) if score is not None and score >= threshold]
        selected.sort(reverse=True)
        selected_features = [col for score, col in selected]
        
        print(f"Features seleccionadas (score >= {threshold:.3f}): {selected_features}")
        
        return selected_features


    def select_pca_features(self, df, variance_threshold=0.95):
        """
        Aplica PCA para reducir dimensionalidad manteniendo el 95% de la varianza.
        Devuelve el DataFrame transformado con las componentes principales y la variable objetivo.
        """
        
        # Separar X y y
        if 'Tipo_Tejido' not in df.columns:
            print("La columna 'Tipo_Tejido' no está en el DataFrame.")
            return None
        
        X = df.drop(columns=['Tipo_Tejido'])
        y = df['Tipo_Tejido']
        
        # Estandarizar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar PCA con threshold de varianza
        pca = PCA(n_components=variance_threshold)
        X_pca = pca.fit_transform(X_scaled)
        
        # Crear DataFrame con componentes principales
        n_components = X_pca.shape[1]
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
        
        # Agregar la variable objetivo
        df_pca['Tipo_Tejido'] = y
        
        print(f"PCA aplicado: {len(X.columns)} features -> {n_components} componentes principales")
        print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.3f}")
        
        return df_pca
    
    
    def select_kpca_features(self, df, kernel='rbf', n_components=None, variance_threshold=0.95, gamma=None, degree=3):
        """
        Aplica Kernel PCA para reducir dimensionalidad con transformaciones no lineales.
        Captura patrones complejos que PCA lineal no puede detectar.
        
        Args:
            df: DataFrame con features + Tipo_Tejido
            kernel: 'linear', 'poly', 'rbf', 'sigmoid' (default: 'rbf')
            n_components: Número de componentes (None = automático basado en PCA lineal)
            variance_threshold: Varianza acumulada a mantener para calcular n_components (default: 0.95)
            gamma: Parámetro del kernel rbf/poly/sigmoid (None = automático: 1/n_features)
            degree: Grado del polinomio para kernel 'poly' (default: 3)
        
        Returns:
            DataFrame con componentes KPCA + Tipo_Tejido
        """

        # Separar X y y
        if 'Tipo_Tejido' not in df.columns:
            print("La columna 'Tipo_Tejido' no está en el DataFrame.")
            return None
        
        X = df.drop(columns=['Tipo_Tejido'])
        y = df['Tipo_Tejido']
        
        # Estandarizar los datos (CRÍTICO para KPCA)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

            # Calcular gamma si es string
        n_features = X_scaled.shape[1]
        
        # Calcular gamma si es string
        if gamma == 'scale':
            gamma_value = 1.0 / (n_features * X_scaled.var())
            print(f"  → gamma='scale' calculado: {gamma_value:.6f}")
        elif gamma == 'auto':
            gamma_value = 1.0 / n_features
            print(f"  → gamma='auto' calculado: {gamma_value:.6f}")
        elif gamma is None:
            gamma_value = None
            print(f"  → gamma=None (default: {1.0/n_features:.6f})")
        else:
            gamma_value = float(gamma)
            print(f"  → gamma manual: {gamma_value:.6f}")
        
        # Calcular n_components óptimo si no se proporciona
        if n_components is None:
            # Usar PCA lineal como referencia para estimar componentes
            pca_ref = PCA(n_components=variance_threshold)
            pca_ref.fit(X_scaled)
            n_components = pca_ref.n_components_
            print(f"Número de componentes estimado: {n_components} (basado en PCA lineal con varianza {variance_threshold})")
        
        # Aplicar Kernel PCA
        kpca = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma_value,
            degree=degree,
            fit_inverse_transform=True,  # Permite reconstrucción aproximada
            random_state=42
        )
        
        print(f"Aplicando KPCA con kernel '{kernel}'...")
        X_kpca = kpca.fit_transform(X_scaled)
        
        # Crear DataFrame con componentes KPCA
        kpca_columns = [f'KPCA{i+1}_{kernel}' for i in range(n_components)]
        df_kpca = pd.DataFrame(X_kpca, columns=kpca_columns, index=df.index)
        
        # Agregar variable objetivo
        df_kpca['Tipo_Tejido'] = y
        
        # Calcular varianza explicada aproximada solo para kernel lineal
        if kernel == 'linear' and hasattr(kpca, 'eigenvalues_'):
            lambdas = kpca.eigenvalues_
            if lambdas is not None and len(lambdas) > 0:
                variance_explained = lambdas / np.sum(lambdas)
                cumsum_var = np.cumsum(variance_explained)
                print(f"KPCA ({kernel}): {len(X.columns)} features -> {n_components} componentes")
                print(f"Varianza explicada: {cumsum_var[-1]:.3f}")
            else:
                print(f"KPCA ({kernel}): {len(X.columns)} features -> {n_components} componentes")
        else:
            print(f"KPCA ({kernel}): {len(X.columns)} features -> {n_components} componentes")
            print("(Varianza explicada no disponible para kernels no lineales)")
        
        return df_kpca
    

    def convert_numeric_simple(self, df):
        """
        Convierte todas las columnas del DataFrame a tipo numérico cuando sea posible.
        Reemplaza comas por puntos para decimales antes de la conversión.
        """

        # Crear copia para no modificar original
        df_converted = df.copy()

        # Convertir cada columna a numérica
        for column in df_converted.columns:
            # Reemplazar comas por puntos para decimales
            column_cleaned = df_converted[column].astype(str).str.replace(',', '.', regex=False)
            try:
                df_converted[column] = pd.to_numeric(column_cleaned)
            except (ValueError, TypeError):
                pass  # Mantener como está si no se puede convertir
            # print(f"Columna '{column}': {df_converted[column].dtype}")
        return df_converted

    def run_complete_pipeline(self):
        """Pipeline completo de selección de características"""

        try:
            print("=== Iniciando Pipeline de Selección de Características ===")
            # 1. Cargar datos procesados
            df = self.load_processed_data()
            if df is None:
                print("Error: No se pudieron cargar los datos procesados")
                return None

            # --- Procesamiento previo del DataFrame ---
            df = self.convert_numeric_simple(df)
            print("\nTipos de datos tras conversión:")
            print(df.dtypes)

            # Separar características y variable objetivo
            target_variable = 'Tipo_Tejido'
            if target_variable not in df.columns:
                print(f"No se encontró la columna objetivo '{target_variable}' en el DataFrame.")
                return None
            
            y = df[target_variable]
            X = df.drop(columns=[target_variable])

            non_numeric_cols_to_drop = X.select_dtypes(exclude=np.number).columns
            X = X.drop(columns=non_numeric_cols_to_drop)
            X = X.fillna(X.mean())

            # Reconstruir df para selección de features
            df_features = X.copy()
            df_features[target_variable] = y

            # 2. Selección por correlación
            print("\n--- Selección por correlación ---")
            corr_features = self.correlation_select(df_features)
            print(f"Features seleccionadas por correlación: {corr_features}")

            # 3. Selección por SelectKBest
            print("\n--- Selección por SelectKBest ---")
            kbest_features = self.select_kbest_features(df_features)
            print(f"Features seleccionadas por SelectKBest: {kbest_features}")

            # 4. Aplicar PCA y obtener dataset transformado
            print("\n--- Aplicando PCA ---")
            df_pca = self.select_pca_features(df_features)
            if df_pca is None:
                print("Error aplicando PCA")
                return None

            # 5. Aplicar KPCA con diferentes kernels
            print("\n--- Aplicando Kernel PCA (KPCA) ---")
            
            # KPCA con kernel RBF (Radial Basis Function - más común para datos complejos)
            print("\n[1/3] KPCA con kernel RBF...")
            df_kpca_rbf = self.select_kpca_features(df_features, kernel='rbf', gamma='scale', variance_threshold=0.98)
            
            # KPCA con kernel Polynomial (captura interacciones polinomiales)
            print("\n[2/3] KPCA con kernel Polynomial...")
            df_kpca_poly = self.select_kpca_features(df_features, kernel='poly', gamma='scale', variance_threshold=0.98, degree=3)
            
            # KPCA con kernel Sigmoid (similar a redes neuronales)
            print("\n[3/3] KPCA con kernel Sigmoid...")
            df_kpca_sigmoid = self.select_kpca_features(df_features, kernel='sigmoid', gamma='scale', variance_threshold=0.98)

            # Guardar resultados en carpeta best_features_TIMESTAMP
            now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_folder = Path("data/features") / f"best_features_{now_ts}"
            out_folder.mkdir(parents=True, exist_ok=True)

            # Guardar cada lista en un archivo .txt (fácil de leer y cargar)
            with open(out_folder / "correlation.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(corr_features))
            with open(out_folder / "kbest.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(kbest_features))
            
            # Guardar dataset con PCA aplicado
            pca_csv_path = out_folder / "pca_transformed.csv"
            df_pca.to_csv(pca_csv_path, sep=';', index=False)
            print(f"Dataset con PCA guardado en: {pca_csv_path}")

            # Guardar datasets con KPCA aplicados
            if df_kpca_rbf is not None:
                kpca_rbf_csv = out_folder / "kpca_rbf_transformed.csv"
                df_kpca_rbf.to_csv(kpca_rbf_csv, sep=';', index=False)
                print(f"Dataset con KPCA (RBF) guardado en: {kpca_rbf_csv}")
            
            if df_kpca_poly is not None:
                kpca_poly_csv = out_folder / "kpca_poly_transformed.csv"
                df_kpca_poly.to_csv(kpca_poly_csv, sep=';', index=False)
                print(f"Dataset con KPCA (Polynomial) guardado en: {kpca_poly_csv}")
            
            if df_kpca_sigmoid is not None:
                kpca_sigmoid_csv = out_folder / "kpca_sigmoid_transformed.csv"
                df_kpca_sigmoid.to_csv(kpca_sigmoid_csv, sep=';', index=False)
                print(f"Dataset con KPCA (Sigmoid) guardado en: {kpca_sigmoid_csv}")

            # Imprimir resumen final
            print(f"\nListas de mejores características guardadas en: {out_folder}")
            print("\n=== Selección de características completada ===")
            return {
                'correlation': corr_features,
                'kbest': kbest_features,
                'pca_dataframe': df_pca,
                'kpca_rbf_dataframe': df_kpca_rbf,
                'kpca_poly_dataframe': df_kpca_poly,
                'kpca_sigmoid_dataframe': df_kpca_sigmoid,
                'output_folder': str(out_folder)
            }
        except Exception as e:
            print(f"Error en pipeline completo: {e}")
    
def main():
    """Función principal para ejecución independiente"""

    # Configuración de argumentos
    parser = argparse.ArgumentParser(description='Entrenamiento ML Radiómico')
    parser.add_argument('--result-csv', '-r', default=None, help='Archivo CSV de resultados')
    
    args = parser.parse_args()
    
    # Crear y ejecutar pipeline
    ml_trainer = RadiomicsSelector(args.result_csv)
    
    print("Iniciando entrenamiento de modelo ML con radiómicas...")
    results = ml_trainer.run_complete_pipeline()

if __name__ == "__main__":
    exit(main())
