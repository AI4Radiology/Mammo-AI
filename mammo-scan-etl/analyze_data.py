"""
Ejemplo de uso del CSV de features para análisis y reentrenamiento
===================================================================

Este script muestra cómo utilizar el CSV generado por el sistema ETL
para análisis, validación y potencial reentrenamiento de modelos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class RadiomicsAnalyzer:
    """Analizador de features radiómicos del CSV generado por ETL"""
    
    def __init__(self, csv_path="data/radiomics_features.csv"):
        self.csv_path = Path(csv_path)
        self.df = None
        self.feature_columns = []
        self.metadata_columns = [
            'timestamp', 'patient_id', 'study_uid', 
            'accession_number', 'binary_prediction', 'multiclass_prediction'
        ]
    
    def load_data(self):
        """Cargar datos del CSV"""
        if not self.csv_path.exists():
            print(f"❌ Archivo no encontrado: {self.csv_path}")
            print("   El sistema aún no ha procesado ningún archivo DICOM.")
            return False
        
        self.df = pd.read_csv(self.csv_path)
        
        # Separar columnas de features
        self.feature_columns = [col for col in self.df.columns 
                               if col not in self.metadata_columns]
        
        print(f"✅ Datos cargados: {len(self.df)} registros")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Período: {self.df['timestamp'].min()} a {self.df['timestamp'].max()}")
        
        return True
    
    def show_summary(self):
        """Mostrar resumen de datos procesados"""
        if self.df is None:
            print("⚠️  Primero cargar datos con load_data()")
            return
        
        print("\n" + "="*60)
        print("RESUMEN DE DATOS PROCESADOS")
        print("="*60)
        
        # Estadísticas generales
        print(f"\n📊 Total de registros: {len(self.df)}")
        print(f"📅 Primer registro: {self.df['timestamp'].min()}")
        print(f"📅 Último registro: {self.df['timestamp'].max()}")
        
        # Pacientes únicos
        n_patients = self.df['patient_id'].nunique()
        print(f"👤 Pacientes únicos: {n_patients}")
        
        # Estudios únicos
        n_studies = self.df['study_uid'].nunique()
        print(f"🏥 Estudios únicos: {n_studies}")
        
        # Distribución de clasificaciones binarias
        print("\n🔵 Clasificación Binaria (Densidad):")
        binary_counts = self.df['binary_prediction'].value_counts()
        for pred, count in binary_counts.items():
            pct = (count / len(self.df)) * 100
            print(f"   {pred}: {count} ({pct:.1f}%)")
        
        # Distribución de clasificaciones multiclase
        print("\n🔵 Clasificación BI-RADS:")
        multi_counts = self.df['multiclass_prediction'].value_counts()
        for pred, count in multi_counts.items():
            pct = (count / len(self.df)) * 100
            print(f"   {pred}: {count} ({pct:.1f}%)")
    
    def analyze_features(self):
        """Analizar estadísticas de features radiómicos"""
        if self.df is None:
            print("⚠️  Primero cargar datos con load_data()")
            return
        
        print("\n" + "="*60)
        print("ANÁLISIS DE FEATURES RADIÓMICOS")
        print("="*60)
        
        # Estadísticas básicas
        features_df = self.df[self.feature_columns]
        
        print(f"\n📈 Estadísticas de {len(self.feature_columns)} features:")
        print(f"   Media: {features_df.mean().mean():.4f}")
        print(f"   Desviación estándar: {features_df.std().mean():.4f}")
        print(f"   Mínimo: {features_df.min().min():.4f}")
        print(f"   Máximo: {features_df.max().max():.4f}")
        
        # Features con mayor variabilidad
        print("\n📊 Top 10 features con mayor variabilidad:")
        feature_std = features_df.std().sort_values(ascending=False).head(10)
        for feat, std in feature_std.items():
            print(f"   {feat}: {std:.4f}")
        
        # Valores nulos
        null_counts = features_df.isnull().sum()
        if null_counts.sum() > 0:
            print("\n⚠️  Features con valores nulos:")
            for feat, count in null_counts[null_counts > 0].items():
                print(f"   {feat}: {count}")
        else:
            print("\n✅ Sin valores nulos en features")
    
    def check_data_quality(self):
        """Verificar calidad de datos para reentrenamiento"""
        if self.df is None:
            print("⚠️  Primero cargar datos con load_data()")
            return
        
        print("\n" + "="*60)
        print("VERIFICACIÓN DE CALIDAD DE DATOS")
        print("="*60)
        
        issues = []
        
        # 1. Verificar mínimo de registros
        min_records = 100
        if len(self.df) < min_records:
            issues.append(f"⚠️  Pocos registros: {len(self.df)} < {min_records}")
        else:
            print(f"✅ Cantidad de registros adecuada: {len(self.df)}")
        
        # 2. Verificar balance de clases (binario)
        binary_counts = self.df['binary_prediction'].value_counts()
        min_class = binary_counts.min()
        max_class = binary_counts.max()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        if imbalance_ratio > 3:
            issues.append(f"⚠️  Desbalance en clasificación binaria: {imbalance_ratio:.1f}:1")
        else:
            print(f"✅ Balance aceptable en clasificación binaria: {imbalance_ratio:.1f}:1")
        
        # 3. Verificar balance de clases (multiclase)
        multi_counts = self.df['multiclass_prediction'].value_counts()
        min_multi = multi_counts.min()
        max_multi = multi_counts.max()
        multi_imbalance = max_multi / min_multi if min_multi > 0 else float('inf')
        
        if multi_imbalance > 5:
            issues.append(f"⚠️  Desbalance en clasificación BI-RADS: {multi_imbalance:.1f}:1")
        else:
            print(f"✅ Balance aceptable en clasificación BI-RADS: {multi_imbalance:.1f}:1")
        
        # 4. Verificar valores nulos
        features_df = self.df[self.feature_columns]
        null_pct = (features_df.isnull().sum().sum() / features_df.size) * 100
        
        if null_pct > 5:
            issues.append(f"⚠️  Alto porcentaje de valores nulos: {null_pct:.2f}%")
        else:
            print(f"✅ Porcentaje de valores nulos bajo: {null_pct:.2f}%")
        
        # 5. Verificar valores infinitos
        inf_count = np.isinf(features_df.values).sum()
        if inf_count > 0:
            issues.append(f"⚠️  Valores infinitos detectados: {inf_count}")
        else:
            print("✅ Sin valores infinitos")
        
        # Resumen
        if issues:
            print("\n⚠️  PROBLEMAS DE CALIDAD DETECTADOS:")
            for issue in issues:
                print(f"   {issue}")
            print("\n   Resolver estos problemas antes de reentrenar modelos.")
        else:
            print("\n✅ DATOS LISTOS PARA REENTRENAMIENTO")
    
    def export_for_training(self, output_path="data/training_data.csv"):
        """Exportar datos preparados para entrenamiento"""
        if self.df is None:
            print("⚠️  Primero cargar datos con load_data()")
            return
        
        # Preparar datos: features + labels
        training_data = self.df[self.feature_columns + ['binary_prediction', 'multiclass_prediction']].copy()
        
        # Eliminar filas con valores nulos
        training_data = training_data.dropna()
        
        # Guardar
        output_path = Path(output_path)
        training_data.to_csv(output_path, index=False)
        
        print(f"\n✅ Datos exportados para entrenamiento: {output_path}")
        print(f"   Registros: {len(training_data)}")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Labels: binary_prediction, multiclass_prediction")
    
    def generate_report(self, output_path="data/analysis_report.txt"):
        """Generar reporte completo de análisis"""
        if self.df is None:
            print("⚠️  Primero cargar datos con load_data()")
            return
        
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE DE ANÁLISIS - MAMMO-SCAN ETL\n")
            f.write("="*70 + "\n")
            f.write(f"Fecha de generación: {datetime.now()}\n")
            f.write(f"Archivo analizado: {self.csv_path}\n")
            f.write("\n")
            
            # Estadísticas generales
            f.write("-"*70 + "\n")
            f.write("ESTADÍSTICAS GENERALES\n")
            f.write("-"*70 + "\n")
            f.write(f"Total de registros: {len(self.df)}\n")
            f.write(f"Pacientes únicos: {self.df['patient_id'].nunique()}\n")
            f.write(f"Estudios únicos: {self.df['study_uid'].nunique()}\n")
            f.write(f"Período: {self.df['timestamp'].min()} a {self.df['timestamp'].max()}\n")
            f.write("\n")
            
            # Clasificación binaria
            f.write("-"*70 + "\n")
            f.write("CLASIFICACIÓN BINARIA (Densidad)\n")
            f.write("-"*70 + "\n")
            binary_counts = self.df['binary_prediction'].value_counts()
            for pred, count in binary_counts.items():
                pct = (count / len(self.df)) * 100
                f.write(f"{pred}: {count} ({pct:.1f}%)\n")
            f.write("\n")
            
            # Clasificación multiclase
            f.write("-"*70 + "\n")
            f.write("CLASIFICACIÓN BI-RADS\n")
            f.write("-"*70 + "\n")
            multi_counts = self.df['multiclass_prediction'].value_counts()
            for pred, count in multi_counts.items():
                pct = (count / len(self.df)) * 100
                f.write(f"{pred}: {count} ({pct:.1f}%)\n")
            f.write("\n")
            
            # Features
            f.write("-"*70 + "\n")
            f.write(f"FEATURES RADIÓMICOS ({len(self.feature_columns)} total)\n")
            f.write("-"*70 + "\n")
            features_df = self.df[self.feature_columns]
            f.write(f"Media: {features_df.mean().mean():.4f}\n")
            f.write(f"Desviación estándar: {features_df.std().mean():.4f}\n")
            f.write(f"Mínimo: {features_df.min().min():.4f}\n")
            f.write(f"Máximo: {features_df.max().max():.4f}\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
        
        print(f"\n✅ Reporte generado: {output_path}")


def main():
    """Función principal de ejemplo"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "ANÁLISIS DE DATOS - MAMMO-SCAN ETL" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Crear analizador
    analyzer = RadiomicsAnalyzer()
    
    # Cargar datos
    if not analyzer.load_data():
        return
    
    # Mostrar resumen
    analyzer.show_summary()
    
    # Analizar features
    analyzer.analyze_features()
    
    # Verificar calidad
    analyzer.check_data_quality()
    
    # Exportar para entrenamiento
    # analyzer.export_for_training()
    
    # Generar reporte
    # analyzer.generate_report()
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)
    print("\nPara exportar datos para reentrenamiento:")
    print("  analyzer.export_for_training('data/training_data.csv')")
    print("\nPara generar reporte completo:")
    print("  analyzer.generate_report('data/analysis_report.txt')")
    print()


if __name__ == "__main__":
    main()
