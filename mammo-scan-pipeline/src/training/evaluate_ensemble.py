import argparse
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from .base_trainer import BaseTrainer


class EnsembleHierarchicalTrainer(BaseTrainer):

    def __init__(self, config_path):
        """Inicializa el entrenador del ensemble jerárquico cargando la configuración desde un archivo YAML."""

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)['ensemble_settings']
        super().__init__(config)
        self.models = {}
        self.scalers = {}
    
    def load_pretrained_models(self):
        """Carga los modelos pre-entrenados y sus escaladores desde las rutas especificadas en la configuración."""

        print(f"\n{'='*60}\nCARGANDO MODELOS PRE-ENTRENADOS\n{'='*60}")

        # Cargar modelos y escaladores para cada nivel jerárquico
        for level_name, level_config in self.config['hierarchy'].items():
            model_path = Path(level_config['model_path'])
            model_data = joblib.load(model_path)
            self.models[level_name] = model_data['model']
            self.scalers[level_name] = model_data['scaler']
            print(f"✓ {level_name}: {model_path}")
        print(f"{'='*60}\n")
    
    def load_and_split_data(self):
        """Carga los datos de prueba y los divide en conjuntos de entrenamiento y prueba."""

        # Cargar datos de prueba
        df = self.load_data(self.config['data']['test_data_path'])
        target_col = self.config['preprocessing']['target_variable']
        
        # Filtrar características y variable objetivo si se especifica un archivo de características
        if self.config['data'].get('features_path'):
            features = self.load_features(self.config['data']['features_path'])
            df = df[[c for c in features if c in df.columns] + [target_col]]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Dividir datos en conjuntos de entrenamiento y prueba
        _, self.X_test, _, self.y_test = train_test_split(
            X, y,
            test_size=self.config['data_split']['test_size'],
            random_state=self.config['data_split']['random_state'],
            stratify=y if self.config['data_split']['stratify'] else None
        )
        print(f"Test: {len(self.X_test)} muestras, {self.X_test.shape[1]} features")
    
    def predict_hierarchical(self, X):
        """Realiza predicciones utilizando el ensemble jerárquico."""

        # Nivel 1: Clasificación inicial
        pred_l1 = self.models['level1'].predict(self.scalers['level1'].transform(X))
        final_pred = np.empty(len(X), dtype=int)
        
        # Nivel 2: Clasificación especializada según la predicción del nivel 1
        mask_denso = pred_l1 == self.config['class_mapping']['denso_label']
        mask_nodenso = pred_l1 == self.config['class_mapping']['nodenso_label']
        
        # Predicciones para cada submodelo del nivel 2
        if mask_denso.sum() > 0:
            final_pred[mask_denso] = self.models['level2a'].predict(
                self.scalers['level2a'].transform(X[mask_denso]))

        if mask_nodenso.sum() > 0:
            final_pred[mask_nodenso] = self.models['level2b'].predict(
                self.scalers['level2b'].transform(X[mask_nodenso])) + 2
        
        return final_pred
    
    def train_and_evaluate(self):
        """Evalúa el ensemble jerárquico en el conjunto de prueba y calcula métricas de rendimiento."""

        # Cargar modelos pre-entrenados y datos spliteados
        self.load_pretrained_models()
        self.load_and_split_data()
        
        # Realizar predicciones y carga configuración de métricas
        y_pred = self.predict_hierarchical(self.X_test)
        cfg = self.config['metrics']
        
        # Calcular y retornar métricas de evaluación
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average=cfg['precision_average'], zero_division=0),
            'recall': recall_score(self.y_test, y_pred, average=cfg['recall_average'], zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, average=cfg['f1_average'], zero_division=0),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred, 
                target_names=self.config['class_mapping']['final_classes'], zero_division=0),
            'n_samples': self.X_test.shape[0],
            'n_features': self.X_test.shape[1],
            'feature_names': list(self.X_test.columns),
            'model_paths': {k: v['model_path'] for k, v in self.config['hierarchy'].items()},
            'train_accuracy': 0.0,
            'overfit_difference': 0.0,
            'model': None
        }
    
    def get_model(self):
        return None
    
    def _save_model_specific(self, results, model_dir, base_name):
        """Guarda la configuración del ensemble jerárquico en un archivo pickle."""

        # Establece la ruta para guardar la configuración del ensemble y la guarda
        ensemble_path = model_dir / f"{base_name}_config.pkl"

        # Guardar configuración del ensemble
        joblib.dump({
            'model_paths': results['model_paths'],
            'config': self.config,
            'feature_names': results['feature_names'],
            'evaluation_date': pd.Timestamp.now().isoformat()
        }, ensemble_path)
        print(f"Config guardada: {ensemble_path}")
    
    def _add_extra_results_to_json(self, results, json_results):
        """Agrega información adicional específica del ensemble jerárquico al JSON de resultados."""

        json_results['ensemble_type'] = 'hierarchical'
        json_results['model_paths'] = results['model_paths']
    
    def _print_extra_results(self, results):
        pass
    
    def print_results(self, results):
        """Imprime los resultados de evaluación del ensemble jerárquico."""

        print(f"\n{'='*60}\nRESULTADOS ENSEMBLE\n{'='*60}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        print(f"\nMatriz de Confusión:\n{results['confusion_matrix']}")
        print(f"\n{results['classification_report']}")
        print(f"{'='*60}")


def main():

    # Configuración de argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ensemble_config.yaml')
    parser.add_argument('--save-results', action='store_true')
    args = parser.parse_args()
    
    # LLamado de entrenador y evaluación
    trainer = EnsembleHierarchicalTrainer(args.config)
    results = trainer.train_and_evaluate()
    trainer.print_results(results)
    
    if args.save_results:
        trainer.save_model(results, model_prefix='ensemble')


if __name__ == "__main__":
    main()


