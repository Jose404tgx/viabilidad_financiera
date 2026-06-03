import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_EVAL, DATASET_VIAB, FEATURES
from services.ml_engine import MLEngine
from services.data_processor import DataProcessor


class TrainingPipeline:
    def __init__(self):
        self.ml_engine = MLEngine()
        self.processor = DataProcessor()

    def run_full_training(self):
        results = {}

        print("=" * 60)
        print("FASE 1: Entrenamiento en dataset VIABILIDAD BINARIO")
        print("=" * 60)
        df_viab = pd.read_csv(str(DATASET_VIAB), encoding='utf-8-sig')
        print(f"Dataset cargado: {df_viab.shape[0]} registros, {df_viab.shape[1]} columnas")
        print(f"Distribucion target: {df_viab['viable'].value_counts().to_dict()}")

        metrics_viab = self.ml_engine.train_all(df_viab, target_col="viable")
        results["viabilidad"] = self.ml_engine.get_model_summary()

        print(f"\nMejor modelo: {self.ml_engine.best_model_name}")
        print("Metricas:")
        for m in metrics_viab:
            if "error" not in m:
                print(f"  {m['model']}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}, ROC-AUC={m['roc_auc']:.4f}")

        print("\nCalculando SHAP values...")
        shap_vals = self.ml_engine.compute_shap(df_viab, target_col="viable")
        if shap_vals:
            print("Top 5 variables SHAP:")
            for i, (k, v) in enumerate(sorted(shap_vals.items(), key=lambda x: x[1], reverse=True)[:5]):
                print(f"  {i+1}. {k}: {v:.4f}")

        return results

    def run_evaluation_training(self):
        print("\n" + "=" * 60)
        print("FASE 2: Entrenamiento en dataset EVALUACIONES")
        print("=" * 60)
        df_eval = pd.read_csv(str(DATASET_EVAL))

        eval_results = {}

        for target in ["resultado_aprobado", "resultado_rechazado", "resultado_riesgo_medio"]:
            print(f"\n--- Entrenando para target: {target} ---")
            ml = MLEngine()
            metrics = ml.train_all(df_eval, target_col=target)
            eval_results[target] = ml.get_model_summary()
            print(f"Mejor modelo: {ml.best_model_name}")
            for m in metrics:
                if "error" not in m:
                    print(f"  {m['model']}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}, ROC-AUC={m['roc_auc']:.4f}")

        return eval_results


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run_full_training()
    eval_results = pipeline.run_evaluation_training()
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
