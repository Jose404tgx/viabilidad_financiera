from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASET_EVAL, DATASET_VIAB
from services.ml_engine import MLEngine
from services.scoring_engine import ScoringEngine
from services.recommendation_engine import RecommendationEngine
from services.simulation_engine import SimulationEngine
from services.report_engine import ReportEngine
from services.data_processor import DataProcessor
from services.feedback_service import FeedbackService
from services.additional_algorithms import AdditionalAlgorithms

app = FastAPI(
    title="Predictor de Viabilidad Financiera",
    description="Sistema de Inteligencia Artificial para la predicción de viabilidad crediticia. Proyecto universitario.",
    version="2.0.0",
    contact={
        "name": "Proyecto Universitario - Sustentación Semestral",
        "url": "https://github.com/"
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
ml_engine = MLEngine()
ml_engine.load_best_model()
scoring_engine = ScoringEngine()
recommendation_engine = RecommendationEngine()
simulation_engine = SimulationEngine()
report_engine = ReportEngine()
data_processor = DataProcessor()
feedback_service = FeedbackService()
additional_algorithms = AdditionalAlgorithms()

# Load datasets
try:
    df_viabilidad = pd.read_csv(str(DATASET_VIAB), encoding='utf-8-sig')
    df_evaluaciones = pd.read_csv(str(DATASET_EVAL))
except Exception as e:
    print(f"Warning: Could not load datasets: {e}")
    df_viabilidad = pd.DataFrame()
    df_evaluaciones = pd.DataFrame()


# ===== Pydantic Models =====

class ClientData(BaseModel):
    id_cliente: Optional[int] = None
    ingreso_mensual: float = Field(..., gt=0)
    total_gastos: float = Field(..., ge=0)
    total_costos: float = Field(..., ge=0)
    total_activos: float = Field(..., ge=0)
    total_deudas: float = Field(..., ge=0)
    num_deudas: int = Field(..., ge=0)
    excedente: float = 0
    capacidad_pago: float = 0
    endeudamiento_patrimonial: float = 0
    capital_trabajo: float = 0
    monto_solicitado: float = Field(..., ge=0)
    tem: float = Field(..., ge=0, le=1)
    num_cuotas: int = Field(..., ge=1, le=120)
    cuota_estimada: float = 0
    monto_propuesto: float = 0
    mora_diaria: float = 0

class BatchPredictRequest(BaseModel):
    clients: List[ClientData]

class SimulationRequest(BaseModel):
    monto_solicitado: float = 10000
    tem: float = 0.03
    num_cuotas: int = 12
    cuota_estimada: Optional[float] = 500
    ingreso_mensual: Optional[float] = 5000
    total_gastos: Optional[float] = 3000
    total_deudas: Optional[float] = 10000
    mora_diaria: Optional[float] = 0


# ===== API Endpoints =====

@app.get("/api/v1/health")
def health_check():
    """Health check del sistema."""
    model_loaded = ml_engine.best_model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_name": ml_engine.best_model_name,
        "dataset_size": len(df_viabilidad),
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/api/v1/predict")
def predict(client: ClientData):
    """Predice la viabilidad financiera de un cliente y retorna score + recomendaciones."""
    try:
        client_dict = client.model_dump()
        prediction = ml_engine.predict(client_dict)
        score_data = scoring_engine.calculate_score(client_dict)
        score_data = scoring_engine.adjust_with_ml(score_data, prediction["probability"])
        recommendations = recommendation_engine.generate(prediction, score_data, client_dict)

        feedback_service.save_evaluation(client_dict, prediction, score_data, recommendations)

        return {
            "success": True,
            "prediction": prediction,
            "scoring": score_data,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/score")
def get_score(client: ClientData):
    """Calcula el score financiero de un cliente."""
    try:
        client_dict = client.model_dump()
        score_data = scoring_engine.calculate_score(client_dict)
        prediction = ml_engine.predict(client_dict)
        score_data = scoring_engine.adjust_with_ml(score_data, prediction["probability"])
        return {"success": True, "scoring": score_data, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/batch")
def predict_batch(request: BatchPredictRequest):
    """Predicción por lote para múltiples clientes."""
    results = []
    for client in request.clients:
        try:
            client_dict = client.model_dump()
            prediction = ml_engine.predict(client_dict)
            score_data = scoring_engine.calculate_score(client_dict)
            score_data = scoring_engine.adjust_with_ml(score_data, prediction["probability"])
            recommendations = recommendation_engine.generate(prediction, score_data, client_dict)
            results.append({
                "client_id": client_dict.get("id_cliente"),
                "prediction": prediction,
                "scoring": score_data,
                "recommendations": recommendations,
            })
        except Exception as e:
            results.append({"client_id": client.id_cliente, "error": str(e)})
    return {"success": True, "results": results, "total": len(results)}

@app.post("/api/v1/recommend")
def get_recommendations(client: ClientData):
    """Genera recomendaciones basadas en el perfil del cliente."""
    try:
        client_dict = client.model_dump()
        prediction = ml_engine.predict(client_dict)
        score_data = scoring_engine.calculate_score(client_dict)
        score_data = scoring_engine.adjust_with_ml(score_data, prediction["probability"])
        recommendations = recommendation_engine.generate(prediction, score_data, client_dict)
        return {"success": True, "recommendations": recommendations, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/simulate/loan")
def simulate_loan(request: SimulationRequest):
    """Simula condiciones de préstamo."""
    try:
        result = simulation_engine.simulate_loan(request.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/simulate/risk")
def simulate_risk(request: SimulationRequest):
    """Simula escenarios de riesgo."""
    try:
        result = simulation_engine.simulate_risk_scenarios(request.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/simulate/mora")
def simulate_mora(request: SimulationRequest):
    """Simula impacto de mora."""
    try:
        result = simulation_engine.simulate_mora(request.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/simulate/amortization")
def simulate_amortization(request: SimulationRequest):
    """Simula tabla de amortización."""
    try:
        result = simulation_engine.simulate_amortization(request.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/report/evaluation")
def generate_evaluation_report(client: ClientData):
    """Genera reporte PDF de evaluación."""
    try:
        client_dict = client.model_dump()
        prediction = ml_engine.predict(client_dict)
        score_data = scoring_engine.calculate_score(client_dict)
        score_data = scoring_engine.adjust_with_ml(score_data, prediction["probability"])
        recommendations = recommendation_engine.generate(prediction, score_data, client_dict)
        report = report_engine.generate_evaluation_report(client_dict, prediction, score_data, recommendations)
        return {"success": True, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dashboard")
def get_dashboard():
    """Retorna KPIs y estadísticas del dataset."""
    try:
        df = df_viabilidad.copy()
        if df.empty:
            return {"success": True, "data": {"kpis": {}, "charts": {}, "recent_predictions": []}}

        total = len(df)
        viable_count = int(df["viable"].sum())
        non_viable = total - viable_count
        approval_rate = round(viable_count / total * 100, 2)

        avg_score = int(scoring_engine.calculate_score({
            "ingreso_mensual": df["ingreso_mensual"].mean(),
            "total_gastos": df["total_gastos"].mean(),
            "total_activos": df["total_activos"].mean(),
            "total_deudas": df["total_deudas"].mean(),
            "excedente": df["excedente"].mean(),
            "mora_diaria": df["mora_diaria"].mean(),
            "num_deudas": df["num_deudas"].mean(),
            "capacidad_pago": df["capacidad_pago"].mean(),
            "endeudamiento_patrimonial": df["endeudamiento_patrimonial"].mean(),
            "capital_trabajo": df["capital_trabajo"].mean(),
            "monto_solicitado": df["monto_solicitado"].mean(),
            "total_costos": df["total_costos"].mean(),
            "tem": df["tem"].mean(),
            "num_cuotas": df["num_cuotas"].mean(),
            "cuota_estimada": df["cuota_estimada"].mean(),
            "monto_propuesto": df["monto_propuesto"].mean(),
        })["total_score"])

        score_dist = {"excelente": 0, "bueno": 0, "regular": 0, "riesgoso": 0, "critico": 0}
        for _, row in df.iterrows():
            sc = scoring_engine.calculate_score(row.to_dict())
            cat = sc["classification"]
            if cat in score_dist:
                score_dist[cat] += 1

        return {
            "success": True,
            "data": {
                "kpis": {
                    "total_clientes": total,
                    "viables": viable_count,
                    "no_viables": non_viable,
                    "tasa_aprobacion": approval_rate,
                    "score_promedio": avg_score,
                    "riesgo_promedio": round(100 - approval_rate, 2),
                    "morosidad": round(float(df["mora_diaria"].mean() > 15) * 100, 2),
                },
                "charts": {
                    "score_distribution": score_dist,
                    "viable_ratio": {"Viables": viable_count, "No Viables": non_viable},
                    "avg_income_by_result": {
                        "Viables": round(df[df["viable"]==1]["ingreso_mensual"].mean(), 2),
                        "No Viables": round(df[df["viable"]==0]["ingreso_mensual"].mean(), 2),
                    },
                    "avg_debt_by_result": {
                        "Viables": round(df[df["viable"]==1]["total_deudas"].mean(), 2),
                        "No Viables": round(df[df["viable"]==0]["total_deudas"].mean(), 2),
                    },
                },
                "recent_predictions": [],
                "timestamp": datetime.now().isoformat(),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/admin/model-info")
def get_model_info():
    """Información detallada del modelo ML entrenado."""
    return {"success": True, "data": ml_engine.get_model_summary()}

@app.get("/api/v1/admin/model-metrics/{model_name}")
def get_model_metrics(model_name: str):
    """Métricas específicas de un modelo."""
    cm = ml_engine.get_confusion_matrix(model_name)
    return {"success": True, "data": {"confusion_matrix": cm, "metrics": ml_engine.metrics}}

@app.get("/api/v1/admin/feature-importance")
def get_feature_importance():
    """Importancia de características del modelo."""
    return {"success": True, "data": ml_engine.feature_importance}

@app.get("/api/v1/admin/shap-values")
def get_shap_values():
    """Valores SHAP para interpretabilidad del modelo."""
    return {"success": True, "data": ml_engine.shap_values}

@app.post("/api/v1/admin/retrain")
def retrain_model():
    """Reentrena todos los modelos desde cero."""
    try:
        from services.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_full_training()
        ml_engine.load_best_model()
        return {"success": True, "message": "Modelo reentrenado exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/admin/feedback/count")
def feedback_count():
    """Cantidad de evaluaciones y retroalimentación."""
    return {
        "success": True,
        "data": {
            "total_evaluations": feedback_service.get_evaluation_count(),
            "labeled_evaluations": feedback_service.get_labeled_count(),
        }
    }

@app.get("/api/v1/admin/feedback/list")
def feedback_list(limit: int = 100):
    """Lista de evaluaciones guardadas."""
    try:
        items = feedback_service.get_all_evaluations(limit)
        return {"success": True, "data": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/admin/feedback/label/{eval_id}/{true_label}")
def feedback_label(eval_id: int, true_label: int):
    """Actualiza la etiqueta real de una evaluación."""
    if true_label not in (0, 1):
        raise HTTPException(status_code=400, detail="true_label must be 0 or 1")
    ok = feedback_service.update_true_label(eval_id, true_label)
    if not ok:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return {"success": True, "message": "Etiqueta actualizada"}

@app.delete("/api/v1/admin/feedback/{eval_id}")
def feedback_delete(eval_id: int):
    """Elimina una evaluación."""
    ok = feedback_service.delete_evaluation(eval_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return {"success": True, "message": "Evaluacion eliminada"}

@app.post("/api/v1/admin/retrain-with-feedback")
def retrain_with_feedback():
    """Reentrena el modelo incluyendo feedback etiquetado."""
    try:
        result = feedback_service.retrain_with_feedback()
        ml_engine.load_best_model()
        result["success"] = True
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dataset/info")
def get_dataset_info():
    """Información y estadísticas del dataset."""
    try:
        df = df_viabilidad
        return {
            "success": True,
            "data": {
                "rows": len(df),
                "columns": list(df.columns),
                "target_distribution": df["viable"].value_counts().to_dict(),
                "correlations": df.select_dtypes(include=[np.number]).corr()["viable"].to_dict(),
                "stats": df.describe().to_dict(),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/algorithms/linear-regression")
def linear_regression_analysis():
    """Regresión Lineal con métrica MSE sobre el dataset."""
    try:
        df = df_viabilidad.copy()
        df = data_processor.load_and_clean(df)
        df = data_processor.engineer_features(df)
        feature_cols = [c for c in df.columns if c != "viable"][:10]
        result = additional_algorithms.train_linear_regression(df, feature_cols)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/algorithms/apriori")
def apriori_analysis(min_support: float = 0.1, min_confidence: float = 0.5):
    """Algoritmo Apriori: reglas de asociación entre variables financieras."""
    try:
        df = df_viabilidad.copy()
        df = data_processor.load_and_clean(df)
        df = data_processor.engineer_features(df)
        result = additional_algorithms.apriori_analysis(df, min_support, min_confidence)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/algorithms/nn-architecture")
def nn_architecture():
    """Arquitectura de la Red Neuronal (MLP): capas, neuronas, pesos, sesgos, Adam."""
    try:
        nn_model = None
        if "Neural Network" in ml_engine.models and "error" not in ml_engine.models["Neural Network"]:
            nn_model = ml_engine.models["Neural Network"].get("model")
        result = additional_algorithms.get_nn_architecture(nn_model)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/algorithms/list")
def list_algorithms():
    """Lista todos los algoritmos implementados en el proyecto."""
    return {
        "success": True,
        "data": {
            "clasificacion": [
                "Random Forest (Bagging - Árboles de decisión)",
                "KNN (K-Nearest Neighbors - Distancia Euclidiana)",
                "Decision Tree ID3 (Criterio Entropía - Ganancia de Información)",
                "XGBoost (Gradient Boosting - Árboles potenciados)",
                "LightGBM (Gradient Boosting - Hojas eficientes)",
                "CatBoost (Gradient Boosting - Categórico)",
                "Gradient Boosting (Scikit-learn)",
                "Logistic Regression (Regresión Logística)",
                "AdaBoost (Adaptive Boosting)",
                "Ensemble Stacking (Meta-aprendizaje con Regresión Logística)",
            ],
            "regresion": [
                "Linear Regression (Mínimos Cuadrados Ordinarios - MSE)",
            ],
            "no_supervisado": [
                "K-Means (Clustering de clientes por perfil financiero)",
            ],
            "asociacion": [
                "Apriori (Reglas de Asociación - Support, Confidence, Lift)",
            ],
            "deep_learning": [
                "MLPClassifier (Perceptrón Multicapa - Optimizador Adam)",
            ],
            "metricas": [
                "Accuracy (Precisión global)",
                "Precision (Precisión por clase)",
                "Recall (Sensibilidad / Exhaustividad)",
                "F1-Score (Media armónica Precision-Recall)",
                "ROC-AUC (Área bajo la curva ROC)",
                "MSE (Mean Squared Error - Error cuadrático medio)",
                "R² Score (Coeficiente de determinación)",
                "Matriz de Confusión",
            ],
            "librerias": [
                "numpy (Cálculos numéricos)",
                "pandas (Manipulación de datos)",
                "scikit-learn (Modelos ML, métricas, preprocesamiento)",
                "imbalanced-learn (SMOTE - Balanceo de clases)",
                "xgboost (XGBoost)",
                "lightgbm (LightGBM)",
                "catboost (CatBoost)",
                "shap (Interpretabilidad - SHAP Values)",
                "joblib (Persistencia de modelos)",
                "matplotlib / plotly (Visualización)",
            ],
        }
    }

# Mount frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
