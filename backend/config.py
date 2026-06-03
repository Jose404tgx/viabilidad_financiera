import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_EVAL = BASE_DIR.parent / "dataset_evaluaciones.csv"
DATASET_VIAB = BASE_DIR.parent / "dataset_viabilidad_binario.csv"
MODELS_DIR = BASE_DIR / "models_ml"

SECRET_KEY = "FinPredict-Pro-SecureKey-2026-x9k2m7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480

DB_URL = "sqlite:///./financial_predictor.db"

# Scoring thresholds
SCORING_RANGES = {
    "excelente": (800, 1000),
    "bueno": (650, 799),
    "regular": (450, 649),
    "riesgoso": (250, 449),
    "critico": (0, 249),
}

# Risk levels
RISK_LEVELS = {
    "bajo": (0.0, 0.25),
    "moderado": (0.25, 0.50),
    "alto": (0.50, 0.75),
    "critico": (0.75, 1.0),
}

FEATURES = [
    "ingreso_mensual", "total_gastos", "total_costos", "total_activos",
    "total_deudas", "num_deudas", "excedente", "capacidad_pago",
    "endeudamiento_patrimonial", "capital_trabajo", "monto_solicitado",
    "tem", "num_cuotas", "cuota_estimada", "monto_propuesto", "mora_diaria"
]

TARGET_BINARIO = "viable"
TARGET_APROBADO = "resultado_aprobado"
TARGET_RECHAZADO = "resultado_rechazado"
TARGET_RIESGO = "resultado_riesgo_medio"
