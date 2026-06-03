import sqlite3
import json
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_VIAB, FEATURES
from services.ml_engine import MLEngine
from services.data_processor import DataProcessor

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluations.db")


class FeedbackService:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingreso_mensual REAL,
                total_gastos REAL,
                total_costos REAL,
                total_activos REAL,
                total_deudas REAL,
                num_deudas INTEGER,
                excedente REAL,
                capacidad_pago REAL,
                endeudamiento_patrimonial REAL,
                capital_trabajo REAL,
                monto_solicitado REAL,
                tem REAL,
                num_cuotas INTEGER,
                cuota_estimada REAL,
                monto_propuesto REAL,
                mora_diaria REAL,
                prediction_label TEXT,
                prediction_probability REAL,
                total_score INTEGER,
                classification TEXT,
                final_decision TEXT,
                true_label INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_evaluation(self, client_data: dict, prediction: dict,
                        score_data: dict, recommendations: dict) -> int:
        flat = {k: client_data.get(k) for k in FEATURES}
        flat["prediction_label"] = prediction.get("label")
        flat["prediction_probability"] = prediction.get("confidence", 0) / 100.0
        flat["total_score"] = score_data.get("total_score", 0)
        flat["classification"] = score_data.get("classification", "")
        flat["final_decision"] = recommendations.get("final_decision", "")
        flat["true_label"] = None
        flat["created_at"] = datetime.now().isoformat()
        flat["updated_at"] = datetime.now().isoformat()

        conn = sqlite3.connect(DB_PATH)
        cols = ", ".join(flat.keys())
        placeholders = ", ".join("?" for _ in flat)
        sql = f"INSERT INTO evaluations ({cols}) VALUES ({placeholders})"
        cur = conn.cursor()
        cur.execute(sql, list(flat.values()))
        eval_id = cur.lastrowid
        conn.commit()
        conn.close()
        return eval_id

    def get_evaluation_count(self) -> int:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM evaluations")
        count = cur.fetchone()[0]
        conn.close()
        return count

    def get_labeled_count(self) -> int:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM evaluations WHERE true_label IS NOT NULL")
        count = cur.fetchone()[0]
        conn.close()
        return count

    def get_all_evaluations(self, limit: int = 100) -> list:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def update_true_label(self, eval_id: int, true_label: int) -> bool:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "UPDATE evaluations SET true_label = ?, updated_at = ? WHERE id = ?",
            (true_label, datetime.now().isoformat(), eval_id)
        )
        ok = cur.rowcount > 0
        conn.commit()
        conn.close()
        return ok

    def delete_evaluation(self, eval_id: int) -> bool:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM evaluations WHERE id = ?", (eval_id,))
        ok = cur.rowcount > 0
        conn.commit()
        conn.close()
        return ok

    def retrain_with_feedback(self) -> dict:
        df_orig = pd.read_csv(str(DATASET_VIAB), encoding='utf-8-sig')
        conn = sqlite3.connect(DB_PATH)
        df_feedback = pd.read_sql_query(
            "SELECT * FROM evaluations WHERE true_label IS NOT NULL", conn
        )
        conn.close()

        if df_feedback.empty:
            return {"message": "No hay evaluaciones etiquetadas para reentrenar",
                    "total_original": len(df_orig), "feedback_added": 0}

        fb = df_feedback[FEATURES].copy()
        fb["viable"] = df_feedback["true_label"].values
        df_combined = pd.concat([df_orig, fb], ignore_index=True)

        ml = MLEngine()
        metrics = ml.train_all(df_combined, target_col="viable")
        shap_vals = ml.compute_shap(df_combined, target_col="viable")
        ml.save_best_model()

        return {
            "message": "Modelo reentrenado con nuevos datos",
            "total_original": len(df_orig),
            "feedback_added": len(fb),
            "total_combined": len(df_combined),
            "best_model": ml.best_model_name,
            "metrics": metrics,
        }
