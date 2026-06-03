import numpy as np
from typing import Any, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCORING_RANGES


class ScoringEngine:
    """
    Motor de scoring financiero.

    Evalúa 9 dimensiones clave del perfil financiero del cliente
    y genera un puntaje compuesto de 0 a 1000 puntos, con
    clasificación automática (excelente, bueno, regular, critico).

    Dimensiones evaluadas:
        - Ingreso mensual (0-150)
        - Ratio gastos/ingresos (0-100)
        - Nivel de deuda (0-150)
        - Excedente mensual (0-120)
        - Capacidad de pago (0-100)
        - Número de deudas (0-80)
        - Mora diaria (0-100)
        - Endeudamiento patrimonial (0-100)
        - Capital de trabajo (0-100)
    """

    def __init__(self) -> None:
        self.ranges: Dict[str, tuple] = SCORING_RANGES

    def calculate_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula el score financiero completo del cliente.

        Args:
            data: Diccionario con datos financieros del cliente.

        Returns:
            Diccionario con score total, desglose por dimensión,
            clasificación y explicaciones.
        """
        ingreso = data.get("ingreso_mensual", 0)
        gastos = data.get("total_gastos", 0)
        activos = data.get("total_activos", 0)
        deudas = data.get("total_deudas", 0)
        excedente = data.get("excedente", 0)
        mora = data.get("mora_diaria", 0)
        num_deudas = data.get("num_deudas", 0)
        capacidad = data.get("capacidad_pago", 0)
        endeudamiento = data.get("endeudamiento_patrimonial", 0)
        monto_solicitado = data.get("monto_solicitado", 0)
        capital_trabajo = data.get("capital_trabajo", 0)

        scores: Dict[str, int] = {}
        explanations: list = []

        # Income score (0-150)
        if ingreso <= 0:
            scores["ingreso"] = 0
        elif ingreso < 5000:
            scores["ingreso"] = 30
        elif ingreso < 15000:
            scores["ingreso"] = 60
        elif ingreso < 30000:
            scores["ingreso"] = 100
        elif ingreso < 50000:
            scores["ingreso"] = 130
        else:
            scores["ingreso"] = 150
        explanations.append(f"Ingreso mensual: S/{ingreso:.2f} -> {scores['ingreso']}/150 pts")

        # Expense ratio score (0-100)
        ratio_gastos = gastos / (ingreso + 1)
        if ratio_gastos <= 0.3:
            scores["gastos"] = 100
        elif ratio_gastos <= 0.5:
            scores["gastos"] = 75
        elif ratio_gastos <= 0.7:
            scores["gastos"] = 50
        elif ratio_gastos <= 0.9:
            scores["gastos"] = 25
        else:
            scores["gastos"] = 10
        explanations.append(f"Ratio gastos/ingresos: {ratio_gastos:.2%} -> {scores['gastos']}/100 pts")

        # Debt score (0-150)
        ratio_deuda = deudas / (activos + 1)
        if ratio_deuda <= 0.2:
            scores["deuda"] = 150
        elif ratio_deuda <= 0.4:
            scores["deuda"] = 120
        elif ratio_deuda <= 0.6:
            scores["deuda"] = 80
        elif ratio_deuda <= 0.8:
            scores["deuda"] = 40
        else:
            scores["deuda"] = 15
        explanations.append(f"Ratio deuda/activos: {ratio_deuda:.2%} -> {scores['deuda']}/150 pts")

        # Surplus score (0-120)
        if excedente <= 0:
            scores["excedente"] = 0
        elif excedente < 1000:
            scores["excedente"] = 30
        elif excedente < 3000:
            scores["excedente"] = 60
        elif excedente < 8000:
            scores["excedente"] = 90
        else:
            scores["excedente"] = 120
        explanations.append(f"Excedente mensual: S/{excedente:.2f} -> {scores['excedente']}/120 pts")

        # Payment capacity score (0-100)
        if capacidad <= 0:
            scores["capacidad"] = 0
        elif capacidad < 1:
            scores["capacidad"] = 25
        elif capacidad < 2:
            scores["capacidad"] = 50
        elif capacidad < 5:
            scores["capacidad"] = 75
        else:
            scores["capacidad"] = 100
        explanations.append(f"Capacidad de pago: {capacidad:.2f} -> {scores['capacidad']}/100 pts")

        # Number of debts score (0-80)
        if num_deudas == 0:
            scores["num_deudas"] = 80
        elif num_deudas <= 1:
            scores["num_deudas"] = 70
        elif num_deudas <= 2:
            scores["num_deudas"] = 50
        elif num_deudas <= 3:
            scores["num_deudas"] = 30
        else:
            scores["num_deudas"] = 10
        explanations.append(f"Nro de deudas: {num_deudas} -> {scores['num_deudas']}/80 pts")

        # Mora score (0-100)
        if mora <= 0:
            scores["mora"] = 100
        elif mora < 5:
            scores["mora"] = 80
        elif mora < 15:
            scores["mora"] = 60
        elif mora < 30:
            scores["mora"] = 40
        elif mora < 50:
            scores["mora"] = 20
        else:
            scores["mora"] = 5
        explanations.append(f"Mora diaria: {mora:.2f} d\u00edas -> {scores['mora']}/100 pts")

        # Indebtedness score (0-100)
        if endeudamiento <= 0:
            scores["endeudamiento"] = 100
        elif endeudamiento < 0.3:
            scores["endeudamiento"] = 90
        elif endeudamiento < 0.5:
            scores["endeudamiento"] = 70
        elif endeudamiento < 0.7:
            scores["endeudamiento"] = 50
        elif endeudamiento < 1.0:
            scores["endeudamiento"] = 30
        else:
            scores["endeudamiento"] = 10
        explanations.append(f"Endeudamiento patrimonial: {endeudamiento:.2%} -> {scores['endeudamiento']}/100 pts")

        # Working capital score (0-100)
        if capital_trabajo > 50000:
            scores["capital"] = 100
        elif capital_trabajo > 20000:
            scores["capital"] = 80
        elif capital_trabajo > 0:
            scores["capital"] = 60
        elif capital_trabajo > -20000:
            scores["capital"] = 30
        else:
            scores["capital"] = 10
        explanations.append(f"Capital de trabajo: S/{capital_trabajo:.2f} -> {scores['capital']}/100 pts")

        # Loan burden score (0-100)
        carga_prestamo = monto_solicitado / (ingreso * 12 + 1)
        if carga_prestamo <= 0.5:
            scores["carga"] = 100
        elif carga_prestamo <= 1.0:
            scores["carga"] = 80
        elif carga_prestamo <= 2.0:
            scores["carga"] = 50
        elif carga_prestamo <= 3.0:
            scores["carga"] = 25
        else:
            scores["carga"] = 5
        explanations.append(f"Carga financiera: {carga_prestamo:.2f}x ingreso anual -> {scores['carga']}/100 pts")

        total_score = sum(scores.values())
        classification = self.classify(total_score)

        return {
            "total_score": total_score,
            "max_score": 1000,
            "classification": classification,
            "category": classification.upper(),
            "details": scores,
            "explanations": explanations,
            "breakdown": {
                "ingreso": {"score": scores["ingreso"], "max": 150, "weight": "15%"},
                "gastos": {"score": scores["gastos"], "max": 100, "weight": "10%"},
                "deuda": {"score": scores["deuda"], "max": 150, "weight": "15%"},
                "excedente": {"score": scores["excedente"], "max": 120, "weight": "12%"},
                "capacidad": {"score": scores["capacidad"], "max": 100, "weight": "10%"},
                "num_deudas": {"score": scores["num_deudas"], "max": 80, "weight": "8%"},
                "mora": {"score": scores["mora"], "max": 100, "weight": "10%"},
                "endeudamiento": {"score": scores["endeudamiento"], "max": 100, "weight": "10%"},
                "capital": {"score": scores["capital"], "max": 100, "weight": "10%"},
            }
        }

    def classify(self, score: int) -> str:
        """Clasifica un score en una categoría según los rangos configurados."""
        for label, (low, high) in self.ranges.items():
            if low <= score <= high:
                return label
        return "critico"

    def adjust_with_ml(self, score_data: Dict[str, Any], ml_probability: float) -> Dict[str, Any]:
        """
        Ajusta el score financiero usando la probabilidad del modelo ML.

        El factor de ajuste se calcula como (ml_prob - 0.5) * 200,
        permitiendo que el ML refine el score tradicional.
        """
        ml_factor = (ml_probability - 0.5) * 200
        original_score = score_data["total_score"]
        adjusted_score = max(0, min(1000, original_score + ml_factor))
        adjusted_class = self.classify(int(adjusted_score))

        score_data["total_score_original"] = original_score
        score_data["total_score"] = int(adjusted_score)
        score_data["classification_original"] = score_data["classification"]
        score_data["classification"] = adjusted_class
        score_data["ml_adjustment"] = round(ml_factor, 2)
        score_data["ml_factor"] = round(ml_probability, 4)

        return score_data
