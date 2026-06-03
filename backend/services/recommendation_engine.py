from typing import Any, Dict, List


class RecommendationEngine:
    """
    Motor de recomendaciones financieras.

    Genera recomendaciones accionables basadas en la predicción del
    modelo ML, el score financiero y los datos del cliente.
    """

    def __init__(self) -> None:
        pass

    def generate(
        self, prediction: Dict[str, Any], score_data: Dict[str, Any], client_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Genera recomendaciones para un cliente.

        Args:
            prediction: Resultado de la predicción ML.
            score_data: Resultado del scoring financiero.
            client_data: Datos crudos del cliente.

        Returns:
            Diccionario con recomendaciones, resumen de decisión,
            razones, observaciones y nivel de riesgo.
        """
        probability = prediction.get("probability", 0.5)
        pred_label = prediction.get("label", "NO VIABLE")
        score = score_data.get("total_score", 0)
        classification = score_data.get("classification", "critico")

        recommendations: List[Dict[str, Any]] = []
        reasons: List[str] = []
        observations: List[str] = []

        # Core recommendation
        if pred_label == "VIABLE" and classification in ("excelente", "bueno"):
            recommendations.append({
                "action": "APROBAR CR\u00c9DITO",
                "type": "success",
                "description": "El cliente cumple con todos los requisitos financieros.",
                "priority": 1,
            })
            reasons.append("Alta capacidad de pago y buen historial crediticio.")
            observations.append("Cliente con perfil de bajo riesgo recomendado para aprobaci\u00f3n.")

        elif pred_label == "VIABLE" and classification == "regular":
            recommendations.append({
                "action": "APROBAR CON CONDICIONES",
                "type": "warning",
                "description": "Aprobar con monto reducido o garant\u00eda adicional.",
                "priority": 2,
            })
            recommendations.append({
                "action": "REDUCIR MONTO",
                "type": "info",
                "description": "Sugerir un monto menor al solicitado para mitigar riesgo.",
                "priority": 3,
            })
            reasons.append("Perfil regular que requiere condiciones especiales.")
            observations.append("Se recomienda ajustar cuotas para mejorar capacidad de pago.")

        elif pred_label == "NO VIABLE" and classification in ("excelente", "bueno"):
            recommendations.append({
                "action": "REVISAR MANUALMENTE",
                "type": "warning",
                "description": "Discrepancia entre score y predicci\u00f3n IA. Revisar manualmente.",
                "priority": 2,
            })
            reasons.append("Posible inconsistencia en datos financieros reportados.")
            observations.append("Verificar ingresos, deudas y documentaci\u00f3n presentada.")

        else:
            recommendations.append({
                "action": "RECHAZAR CR\u00c9DITO",
                "type": "danger",
                "description": "El cliente no cumple con los requisitos m\u00ednimos de riesgo.",
                "priority": 1,
            })
            recommendations.append({
                "action": "SOLICITAR GARANT\u00cdA",
                "type": "info",
                "description": "Si se considera aprobar, requerir garant\u00eda real o aval solidario.",
                "priority": 3,
            })
            reasons.append("Alto nivel de endeudamiento y baja capacidad de pago.")
            observations.append("Cliente con perfil de alto riesgo. Se recomienda rechazo.")

        # Additional risk-based recommendations
        mora = client_data.get("mora_diaria", 0)
        if mora > 30:
            recommendations.append({
                "action": "ALERTA DE MORA",
                "type": "danger",
                "description": f"Cliente presenta {mora:.0f} d\u00edas de mora. Alto riesgo de incumplimiento.",
                "priority": 1,
            })
            reasons.append("Historial de morosidad elevado.")

        deudas = client_data.get("num_deudas", 0)
        if deudas >= 4:
            recommendations.append({
                "action": "ALERTA SOBRE-ENDEUDAMIENTO",
                "type": "danger",
                "description": f"Cliente tiene {deudas} deudas activas. Riesgo de sobreendeudamiento.",
                "priority": 2,
            })
            reasons.append("M\u00faltiples obligaciones financieras activas.")

        excedente = client_data.get("excedente", 0)
        if excedente < 0:
            recommendations.append({
                "action": "D\u00c9FICIT FINANCIERO",
                "type": "danger",
                "description": "Cliente tiene d\u00e9ficit mensual. No recomendable para cr\u00e9dito.",
                "priority": 1,
            })
            reasons.append("Ingresos insuficientes para cubrir gastos mensuales.")

        capacidad = client_data.get("capacidad_pago", 0)
        if capacidad < 1:
            recommendations.append({
                "action": "BAJA CAPACIDAD DE PAGO",
                "type": "warning",
                "description": f"Capacidad de pago de {capacidad:.2f}. Umbral m\u00ednimo es 1.0.",
                "priority": 2,
            })

        # Score-based recommendations
        if score < 450:
            recommendations.append({
                "action": "REFINANCIAR DEUDAS",
                "type": "info",
                "description": "Sugerir refinanciamiento de deudas existentes antes de nuevo cr\u00e9dito.",
                "priority": 4,
            })

        return {
            "recommendations": sorted(recommendations, key=lambda x: x["priority"]),
            "decision_summary": {
                "viable": pred_label == "VIABLE",
                "score": score,
                "classification": classification,
                "confidence": prediction.get("confidence", 0),
            },
            "reasons": reasons,
            "observations": observations,
            "risk_level": self._get_risk_level(score, probability),
            "final_decision": recommendations[0]["action"] if recommendations else "INDEFINIDO",
        }

    def _get_risk_level(self, score: int, probability: float) -> str:
        """Calcula el nivel de riesgo combinando score financiero y probabilidad ML."""
        risk_score = (1000 - score) / 1000 * 0.6 + (1 - probability) * 0.4
        if risk_score < 0.25:
            return "Bajo"
        elif risk_score < 0.5:
            return "Moderado"
        elif risk_score < 0.75:
            return "Alto"
        return "Cr\u00edtico"
