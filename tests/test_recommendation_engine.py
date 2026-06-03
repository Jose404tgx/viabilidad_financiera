import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import pytest
from services.recommendation_engine import RecommendationEngine


@pytest.fixture
def engine():
    return RecommendationEngine()


class TestRecommendationEngine:
    def test_viable_client(self, engine):
        prediction = {"probability": 0.92, "label": "VIABLE", "confidence": 92.0}
        score_data = {"total_score": 780, "classification": "bueno"}
        client_data = {"mora_diaria": 0, "num_deudas": 1, "excedente": 2000, "capacidad_pago": 3.0}
        result = engine.generate(prediction, score_data, client_data)
        assert result["decision_summary"]["viable"] is True
        assert len(result["recommendations"]) > 0
        assert result["risk_level"] in ("Bajo", "Moderado", "Alto", "Cr\u00edtico")

    def test_non_viable_client(self, engine):
        prediction = {"probability": 0.15, "label": "NO VIABLE", "confidence": 85.0}
        score_data = {"total_score": 250, "classification": "critico"}
        client_data = {"mora_diaria": 45, "num_deudas": 5, "excedente": -500, "capacidad_pago": 0.3}
        result = engine.generate(prediction, score_data, client_data)
        assert result["decision_summary"]["viable"] is False
        assert any("RECHAZAR" in r["action"] for r in result["recommendations"])

    def test_recommendations_sorted_by_priority(self, engine):
        prediction = {"probability": 0.5, "label": "NO VIABLE", "confidence": 50.0}
        score_data = {"total_score": 400, "classification": "regular"}
        client_data = {"mora_diaria": 0, "num_deudas": 2, "excedente": 500, "capacidad_pago": 1.5}
        result = engine.generate(prediction, score_data, client_data)
        priorities = [r["priority"] for r in result["recommendations"]]
        assert priorities == sorted(priorities)

    def test_high_mora_triggers_alert(self, engine):
        prediction = {"probability": 0.8, "label": "VIABLE", "confidence": 80.0}
        score_data = {"total_score": 600, "classification": "bueno"}
        client_data = {"mora_diaria": 90, "num_deudas": 1, "excedente": 1000, "capacidad_pago": 2.0}
        result = engine.generate(prediction, score_data, client_data)
        alerts = [r["action"] for r in result["recommendations"]]
        assert "ALERTA DE MORA" in alerts
