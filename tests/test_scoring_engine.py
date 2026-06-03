import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import pytest
from services.scoring_engine import ScoringEngine


@pytest.fixture
def engine():
    return ScoringEngine()


@pytest.fixture
def sample_client():
    return {
        "ingreso_mensual": 15000.0,
        "total_gastos": 5000.0,
        "total_costos": 2000.0,
        "total_activos": 100000.0,
        "total_deudas": 20000.0,
        "num_deudas": 2,
        "excedente": 3000.0,
        "capacidad_pago": 3.5,
        "endeudamiento_patrimonial": 0.2,
        "capital_trabajo": 30000.0,
        "monto_solicitado": 15000.0,
        "tem": 0.03,
        "num_cuotas": 12,
        "cuota_estimada": 1500.0,
        "monto_propuesto": 15000.0,
        "mora_diaria": 0,
    }


class TestScoringEngine:
    def test_calculate_score_returns_all_keys(self, engine, sample_client):
        result = engine.calculate_score(sample_client)
        expected_keys = {"total_score", "max_score", "classification", "category", "details", "explanations", "breakdown"}
        assert expected_keys.issubset(result.keys())

    def test_calculate_score_max_1000(self, engine, sample_client):
        result = engine.calculate_score(sample_client)
        assert 0 <= result["total_score"] <= 1000

    def test_calculate_score_details_9_dimensions(self, engine, sample_client):
        result = engine.calculate_score(sample_client)
        expected = {"ingreso", "gastos", "deuda", "excedente", "capacidad", "num_deudas", "mora", "endeudamiento", "capital"}
        assert expected.issubset(result["details"].keys())

    def test_classify_ranges(self, engine):
        assert engine.classify(850) == "excelente"
        assert engine.classify(650) == "bueno"
        assert engine.classify(450) == "regular"
        assert engine.classify(200) == "critico"

    def test_adjust_with_ml_high_probability(self, engine, sample_client):
        score = engine.calculate_score(sample_client)
        adjusted = engine.adjust_with_ml(score, 0.95)
        assert "ml_adjustment" in adjusted
        assert "ml_factor" in adjusted
        assert adjusted["ml_factor"] == 0.95

    def test_adjust_with_ml_low_probability(self, engine, sample_client):
        score = engine.calculate_score(sample_client)
        adjusted = engine.adjust_with_ml(score, 0.1)
        assert adjusted["total_score"] <= score["total_score"]

    def test_zero_income(self, engine):
        data = {k: 0 for k in ["ingreso_mensual", "total_gastos", "total_costos",
                                "total_activos", "total_deudas", "num_deudas",
                                "excedente", "capacidad_pago", "endeudamiento_patrimonial",
                                "capital_trabajo", "monto_solicitado", "tem",
                                "num_cuotas", "cuota_estimada", "monto_propuesto",
                                "mora_diaria"]}
        data["num_cuotas"] = 1
        data["ingreso_mensual"] = 0.01
        result = engine.calculate_score(data)
        assert result["total_score"] >= 0
        assert result["classification"] in ("excelente", "bueno", "regular", "critico")
