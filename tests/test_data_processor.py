import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import pytest
import pandas as pd
import numpy as np
from services.data_processor import DataProcessor


@pytest.fixture
def processor():
    return DataProcessor()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "ingreso_mensual": [15000, 8000, 30000, 12000, 9000, 25000, 18000, 5000],
        "total_gastos": [5000, 6000, 10000, 8000, 7000, 9000, 6000, 4000],
        "total_costos": [2000, 3000, 4000, 3000, 2500, 3500, 2000, 1500],
        "total_activos": [100000, 50000, 200000, 80000, 60000, 150000, 120000, 30000],
        "total_deudas": [20000, 40000, 50000, 30000, 35000, 45000, 25000, 15000],
        "num_deudas": [2, 4, 1, 3, 3, 2, 1, 5],
        "excedente": [3000, -1000, 8000, 2000, 1000, 5000, 4000, -500],
        "capacidad_pago": [3.5, 0.8, 5.0, 2.0, 1.5, 4.0, 3.0, 0.5],
        "endeudamiento_patrimonial": [0.2, 0.8, 0.25, 0.4, 0.6, 0.3, 0.2, 0.9],
        "capital_trabajo": [30000, -10000, 80000, 20000, 15000, 50000, 40000, -5000],
        "monto_solicitado": [15000, 25000, 50000, 20000, 18000, 35000, 30000, 10000],
        "tem": [0.03, 0.05, 0.02, 0.04, 0.035, 0.025, 0.03, 0.045],
        "num_cuotas": [12, 24, 6, 18, 12, 24, 12, 36],
        "cuota_estimada": [1500, 1500, 8800, 1400, 1700, 1800, 2900, 400],
        "monto_propuesto": [15000, 25000, 50000, 20000, 18000, 35000, 30000, 10000],
        "mora_diaria": [0, 45, 0, 5, 10, 0, 0, 60],
        "viable": [1, 0, 1, 1, 0, 1, 1, 0],
    })


class TestDataProcessor:
    def test_load_and_clean_removes_bom(self, processor):
        df = pd.DataFrame({"\ufeffcol": [1, 2]})
        result = processor.load_and_clean(df)
        assert "\ufeff" not in result.columns[0]

    def test_load_and_clean_handles_inf(self, processor):
        df = pd.DataFrame({"ingreso_mensual": [1, np.inf, 3]})
        result = processor.load_and_clean(df)
        assert not result.isin([np.inf, -np.inf]).any().any()

    def test_engineer_features_creates_expected_columns(self, processor, sample_df):
        result = processor.engineer_features(sample_df)
        expected = {"ratio_gastos_ingresos", "ratio_deuda_ingreso", "ratio_cuota_ingreso",
                    "liquidez", "capacidad_ahorro", "estres_financiero",
                    "mora_riesgo", "deuda_alta", "ahorro_positivo", "capacidad_baja"}
        assert expected.issubset(result.columns)

    def test_prepare_features_returns_expected_types(self, processor, sample_df):
        X, y, feature_names = processor.prepare_features(sample_df, target_col="viable")
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
        assert len(X) == len(sample_df)

    def test_split_and_balance_maintains_ratio(self, processor, sample_df):
        X, y, _ = processor.prepare_features(sample_df, target_col="viable")
        X_tr, X_te, y_tr, y_te = processor.split_and_balance(X, y)
        assert len(X_te) > 0
        assert len(y_te) > 0
        assert len(y_tr) > 0

    def test_preprocess_single(self, processor, sample_df):
        processor.prepare_features(sample_df, target_col="viable")
        processor.fit_scaler(np.array([[1, 2, 3]]), ["a", "b", "c"])
        single = processor.preprocess_single(sample_df.iloc[0].to_dict())
        assert isinstance(single, np.ndarray)
