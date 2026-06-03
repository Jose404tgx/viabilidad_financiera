import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
from typing import Any, Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES, TARGET_BINARIO, MODELS_DIR


class DataProcessor:
    """
    Procesamiento y transformación de datos financieros.

    Responsabilidades:
        - Limpieza y validación de datos crudos.
        - Ingeniería de características (24 features derivadas).
        - Selección automática de features relevantes.
        - Escalado robusto (RobustScaler + PowerTransformer).
        - Balanceo de clases con SMOTE.
        - División train/test estratificada.
    """

    def __init__(self) -> None:
        self.scaler: RobustScaler = RobustScaler()
        self.power_transformer: PowerTransformer = PowerTransformer(method='yeo-johnson')
        self.features: List[str] = FEATURES
        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None

    def load_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia el DataFrame: elimina columnas irrelevantes, maneja infinitos y nulos."""
        df = df.copy()
        df.columns = df.columns.str.replace('\ufeff', '', regex=False)
        for drop_col in ["id_cliente", "resultado_aprobado", "resultado_rechazado", "resultado_riesgo_medio"]:
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        df = df.fillna(0)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera 24 características derivadas a partir de los datos base.

        Categorías:
            - Ratios financieros (gastos/ingreso, deuda/ingreso, cuota/ingreso)
            - Indicadores de liquidez y solvencia
            - Métricas de capacidad de pago y estrés financiero
            - Features binarias de riesgo
            - Transformaciones logarítmicas
        """
        df = df.copy()
        eps = 1e-8

        df["ratio_gastos_ingresos"] = df["total_gastos"] / (df["ingreso_mensual"] + eps)
        df["ratio_deuda_ingreso"] = df["total_deudas"] / (df["ingreso_mensual"] + eps)
        df["ratio_cuota_ingreso"] = df["cuota_estimada"] / (df["ingreso_mensual"] + eps)
        df["liquidez"] = df["total_activos"] / (df["total_deudas"] + eps)
        df["cobertura_gastos"] = df["excedente"] / (df["total_gastos"] + eps)
        df["peso_deuda"] = df["total_deudas"] / (df["total_activos"] + eps)
        df["capacidad_ahorro"] = df["excedente"] / (df["ingreso_mensual"] + eps)
        df["estres_financiero"] = df["monto_solicitado"] / (df["ingreso_mensual"] * df["num_cuotas"] + eps)

        df["carga_financiera_total"] = (df["total_deudas"] + df["monto_solicitado"]) / (df["ingreso_mensual"] * 12 + eps)
        df["cobertura_cuota"] = df["excedente"] / (df["cuota_estimada"] + eps)
        df["eficiencia_operativa"] = df["total_costos"] / (df["total_gastos"] + eps)
        df["apalancamiento"] = df["total_deudas"] / (df["capital_trabajo"].abs() + eps)
        df["solvencia"] = df["total_activos"] / (df["total_pasivos"] + eps) if "total_pasivos" in df.columns else df["total_activos"] / (df["total_deudas"] + eps)
        df["ingreso_por_deuda"] = df["ingreso_mensual"] / (df["num_deudas"] + eps)

        df["mora_riesgo"] = (df["mora_diaria"] > 30).astype(int)
        df["deuda_alta"] = (df["ratio_deuda_ingreso"] > 0.5).astype(int)
        df["ahorro_positivo"] = (df["excedente"] > 0).astype(int)
        df["capacidad_baja"] = (df["capacidad_pago"] < 1).astype(int)

        if "ingreso_mensual" in df.columns and "total_gastos" in df.columns:
            df["ingreso_log"] = np.log1p(df["ingreso_mensual"])
            df["activos_log"] = np.log1p(df["total_activos"])
            df["deudas_log"] = np.log1p(df["total_deudas"])

        return df

    def select_features(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Selecciona las features más relevantes usando SelectFromModel con Random Forest."""
        try:
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
                threshold='median', max_features=20
            )
            selector.fit(X, y)
            selected = [feature_names[i] for i in range(len(feature_names)) if selector.get_support()[i]]
            if selected:
                self.selected_features = selected
                X_selected = selector.transform(X)
                print(f"Feature selection: {len(feature_names)} -> {len(selected)} features")
                return X_selected, selected
        except Exception as e:
            print(f"Feature selection failed: {e}")
        return X, feature_names

    def prepare_features(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Pipeline completo: limpieza -> ingeniería -> selección.

        Returns:
            Tupla (X, y, feature_names).
        """
        df_orig = df.copy()
        df = self.load_and_clean(df_orig)
        if target_col and target_col in df_orig.columns and target_col not in df.columns:
            target_series = df_orig[target_col]
        elif target_col and target_col in df.columns:
            target_series = df[target_col]
            df = df.drop(columns=[target_col])
        else:
            target_series = None
        df = self.engineer_features(df)
        feature_cols = list(df.columns)
        X = df.values
        y = target_series.values if target_series is not None else None

        if y is not None:
            X, feature_cols = self.select_features(X, y, feature_cols)

        return X, y, feature_cols

    def split_and_balance(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divide en train/test estratificado y aplica SMOTE para balancear clases."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=3)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"SMOTE applied: {X_train.shape[0]} -> {X_train_res.shape[0]} samples")
            return X_train_res, X_test, y_train_res, y_test
        except Exception as e:
            print(f"SMOTE failed, using original data: {e}")
            return X_train, X_test, y_train, y_test

    def fit_scaler(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> RobustScaler:
        """Ajusta el scaler y guarda a disco."""
        self.scaler.fit(X_train)
        try:
            self.power_transformer.fit(X_train)
        except Exception:
            pass
        save_data: Dict[str, Any] = {
            "scaler": self.scaler,
            "power_transformer": self.power_transformer,
        }
        if feature_names:
            save_data["feature_names"] = feature_names
        if self.selected_features:
            save_data["selected_features"] = self.selected_features
        joblib.dump(save_data, MODELS_DIR / "scaler.pkl")
        self.feature_names = feature_names
        return self.scaler

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica escalado robusto."""
        return self.scaler.transform(X)

    def load_scaler(self) -> RobustScaler:
        """Carga scaler previamente guardado."""
        scaler_path = MODELS_DIR / "scaler.pkl"
        if scaler_path.exists():
            data = joblib.load(scaler_path)
            self.scaler = data.get("scaler") if isinstance(data, dict) else data
            self.feature_names = data.get("feature_names") if isinstance(data, dict) else None
            self.selected_features = data.get("selected_features") if isinstance(data, dict) else None
            if isinstance(data, dict) and "power_transformer" in data:
                self.power_transformer = data["power_transformer"]
        return self.scaler

    def preprocess_single(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocesa un solo registro para predicción."""
        df = pd.DataFrame([data])
        df = self.load_and_clean(df)
        df = self.engineer_features(df)
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        return df.values
