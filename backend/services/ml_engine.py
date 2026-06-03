import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, FEATURES
from services.data_processor import DataProcessor


class MLEngine:
    """
    Motor principal de Machine Learning.

    Gestiona el entrenamiento, evaluación, ajuste de hiperparámetros
    y predicción de múltiples modelos de clasificación binaria para
    determinar la viabilidad financiera de un cliente.

    Modelos disponibles:
        - Random Forest
        - XGBoost
        - LightGBM
        - CatBoost
        - KNN (K-Nearest Neighbors)
        - Decision Tree (ID3 - entropy)
        - Gradient Boosting
        - Logistic Regression
        - Linear Regression (para score continuo)
        - Neural Network (MLP - Adam optimizer)
        - AdaBoost
        - Ensemble (Stacking)
    """

    def __init__(self) -> None:
        self.processor: DataProcessor = DataProcessor()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.metrics: List[Dict[str, Any]] = []
        self.feature_importance: Optional[Dict[str, Any]] = None
        self.shap_values: Optional[Dict[str, float]] = None

    def _get_models(self) -> Dict[str, Any]:
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt',
                random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=300, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, gamma=0.1, min_child_weight=3,
                random_state=42, eval_metric='logloss'
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=300, max_depth=10, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
                random_state=42, verbose=-1, class_weight='balanced'
            ),
            "CatBoost": cb.CatBoostClassifier(
                iterations=300, depth=8, learning_rate=0.05,
                l2_leaf_reg=3, border_count=128,
                random_seed=42, verbose=0, early_stopping_rounds=50
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=250, max_depth=6, learning_rate=0.05,
                min_samples_split=5, min_samples_leaf=2, subsample=0.8,
                random_state=42
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=5000, random_state=42, n_jobs=-1,
                C=0.5, penalty='l2', solver='lbfgs',
                class_weight='balanced'
            ),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32), max_iter=3000,
                random_state=42, early_stopping=True, validation_fraction=0.1,
                alpha=0.001, learning_rate='adaptive', batch_size=64
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=5, weights='distance', metric='minkowski', n_jobs=-1
            ),
            "Decision Tree (ID3)": DecisionTreeClassifier(
                criterion='entropy', max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, class_weight='balanced'
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=300, learning_rate=0.5, random_state=42, algorithm='SAMME'
            ),
        }

    def _cross_validate_model(
        self, model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = {
            "accuracy": cross_val_score(model, X, y, cv=skf, scoring='accuracy').tolist(),
            "precision": cross_val_score(model, X, y, cv=skf, scoring='precision').tolist(),
            "recall": cross_val_score(model, X, y, cv=skf, scoring='recall').tolist(),
            "f1": cross_val_score(model, X, y, cv=skf, scoring='f1').tolist(),
            "roc_auc": cross_val_score(model, X, y, cv=skf, scoring='roc_auc').tolist(),
        }
        return {k: {"values": v, "mean": round(np.mean(v), 4), "std": round(np.std(v), 4)} for k, v in scores.items()}

    def _tune_hyperparameters(
        self, model: Any, X_train: np.ndarray, y_train: np.ndarray, model_name: str
    ) -> Any:
        param_grids: Dict[str, Dict[str, List[Any]]] = {
            "Random Forest": {
                "n_estimators": [200, 300, 400],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
            },
            "XGBoost": {
                "n_estimators": [200, 300],
                "max_depth": [6, 8, 10],
                "learning_rate": [0.03, 0.05, 0.1],
            },
            "LightGBM": {
                "n_estimators": [200, 300],
                "max_depth": [6, 8, 10],
                "learning_rate": [0.03, 0.05, 0.1],
            },
        }
        if model_name not in param_grids:
            return model
        try:
            grid = GridSearchCV(
                model, param_grids[model_name],
                cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
            )
            grid.fit(X_train, y_train)
            print(f"  {model_name} best params: {grid.best_params_}")
            return grid.best_estimator_
        except Exception as e:
            print(f"  Tuning failed for {model_name}: {e}")
            return model

    def train_all(self, df: pd.DataFrame, target_col: str = "viable") -> List[Dict[str, Any]]:
        """
        Entrena todos los modelos, evalúa con validación cruzada,
        construye un ensemble stacking y selecciona el mejor modelo.

        Args:
            df: DataFrame con los datos de entrenamiento.
            target_col: Nombre de la columna objetivo.

        Returns:
            Lista de diccionarios con métricas de cada modelo.
        """
        X, y, feature_names = self.processor.prepare_features(df, target_col)
        X_train, X_test, y_train, y_test = self.processor.split_and_balance(X, y)
        self.processor.fit_scaler(X_train, feature_names)
        X_train_scaled = self.processor.transform(X_train)
        X_test_scaled = self.processor.transform(X_test)

        models = self._get_models()
        results: List[Dict[str, Any]] = []

        for name, model in models.items():
            try:
                print(f"Training {name}...")
                model_tuned = self._tune_hyperparameters(model, X_train_scaled, y_train, name)
                model_tuned.fit(X_train_scaled, y_train)
                y_pred = model_tuned.predict(X_test_scaled)
                y_prob = model_tuned.predict_proba(X_test_scaled)[:, 1]

                cv_scores = self._cross_validate_model(model_tuned, X_train_scaled, y_train)

                metrics = {
                    "model": name,
                    "accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                    "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                    "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                    "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
                    "cv_scores": cv_scores,
                }

                self.models[name] = {
                    "model": model_tuned,
                    "metrics": metrics,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "y_test": y_test,
                }
                results.append(metrics)
                print(f"  {name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
            except Exception as e:
                print(f"  {name} failed: {e}")
                results.append({"model": name, "error": str(e)})

        # Stacking Ensemble
        try:
            estimators = [(n, m["model"]) for n, m in self.models.items() if "error" not in m]
            if estimators:
                meta_learner = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
                ensemble = StackingClassifier(
                    estimators=estimators,
                    final_estimator=meta_learner,
                    cv=5, stack_method='predict_proba', n_jobs=-1
                )
                ensemble.fit(X_train_scaled, y_train)
                y_pred_ens = ensemble.predict(X_test_scaled)
                y_prob_ens = ensemble.predict_proba(X_test_scaled)[:, 1]
                ens_metrics = {
                    "model": "Ensemble (Stacking)",
                    "accuracy": round(accuracy_score(y_test, y_pred_ens), 4),
                    "precision": round(precision_score(y_test, y_pred_ens, zero_division=0), 4),
                    "recall": round(recall_score(y_test, y_pred_ens, zero_division=0), 4),
                    "f1_score": round(f1_score(y_test, y_pred_ens, zero_division=0), 4),
                    "roc_auc": round(roc_auc_score(y_test, y_prob_ens), 4),
                }
                self.models["Ensemble"] = {
                    "model": ensemble, "metrics": ens_metrics,
                    "y_pred": y_pred_ens, "y_prob": y_prob_ens, "y_test": y_test
                }
                results.append(ens_metrics)
                print(f"  Ensemble: Acc={ens_metrics['accuracy']:.4f}, F1={ens_metrics['f1_score']:.4f}, ROC-AUC={ens_metrics['roc_auc']:.4f}")
        except Exception as e:
            print(f"  Ensemble failed: {e}")
            results.append({"model": "Ensemble (Stacking)", "error": str(e)})

        # Select best model (weighted by F1 + ROC-AUC)
        valid = [r for r in results if "error" not in r]
        if valid:
            best = max(valid, key=lambda x: x["f1_score"] * 0.5 + x["roc_auc"] * 0.5)
            self.best_model_name = best["model"]
            self.best_model = self.models[best["model"]]["model"]

        self.metrics = results
        self.save_best_model()
        self._compute_feature_importance(feature_names)
        self.save_best_model()

        return results

    def _compute_feature_importance(self, feature_names: List[str]) -> None:
        results: Dict[str, Any] = {}
        for name, m in self.models.items():
            if "error" in m:
                continue
            model = m["model"]
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                if len(imp) == len(feature_names):
                    results[name] = {
                        fname: round(float(val), 4)
                        for fname, val in sorted(
                            zip(feature_names, imp),
                            key=lambda x: x[1], reverse=True
                        )[:20]
                    }
            elif hasattr(model, "coef_"):
                coef = np.abs(model.coef_[0])
                if len(coef) == len(feature_names):
                    results[name] = {
                        fname: round(float(val), 4)
                        for fname, val in sorted(
                            zip(feature_names, coef),
                            key=lambda x: x[1], reverse=True
                        )[:20]
                    }
            elif hasattr(model, "final_estimator_") and hasattr(model.final_estimator_, "coef_"):
                coef = np.abs(model.final_estimator_.coef_[0])
                if hasattr(model, "named_estimators_"):
                    base_names = list(model.named_estimators_.keys())
                elif hasattr(model, "estimators_"):
                    base_names = [str(type(e).__name__) for e in model.estimators_]
                else:
                    base_names = [f"est_{i}" for i in range(len(coef))]
                results[name] = {
                    base_names[i] if i < len(base_names) else f"estimator_{i}": round(float(val), 4)
                    for i, val in enumerate(coef)
                }
            elif hasattr(model, "estimators_"):
                try:
                    first_est = model.estimators_[0]
                    if isinstance(first_est, tuple):
                        first_est = first_est[1]
                    if hasattr(first_est, "feature_importances_"):
                        imp = first_est.feature_importances_
                        if len(imp) == len(feature_names):
                            results[name] = {
                                fname: round(float(val), 4)
                                for fname, val in sorted(
                                    zip(feature_names, imp),
                                    key=lambda x: x[1], reverse=True
                                )[:20]
                            }
                except Exception:
                    pass
        self.feature_importance = results

    def _extract_shap_summary(self, shap_values: Any, feature_names: List[str]) -> Dict[str, float]:
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        shap_summary = np.abs(shap_values).mean(axis=0)
        return {
            feature_names[i]: round(float(shap_summary[i]), 4)
            for i in range(min(len(feature_names), len(shap_summary)))
        }

    def compute_shap(self, df: pd.DataFrame, target_col: str = "viable") -> Optional[Dict[str, float]]:
        """
        Calcula valores SHAP para interpretar el modelo.

        Args:
            df: DataFrame con datos.
            target_col: Columna objetivo.

        Returns:
            Diccionario con importancia SHAP por feature.
        """
        if self.best_model is None:
            return None
        X, y, feature_names = self.processor.prepare_features(df, target_col)
        X_scaled = self.processor.transform(X)
        try:
            explainer = shap.TreeExplainer(self.best_model)
            shap_values = explainer.shap_values(X_scaled[:100])
            self.shap_values = self._extract_shap_summary(shap_values, feature_names)
        except Exception:
            try:
                explainer = shap.KernelExplainer(
                    self.best_model.predict_proba, X_scaled[:50]
                )
                shap_values = explainer.shap_values(X_scaled[:50])
                self.shap_values = self._extract_shap_summary(shap_values, feature_names)
            except Exception as e:
                print(f"SHAP computation failed: {e}")
                self.shap_values = None
        self.save_best_model()
        return self.shap_values

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predice la viabilidad financiera de un cliente.

        Args:
            data: Diccionario con datos del cliente.

        Returns:
            Diccionario con predicción, probabilidad, etiqueta y confianza.
        """
        if self.best_model is None:
            self.load_best_model()
        self.processor.load_scaler()
        X = self.processor.preprocess_single(data)
        X_scaled = self.processor.transform(X)

        y_prob = float(self.best_model.predict_proba(X_scaled)[:, 1][0])
        y_pred = int(self.best_model.predict(X_scaled)[0])

        return {
            "prediction": y_pred,
            "probability": round(y_prob, 4),
            "label": "VIABLE" if y_pred == 1 else "NO VIABLE",
            "confidence": round(max(y_prob, 1 - y_prob) * 100, 2),
            "model_used": self.best_model_name or "unknown",
        }

    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(d) for d in data_list]

    def save_best_model(self) -> None:
        if self.best_model is not None:
            MODELS_DIR.mkdir(exist_ok=True)
            models_summary: Dict[str, Any] = {}
            for name, m in self.models.items():
                if "error" not in m:
                    summary = {k: m[k] for k in m if k != "model"}
                    model = m.get("model")
                    self._add_architecture_info(summary, model)
                    models_summary[name] = summary
            joblib.dump({
                "model": self.best_model,
                "name": self.best_model_name,
                "metrics": self.metrics,
                "feature_importance": self.feature_importance,
                "shap_values": self.shap_values,
                "models_summary": models_summary,
            }, MODELS_DIR / "best_model.pkl")

    def _add_architecture_info(self, summary: Dict[str, Any], model: Any) -> None:
        if model is None:
            return
        summary["class"] = type(model).__name__
        for attr in ["n_estimators", "max_depth", "learning_rate", "activation", "solver", "max_iter",
                      "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_weight",
                      "num_leaves", "min_child_samples", "l2_leaf_reg", "border_count", "depth",
                      "early_stopping_rounds", "class_weight", "C", "penalty"]:
            if hasattr(model, attr):
                summary[attr] = getattr(model, attr)
        if hasattr(model, "hidden_layer_sizes"):
            summary["hidden_layer_sizes"] = list(model.hidden_layer_sizes)
            summary["n_layers"] = len(model.hidden_layer_sizes) + 2
        if hasattr(model, "early_stopping"):
            summary["early_stopping"] = model.early_stopping
        if hasattr(model, "named_estimators_"):
            summary["n_base_estimators"] = len(model.named_estimators_)
            summary["base_estimators"] = list(model.named_estimators_.keys())
        elif hasattr(model, "estimators_"):
            summary["n_base_estimators"] = len(model.estimators_)
            try:
                summary["base_estimators"] = [str(e).split('(')[0] for e in model.estimators_]
            except Exception:
                summary["base_estimators"] = [type(e).__name__ for e in model.estimators_]
        if hasattr(model, "final_estimator_"):
            summary["meta_learner"] = type(model.final_estimator_).__name__

    def load_best_model(self) -> bool:
        """
        Carga el mejor modelo guardado desde disco.

        Returns:
            True si se cargó correctamente, False en caso contrario.
        """
        model_path = MODELS_DIR / "best_model.pkl"
        if model_path.exists():
            data = joblib.load(model_path)
            self.best_model = data["model"]
            self.best_model_name = data.get("name", "unknown")
            self.metrics = data.get("metrics", [])
            self.feature_importance = data.get("feature_importance", None)
            self.shap_values = data.get("shap_values", None)
            models_summary = data.get("models_summary", {})
            for name, m_data in models_summary.items():
                self.models[name] = {"error": False, **m_data}
            if not self.feature_importance:
                self._compute_feature_importance_from_model()
        return self.best_model is not None

    def _compute_feature_importance_from_model(self) -> None:
        try:
            feature_names = self.processor.feature_names or [
                "ingreso_mensual", "total_gastos", "total_costos", "total_activos",
                "total_deudas", "num_deudas", "excedente", "capacidad_pago",
                "endeudamiento_patrimonial", "capital_trabajo", "monto_solicitado",
                "tem", "num_cuotas", "cuota_estimada", "monto_propuesto", "mora_diaria",
                "ratio_gastos_ingresos", "ratio_deuda_ingreso", "ratio_cuota_ingreso",
                "liquidez", "cobertura_gastos", "peso_deuda", "capacidad_ahorro",
                "estres_financiero"
            ]
            if hasattr(self.best_model, "feature_importances_"):
                imp = self.best_model.feature_importances_
                if len(imp) <= len(feature_names):
                    self.feature_importance = {
                        self.best_model_name or "Model": {
                            feature_names[i]: round(float(val), 4)
                            for i, val in enumerate(imp)
                        }
                    }
                    if self.feature_importance:
                        name = list(self.feature_importance.keys())[0]
                        sorted_imp = dict(sorted(
                            self.feature_importance[name].items(),
                            key=lambda x: x[1], reverse=True
                        )[:20])
                        self.feature_importance[name] = sorted_imp
        except Exception:
            pass

    def get_confusion_matrix(self, model_name: Optional[str] = None) -> Optional[Dict[str, int]]:
        if model_name and model_name in self.models and "error" not in self.models[model_name]:
            m = self.models[model_name]
            cm = confusion_matrix(m["y_test"], m["y_pred"])
            return {
                "true_negative": int(cm[0][0]),
                "false_positive": int(cm[0][1]),
                "false_negative": int(cm[1][0]),
                "true_positive": int(cm[1][1]),
            }
        if self.best_model_name and self.best_model_name in self.models:
            m = self.models[self.best_model_name]
            cm = confusion_matrix(m["y_test"], m["y_pred"])
            return {
                "true_negative": int(cm[0][0]),
                "false_positive": int(cm[0][1]),
                "false_negative": int(cm[1][0]),
                "true_positive": int(cm[1][1]),
            }
        return None

    def get_model_summary(self) -> Dict[str, Any]:
        nn_info = None
        if "Neural Network" in self.models:
            nn_meta = self.models["Neural Network"]
            if nn_meta.get("hidden_layer_sizes"):
                nn_info = {
                    "architecture": nn_meta.get("hidden_layer_sizes"),
                    "activation": nn_meta.get("activation"),
                    "solver": nn_meta.get("solver"),
                    "max_iter": nn_meta.get("max_iter"),
                    "n_layers": nn_meta.get("n_layers"),
                    "early_stopping": nn_meta.get("early_stopping"),
                    "class": nn_meta.get("class"),
                    "metrics": nn_meta.get("metrics"),
                }
        return {
            "best_model": self.best_model_name or "Not trained",
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "shap_values": self.shap_values,
            "num_models": len(self.models),
            "neural_network": nn_info,
            "model_architectures": self._get_all_architectures(),
        }

    def _get_all_architectures(self) -> Dict[str, Any]:
        archs: Dict[str, Any] = {}
        for name, m in self.models.items():
            if "error" in m and m["error"]:
                archs[name] = {"error": m["error"]}
                continue
            info = {}
            for attr in ["class", "n_estimators", "max_depth", "learning_rate",
                          "hidden_layer_sizes", "activation", "solver", "max_iter",
                          "n_layers", "early_stopping", "C", "subsample",
                          "colsample_bytree", "reg_alpha", "reg_lambda",
                          "num_leaves", "l2_leaf_reg", "depth",
                          "n_base_estimators", "base_estimators", "meta_learner",
                          "class_weight", "min_child_weight"]:
                if attr in m:
                    info[attr] = m[attr]
            archs[name] = info or {"note": "architecture info not stored"}
        return archs
