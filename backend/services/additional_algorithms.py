import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json
from typing import Any, Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


class AdditionalAlgorithms:
    """
    Algoritmos adicionales requeridos por el syllabus académico:

        - K-Means (análisis de clusters de clientes)
        - Linear Regression (predicción de score continuo con MSE)
        - Apriori (reglas de asociación entre variables financieras)
        - ID3 (Decision Tree con criterio entropy, en MLEngine)
        - KNN (K-Nearest Neighbors, en MLEngine)
        - Adam (optimizador por defecto en MLPClassifier)
    """

    def __init__(self) -> None:
        self.linear_model: Optional[LinearRegression] = None
        self.mse: Optional[float] = None
        self.r2: Optional[float] = None

    def train_linear_regression(self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "viable") -> Dict[str, Any]:
        """
        Entrena un modelo de Regresión Lineal para predecir
        la probabilidad de viabilidad como variable continua.

        Métrica utilizada: MSE (Mean Squared Error).
        """
        X = df[feature_cols].values
        y = df[target_col].values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)

        y_pred = self.linear_model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)

        coefficients = dict(zip(feature_cols, map(float, self.linear_model.coef_)))
        top_features = dict(sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

        return {
            "model": "Linear Regression",
            "algorithm": "Regresión Lineal (Mínimos Cuadrados Ordinarios)",
            "metric": "MSE (Mean Squared Error)",
            "mse": round(self.mse, 6),
            "rmse": round(np.sqrt(self.mse), 6),
            "r2_score": round(self.r2, 4),
            "intercept": float(self.linear_model.intercept_),
            "top_coefficients": top_features,
            "formula": f"y = {float(self.linear_model.intercept_):.4f} + {' + '.join([f'{v:.4f}*{k}' for k, v in top_features.items()][:5])}",
        }

    def linear_regression_predict(self, data: Dict[str, Any], feature_cols: List[str]) -> Dict[str, Any]:
        if self.linear_model is None:
            return {"error": "Linear Regression model not trained"}
        X = pd.DataFrame([data])[feature_cols].values
        pred = float(self.linear_model.predict(X)[0])
        return {
            "predicted_score": round(pred, 4),
            "predicted_label": "VIABLE" if pred >= 0.5 else "NO VIABLE",
            "model": "Linear Regression",
        }

    def apriori_analysis(self, df: pd.DataFrame, min_support: float = 0.1, min_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Algoritmo Apriori para encontrar reglas de asociación
        entre variables financieras discretizadas.
        """
        if not HAS_MLXTEND:
            return {
                "error": "mlxtend not installed. Run: pip install mlxtend",
                "note": "Apriori algorithm requires mlxtend library"
            }

        df_bin = df.copy()

        # Discretizar variables numéricas a binarias
        if "ingreso_mensual" in df_bin.columns:
            df_bin["ingreso_alto"] = (df_bin["ingreso_mensual"] > df_bin["ingreso_mensual"].median()).astype(bool)
        if "total_deudas" in df_bin.columns:
            df_bin["deuda_alta_ap"] = (df_bin["total_deudas"] > df_bin["total_deudas"].median()).astype(bool)
        if "excedente" in df_bin.columns:
            df_bin["excedente_positivo"] = (df_bin["excedente"] > 0).astype(bool)
        if "mora_diaria" in df_bin.columns:
            df_bin["mora_alta"] = (df_bin["mora_diaria"] > 30).astype(bool)
        if "viable" in df_bin.columns:
            df_bin["resultado_viable"] = df_bin["viable"].astype(bool)
        if "ratio_deuda_ingreso" in df_bin.columns:
            pass
        elif "ingreso_mensual" in df_bin.columns and "total_deudas" in df_bin.columns:
            df_bin["ratio_alto"] = ((df_bin["total_deudas"] / (df_bin["ingreso_mensual"] + 1e-8)) > 0.5).astype(bool)

        bool_cols = [c for c in df_bin.columns if df_bin[c].dtype == bool]
        if not bool_cols:
            return {"error": "No boolean columns available for Apriori"}

        df_ap = df_bin[bool_cols]

        try:
            frequent_itemsets = apriori(df_ap, min_support=min_support, use_colnames=True)
            if len(frequent_itemsets) == 0:
                return {"error": f"No frequent itemsets found with min_support={min_support}"}

            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
            rules = rules.sort_values("lift", ascending=False).head(15)

            rules_list = []
            for _, r in rules.iterrows():
                rules_list.append({
                    "antecedents": list(r["antecedents"]),
                    "consequents": list(r["consequents"]),
                    "support": round(r["support"], 4),
                    "confidence": round(r["confidence"], 4),
                    "lift": round(r["lift"], 4),
                })

            return {
                "algorithm": "Apriori",
                "total_rules": len(rules),
                "min_support": min_support,
                "min_confidence": min_threshold,
                "top_rules": rules_list[:10],
                "interpretation": "Las reglas con lift > 1 indican asociación positiva entre variables financieras.",
            }
        except Exception as e:
            return {"error": f"Apriori failed: {str(e)}"}

    def get_nn_architecture(self, mlp_model) -> Dict[str, Any]:
        """
        Retorna la arquitectura completa de la Red Neuronal (MLP):
        - Capa de entrada (input layer)
        - Capas ocultas (hidden layers)
        - Capa de salida (output layer)
        - Función de activación
        - Pesos (weights) y sesgos (biases)
        - Optimizador (Adam)
        """
        if mlp_model is None:
            return {"error": "MLP model not available"}

        coefs = mlp_model.coefs_
        intercepts = mlp_model.intercepts_

        layers = []
        n_features = coefs[0].shape[0]

        # Input layer
        layers.append({
            "name": "Input Layer",
            "type": "input",
            "neurons": n_features,
            "activation": "identity (no activation)",
            "weights_shape": list(coefs[0].shape),
        })

        # Hidden layers
        for i in range(len(coefs) - 1):
            layers.append({
                "name": f"Hidden Layer {i + 1}",
                "type": "hidden",
                "neurons": coefs[i].shape[1],
                "activation": mlp_model.activation,
                "weights_shape": list(coefs[i].shape),
                "weights_sample": [round(float(w), 4) for w in coefs[i][0][:5]],
                "bias_sample": [round(float(b), 4) for b in intercepts[i][:5]],
            })

        # Output layer
        layers.append({
            "name": "Output Layer",
            "type": "output",
            "neurons": coefs[-1].shape[1],
            "activation": mlp_model.out_activation_ if hasattr(mlp_model, 'out_activation_') else 'sigmoid' if coefs[-1].shape[1] == 1 else 'softmax',
            "weights_shape": list(coefs[-1].shape),
            "weights_sample": [round(float(w), 4) for w in coefs[-1][0][:5]],
            "bias_sample": [round(float(b), 4) for b in intercepts[-1]],
        })

        total_params = sum(w.size for w in coefs) + sum(b.size for b in intercepts)

        return {
            "model_type": "MLPClassifier (Multi-Layer Perceptron)",
            "optimizer": "Adam (Adaptive Moment Estimation)",
            "loss_function": mlp_model.loss if hasattr(mlp_model, 'loss') else 'log_loss',
            "learning_rate_init": mlp_model.learning_rate_init if hasattr(mlp_model, 'learning_rate_init') else 'adaptive',
            "max_iterations": mlp_model.max_iter,
            "early_stopping": mlp_model.early_stopping if hasattr(mlp_model, 'early_stopping') else False,
            "total_parameters": total_params,
            "architecture_diagram": {
                "total_layers": len(layers),
                "input_neurons": layers[0]["neurons"],
                "hidden_layers": len(layers) - 2,
                "output_neurons": layers[-1]["neurons"],
                "layers_detail": layers,
            },
            "note": "Las neuronas se conectan completamente (fully connected) entre capas adyacentes. "
                    "Cada conexión tiene un peso (weight) y cada neurona un sesgo (bias). "
                    "La función de activación ReLU se aplica en capas ocultas; la salida usa sigmoide para clasificación binaria.",
        }
