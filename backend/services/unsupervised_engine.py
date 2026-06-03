import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES


class UnsupervisedEngine:
    def __init__(self):
        self.scaler = RobustScaler()
        self.clustering_models = {}
        self.anomaly_models = {}
        self.dr_models = {}
        self.results = {}
        self.data = None
        self.features = FEATURES

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        raw = df.copy()
        raw.columns = raw.columns.str.replace('\ufeff', '', regex=False)
        drop_cols = ["id_cliente", "viable", "resultado_aprobado",
                     "resultado_rechazado", "resultado_riesgo_medio"]
        for c in drop_cols:
            if c in raw.columns:
                raw = raw.drop(columns=[c])
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        raw = raw.select_dtypes(include=[np.number])

        # Engineer features for better clustering
        if "ingreso_mensual" in raw.columns and "total_gastos" in raw.columns:
            raw["ratio_gastos_ingresos"] = raw["total_gastos"] / (raw["ingreso_mensual"] + 1)
        if "total_deudas" in raw.columns and "total_activos" in raw.columns:
            raw["ratio_deuda_activos"] = raw["total_deudas"] / (raw["total_activos"] + 1)
        if "excedente" in raw.columns and "ingreso_mensual" in raw.columns:
            raw["tasa_ahorro"] = raw["excedente"] / (raw["ingreso_mensual"] + 1)
        if "total_deudas" in raw.columns and "ingreso_mensual" in raw.columns:
            raw["deuda_ingreso"] = raw["total_deudas"] / (raw["ingreso_mensual"] + 1)
        if "monto_solicitado" in raw.columns and "ingreso_mensual" in raw.columns and "num_cuotas" in raw.columns:
            raw["estres_financiero"] = raw["monto_solicitado"] / (raw["ingreso_mensual"] * raw["num_cuotas"] + 1)

        self.data = raw
        X = raw.values
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, raw.columns.tolist()

    # ===== CLUSTERING =====

    def run_kmeans(self, df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> dict:
        X_scaled, feature_names = self._prepare_data(df)
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
        db = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else 0
        ch = calinski_harabasz_score(X_scaled, labels) if len(set(labels)) > 1 else 0

        centers_scaled = model.cluster_centers_
        centers_original = self.scaler.inverse_transform(centers_scaled)

        # Profile each cluster
        df_copy = df.copy().reset_index(drop=True)
        df_copy["cluster"] = labels
        profiles = []
        for c in range(n_clusters):
            mask = labels == c
            cluster_data = df_copy[mask]
            profile = {
                "cluster": int(c),
                "size": int(mask.sum()),
                "percentage": round(float(mask.sum() / len(labels) * 100), 2),
                "center": {
                    feature_names[i]: round(float(centers_original[c][i]), 2)
                    for i in range(min(10, len(feature_names)))
                },
            }
            if "viable" in df_copy.columns:
                viable_ratio = float(cluster_data["viable"].mean())
                profile["viable_ratio"] = round(viable_ratio, 4)
            profiles.append(profile)

        cluster_map = {int(i): int(l) for i, l in enumerate(labels)}

        self.clustering_models["kmeans"] = model
        self.results["kmeans"] = {
            "labels": labels.tolist(),
            "profiles": profiles,
            "metrics": {
                "silhouette_score": round(sil, 4),
                "davies_bouldin_score": round(db, 4),
                "calinski_harabasz_score": round(ch, 2),
                "inertia": float(model.inertia_),
            },
            "cluster_map": cluster_map,
        }
        return self.results["kmeans"]

    def run_dbscan(self, df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> dict:
        X_scaled, feature_names = self._prepare_data(df)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())

        sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0
        db_score = davies_bouldin_score(X_scaled, labels) if n_clusters > 1 else 0
        ch = calinski_harabasz_score(X_scaled, labels) if n_clusters > 1 else 0

        df_copy = df.copy().reset_index(drop=True)
        df_copy["cluster"] = labels
        profiles = []
        for c in sorted(set(labels)):
            mask = labels == c
            cluster_data = df_copy[mask]
            profile = {
                "cluster": int(c),
                "size": int(mask.sum()),
                "percentage": round(float(mask.sum() / len(labels) * 100), 2),
                "is_noise": bool(c == -1),
            }
            if "viable" in df_copy.columns:
                profile["viable_ratio"] = round(float(cluster_data["viable"].mean()), 4)
            profiles.append(profile)

        self.clustering_models["dbscan"] = model
        self.results["dbscan"] = {
            "labels": labels.tolist(),
            "profiles": profiles,
            "metrics": {
                "num_clusters": n_clusters,
                "noise_points": n_noise,
                "noise_percentage": round(n_noise / len(labels) * 100, 2),
                "silhouette_score": round(sil, 4),
                "davies_bouldin_score": round(db_score, 4),
                "calinski_harabasz_score": round(ch, 2),
            },
        }
        return self.results["dbscan"]

    def run_hierarchical(self, df: pd.DataFrame, n_clusters: int = 4) -> dict:
        X_scaled, feature_names = self._prepare_data(df)
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
        db_score = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else 0
        ch = calinski_harabasz_score(X_scaled, labels) if len(set(labels)) > 1 else 0

        df_copy = df.copy().reset_index(drop=True)
        df_copy["cluster"] = labels
        profiles = []
        for c in range(n_clusters):
            mask = labels == c
            cluster_data = df_copy[mask]
            profile = {
                "cluster": int(c),
                "size": int(mask.sum()),
                "percentage": round(float(mask.sum() / len(labels) * 100), 2),
            }
            if "viable" in df_copy.columns:
                profile["viable_ratio"] = round(float(cluster_data["viable"].mean()), 4)
            profiles.append(profile)

        self.clustering_models["hierarchical"] = model
        self.results["hierarchical"] = {
            "labels": labels.tolist(),
            "profiles": profiles,
            "metrics": {
                "silhouette_score": round(sil, 4),
                "davies_bouldin_score": round(db_score, 4),
                "calinski_harabasz_score": round(ch, 2),
            },
        }
        return self.results["hierarchical"]

    def auto_select_k(self, df: pd.DataFrame, k_range: range = range(2, 9)) -> dict:
        X_scaled, _ = self._prepare_data(df)
        results = []
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
            db_score = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else 0
            ch = calinski_harabasz_score(X_scaled, labels) if len(set(labels)) > 1 else 0
            results.append({
                "k": k,
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db_score, 4),
                "calinski_harabasz": round(ch, 2),
                "inertia": float(model.inertia_),
            })
        best_k = max(results, key=lambda x: x["silhouette"])["k"]
        return {"results": results, "optimal_k": best_k}

    # ===== ANOMALY DETECTION =====

    def detect_anomalies_iforest(self, df: pd.DataFrame, contamination: float = 0.1, random_state: int = 42) -> dict:
        X_scaled, feature_names = self._prepare_data(df)
        model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
        preds = model.fit_predict(X_scaled)
        scores = model.score_samples(X_scaled)

        anomaly_mask = preds == -1
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_scores = float(scores[anomaly_mask].mean()) if anomaly_mask.any() else 0

        df_copy = df.copy().reset_index(drop=True)
        anomaly_df = df_copy.iloc[anomaly_indices] if anomaly_indices else pd.DataFrame()

        anomaly_details = []
        for idx in anomaly_indices[:50]:
            row = df_copy.iloc[idx]
            detail = {
                "index": int(idx),
                "score": round(float(scores[idx]), 4),
            }
            if "id_cliente" in row:
                detail["id_cliente"] = int(row["id_cliente"]) if pd.notna(row["id_cliente"]) else None
            if "ingreso_mensual" in row:
                detail["ingreso_mensual"] = float(row["ingreso_mensual"])
            if "total_deudas" in row:
                detail["total_deudas"] = float(row["total_deudas"])
            if "viable" in row:
                detail["viable"] = int(row["viable"])
            anomaly_details.append(detail)

        self.anomaly_models["iforest"] = model
        result = {
            "algorithm": "Isolation Forest",
            "total_samples": len(preds),
            "anomalies_found": int(anomaly_mask.sum()),
            "anomaly_percentage": round(float(anomaly_mask.mean() * 100), 2),
            "contamination_used": contamination,
            "avg_anomaly_score": round(anomaly_scores, 4),
            "anomaly_details": anomaly_details,
            "all_scores": [round(float(s), 4) for s in scores],
        }
        self.results["iforest"] = result
        return result

    def detect_anomalies_lof(self, df: pd.DataFrame, contamination: float = 0.1, n_neighbors: int = 20) -> dict:
        X_scaled, feature_names = self._prepare_data(df)
        model = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors, novelty=False)
        preds = model.fit_predict(X_scaled)
        scores = -model.negative_outlier_factor_

        anomaly_mask = preds == -1
        anomaly_indices = np.where(anomaly_mask)[0].tolist()

        anomaly_details = []
        for idx in anomaly_indices[:50]:
            row = df.copy().reset_index(drop=True).iloc[idx]
            detail = {
                "index": int(idx),
                "lof_score": round(float(scores[idx]), 4),
            }
            if "id_cliente" in row:
                detail["id_cliente"] = int(row["id_cliente"]) if pd.notna(row["id_cliente"]) else None
            if "ingreso_mensual" in row:
                detail["ingreso_mensual"] = float(row["ingreso_mensual"])
            if "total_deudas" in row:
                detail["total_deudas"] = float(row["total_deudas"])
            if "viable" in row:
                detail["viable"] = int(row["viable"])
            anomaly_details.append(detail)

        self.anomaly_models["lof"] = model
        result = {
            "algorithm": "Local Outlier Factor",
            "total_samples": len(preds),
            "anomalies_found": int(anomaly_mask.sum()),
            "anomaly_percentage": round(float(anomaly_mask.mean() * 100), 2),
            "contamination_used": contamination,
            "n_neighbors": n_neighbors,
            "anomaly_details": anomaly_details,
            "all_scores": [round(float(s), 4) for s in scores.tolist()],
        }
        self.results["lof"] = result
        return result

    # ===== DIMENSIONALITY REDUCTION =====

    def run_pca(self, df: pd.DataFrame, n_components: int = 2) -> dict:
        X_scaled, feature_names = self._prepare_data(df)
        model = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]))
        transformed = model.fit_transform(X_scaled)

        explained_variance = model.explained_variance_ratio_.tolist()
        cumulative_variance = np.cumsum(model.explained_variance_ratio_).tolist()

        # Component loadings
        loadings = {}
        for i in range(model.n_components_):
            loadings[f"PC{i+1}"] = {
                feature_names[j]: round(float(model.components_[i][j]), 4)
                for j in range(min(15, len(feature_names)))
            }

        # Add cluster labels if any clustering was done
        result_data = {
            "coordinates": {
                "x": [round(float(p[0]), 4) for p in transformed],
                "y": [round(float(p[1]), 4) for p in transformed] if transformed.shape[1] > 1 else [],
            }
        }
        if transformed.shape[1] >= 3:
            result_data["coordinates"]["z"] = [round(float(p[2]), 4) for p in transformed]

        # Attach cluster labels from previous runs
        for cname, cres in self.results.items():
            if cname in ("kmeans", "dbscan", "hierarchical") and "labels" in cres:
                result_data[f"cluster_{cname}"] = cres["labels"]

        self.dr_models["pca"] = model
        result = {
            "algorithm": "PCA",
            "n_components": model.n_components_,
            "explained_variance": [round(float(v), 4) for v in explained_variance],
            "cumulative_variance": [round(float(v), 4) for v in cumulative_variance],
            "total_explained": round(float(sum(explained_variance)), 4),
            "loadings": loadings,
            "data": result_data,
        }
        self.results["pca"] = result
        return result

    def run_tsne(self, df: pd.DataFrame, n_components: int = 2, perplexity: int = 30, random_state: int = 42) -> dict:
        X_scaled, feature_names = self._prepare_data(df)
        n_samples = X_scaled.shape[0]
        actual_perp = min(perplexity, max(1, n_samples // 3))
        model = TSNE(n_components=n_components, perplexity=actual_perp, random_state=random_state)
        transformed = model.fit_transform(X_scaled)

        result_data = {
            "coordinates": {
                "x": [round(float(p[0]), 4) for p in transformed],
                "y": [round(float(p[1]), 4) for p in transformed] if transformed.shape[1] > 1 else [],
            }
        }
        if transformed.shape[1] >= 3:
            result_data["coordinates"]["z"] = [round(float(p[2]), 4) for p in transformed]

        for cname, cres in self.results.items():
            if cname in ("kmeans", "dbscan", "hierarchical") and "labels" in cres:
                result_data[f"cluster_{cname}"] = cres["labels"]

        self.dr_models["tsne"] = model
        result = {
            "algorithm": "t-SNE",
            "n_components": n_components,
            "perplexity": actual_perp,
            "data": result_data,
        }
        self.results["tsne"] = result
        return result

    # ===== FULL ANALYSIS PIPELINE =====

    def run_full_analysis(self, df: pd.DataFrame) -> dict:
        results = {}

        # Auto-select optimal K
        results["optimal_k"] = self.auto_select_k(df)

        # Run all clustering
        for k in [2, 3, 4, 5]:
            if k == results["optimal_k"]["optimal_k"]:
                results[f"kmeans_k{k}"] = self.run_kmeans(df, n_clusters=k)

        # Default KMeans
        results["kmeans"] = self.run_kmeans(df, n_clusters=4)

        # DBSCAN
        results["dbscan"] = self.run_dbscan(df, eps=0.5, min_samples=5)

        # Hierarchical
        results["hierarchical"] = self.run_hierarchical(df, n_clusters=4)

        # Anomaly detection
        results["iforest"] = self.detect_anomalies_iforest(df, contamination=0.1)
        results["lof"] = self.detect_anomalies_lof(df, contamination=0.1)

        # Dimensionality reduction
        results["pca"] = self.run_pca(df, n_components=2)
        results["tsne"] = self.run_tsne(df, n_components=2)

        return results

    def predict_cluster(self, data: dict, algorithm: str = "kmeans") -> dict:
        if algorithm not in self.clustering_models:
            return {"error": f"Model {algorithm} not trained"}
        model = self.clustering_models[algorithm]
        df = pd.DataFrame([data])
        X_scaled, _ = self._prepare_data(df)
        cluster = int(model.predict(X_scaled)[0])
        return {
            "algorithm": algorithm,
            "cluster": cluster,
            "is_anomaly": cluster == -1 if algorithm == "dbscan" else False,
        }

    def predict_anomaly(self, data: dict, algorithm: str = "iforest") -> dict:
        if algorithm not in self.anomaly_models:
            return {"error": f"Model {algorithm} not trained"}
        model = self.anomaly_models[algorithm]
        df = pd.DataFrame([data])
        X_scaled, _ = self._prepare_data(df)
        pred = int(model.predict(X_scaled)[0])
        score = float(model.score_samples(X_scaled)[0]) if hasattr(model, "score_samples") else 0
        return {
            "algorithm": algorithm,
            "is_anomaly": pred == -1,
            "anomaly_score": round(score, 4),
            "label": "ANOMALY" if pred == -1 else "NORMAL",
        }
