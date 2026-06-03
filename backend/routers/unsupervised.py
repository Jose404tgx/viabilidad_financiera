from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.unsupervised_engine import UnsupervisedEngine
from config import DATASET_VIAB

router = APIRouter(prefix="/api/v1/unsupervised", tags=["Unsupervised Learning"])
engine = UnsupervisedEngine()

# Load dataset for analysis
try:
    df = pd.read_csv(str(DATASET_VIAB), encoding='utf-8-sig')
except Exception as e:
    print(f"Warning: Could not load dataset for unsupervised: {e}")
    df = pd.DataFrame()


@router.get("/status")
def status():
    return jsonable_encoder({
        "success": True,
        "data": {
            "status": "ready",
            "dataset_size": len(df),
            "models_trained": {
                "clustering": list(engine.clustering_models.keys()),
                "anomaly": list(engine.anomaly_models.keys()),
                "dim_reduction": list(engine.dr_models.keys()),
            },
        }
    })


@router.get("/clustering/kmeans")
def kmeans_clustering(
    n_clusters: int = Query(4, ge=2, le=10),
):
    if df.empty:
        raise HTTPException(400, "Dataset not loaded")
    try:
        result = engine.run_kmeans(df, n_clusters=n_clusters)
        return jsonable_encoder({"success": True, "data": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/clustering/dbscan")
def dbscan_clustering(
    eps: float = Query(0.5, ge=0.1, le=5.0),
    min_samples: int = Query(5, ge=2, le=50),
):
    if df.empty:
        raise HTTPException(400, "Dataset not loaded")
    try:
        result = engine.run_dbscan(df, eps=eps, min_samples=min_samples)
        return jsonable_encoder({"success": True, "data": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/clustering/hierarchical")
def hierarchical_clustering(
    n_clusters: int = Query(4, ge=2, le=10),
):
    if df.empty:
        raise HTTPException(400, "Dataset not loaded")
    try:
        result = engine.run_hierarchical(df, n_clusters=n_clusters)
        return jsonable_encoder({"success": True, "data": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/clustering/optimal-k")
def optimal_k():
    if df.empty:
        raise HTTPException(400, "Dataset not loaded")
    try:
        result = engine.auto_select_k(df)
        return jsonable_encoder({"success": True, "data": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/anomaly/isolation-forest")
def isolation_forest(
    contamination: float = Query(0.1, ge=0.01, le=0.5),
):
    if df.empty:
        raise HTTPException(400, "Dataset not loaded")
    try:
        result = engine.detect_anomalies_iforest(df, contamination=contamination)
        return jsonable_encoder({"success": True, "data": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/anomaly/lof")
def lof_anomaly(
    contamination: float = Query(0.1, ge=0.01, le=0.5),
    n_neighbors: int = Query(20, ge=5, le=100),
):
    if df.empty:
        raise HTTPException(400, "Dataset not loaded")
    try:
        result = engine.detect_anomalies_lof(df, contamination=contamination, n_neighbors=n_neighbors)
        return jsonable_encoder({"success": True, "data": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/dim-reduction/pca")
def pca_analysis(
    n_components: int = Query(2, ge=2, le=10),
):
    if df.empty:
        raise HTTPException(400, "Dataset not loaded")
    try:
        engine.run_kmeans(df, n_clusters=4)
        result = engine.run_pca(df, n_components=n_components)
        return jsonable_encoder({"success": True, "data": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/dim-reduction/tsne")
def tsne_analysis(
    n_components: int = Query(2, ge=2, le=3),
    perplexity: int = Query(30, ge=5, le=100),
):
    if df.empty:
        raise HTTPException(400, "Dataset not loaded")
    try:
        engine.run_kmeans(df, n_clusters=4)
        result = engine.run_tsne(df, n_components=n_components, perplexity=perplexity)
        return jsonable_encoder({"success": True, "data": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/predict-cluster")
def predict_cluster(client_data: dict, algorithm: str = Query("kmeans")):
    try:
        result = engine.predict_cluster(client_data, algorithm)
        return jsonable_encoder({"success": True, "data": result})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/predict-anomaly")
def predict_anomaly(client_data: dict, algorithm: str = Query("iforest")):
    try:
        result = engine.predict_anomaly(client_data, algorithm)
        return jsonable_encoder({"success": True, "data": result})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/full-analysis")
def full_analysis():
    if df.empty:
        raise HTTPException(400, "Dataset not loaded")
    try:
        result = engine.run_full_analysis(df)
        return jsonable_encoder({"success": True, "data": result, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(500, str(e))
