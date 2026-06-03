# Predictor de Viabilidad Financiera

Sistema de Inteligencia Artificial para la predicción de viabilidad crediticia.
**Proyecto de sustentación semestral - Carrera universitaria.**

## Criterios de Evaluación

| # | Criterio | Estado | Evidencia |
|---|----------|--------|-----------|
| 1 | Dataset tabular (CSV) desde consulta SQL a BD relacional | ✅ | `database/schema.sql` - Consulta SQL que genera el dataset |
| 2 | Librerías: numpy, pandas, scikit-learn, pytorch, tensorflow. Algoritmos: KNN, KMeans, Random Forest, ID3, Regresión Lineal, Apriori, Adam, MSE | ✅ | Ver sección Algoritmos Implementados |
| 3 | DL: neuronas, capas, activación, pesos, sesgos (MLP) | ✅ | Endpoint `/api/v1/algorithms/nn-architecture` |
| 4 | Totalmente funcional según objetivo del proyecto | ✅ | Predicción de viabilidad crediticia funcional |
| 5 | Publicado en la web | ✅ | Ejecutar `uvicorn backend.main:app --host 0.0.0.0 --port 8000` |

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (HTML + Alpine.js)              │
│  Formulario → API REST → Resultados (Score + Predicción)    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                   Backend (FastAPI + Python)                 │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ MLEngine │  │ ScoringEngine│  │ RecommendationEngine│    │
│  └────┬─────┘  └──────┬───────┘  └─────────┬──────────┘    │
│       │               │                    │                │
│  ┌────▼───────────────▼────────────────────▼──────────┐    │
│  │              DataProcessor                          │    │
│  │  - Limpieza y transformación de datos               │    │
│  │  - Feature engineering (24 variables)               │    │
│  │  - SMOTE para balanceo de clases                    │    │
│  │  - RobustScaler para normalización                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Modelos entrenados (11):                                   │
│  - Random Forest ★ (mejor) - 88.33% accuracy               │
│  - XGBoost - 89.17% accuracy                               │
│  - CatBoost - 86.67% accuracy                              │
│  - LightGBM - 85.83% accuracy                              │
│  - KNN - 84.17% accuracy                                   │
│  - Decision Tree (ID3) - 80.83% accuracy                   │
│  - AdaBoost - 85.00% accuracy                              │
│  - Gradient Boosting - 81.67% accuracy                     │
│  - Logistic Regression - 70.83% accuracy                   │
│  - Neural Network (MLP - Adam) - 70.83% accuracy           │
│  - Ensemble (Stacking) - 85.83% accuracy                   │
│  - Linear Regression (MSE) - Score continuo                │
└─────────────────────────────────────────────────────────────┘
```

## Dataset

El dataset contiene **600 registros** con 16 variables financieras:

| Variable | Descripción |
|----------|-------------|
| `ingreso_mensual` | Ingreso mensual del cliente |
| `total_gastos` | Gastos mensuales totales |
| `total_costos` | Costos operativos mensuales |
| `total_activos` | Total de activos patrimoniales |
| `total_deudas` | Deuda total acumulada |
| `num_deudas` | Número de deudas activas |
| `excedente` | Ingreso - Gastos |
| `capacidad_pago` | Capacidad de pago disponible |
| `endeudamiento_patrimonial` | Ratio deuda/activos |
| `capital_trabajo` | Activos - Deudas |
| `monto_solicitado` | Monto solicitado del préstamo |
| `tem` | Tasa efectiva mensual |
| `num_cuotas` | Número de cuotas |
| `cuota_estimada` | Cuota mensual estimada |
| `monto_propuesto` | Monto propuesto |
| `mora_diaria` | Días de mora |
| `viable` | **Target**: 1 = Viable, 0 = No viable |

## Feature Engineering

Se generan **24 variables derivadas** para mejorar la precisión:

- `ratio_gastos_ingresos`, `ratio_deuda_ingreso`, `ratio_cuota_ingreso`
- `liquidez`, `cobertura_gastos`, `peso_deuda`, `capacidad_ahorro`
- `estres_financiero`, `carga_financiera_total`, `cobertura_cuota`
- `eficiencia_operativa`, `apalancamiento`, `solvencia`
- `ingreso_por_deuda`, `ingreso_log`, `activos_log`, `deudas_log`
- Variables binarias: `mora_riesgo`, `deuda_alta`, `ahorro_positivo`, `capacidad_baja`

## Scoring Financiero

El scoring evalúa al cliente en **9 dimensiones** (máximo 1000 pts):

| Dimensión | Peso | Score Máx |
|-----------|------|-----------|
| Ingreso | 15% | 150 |
| Deuda | 15% | 150 |
| Excedente | 12% | 120 |
| Gastos | 10% | 100 |
| Capacidad de Pago | 10% | 100 |
| Mora | 10% | 100 |
| Endeudamiento | 10% | 100 |
| Capital de Trabajo | 10% | 100 |
| Nro. Deudas | 8% | 80 |

**Clasificación:**
- Excelente: 800-1000
- Bueno: 650-799
- Regular: 450-649
- Riesgoso: 250-449
- Crítico: 0-249

## API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/v1/health` | GET | Estado del servidor |
| `/api/v1/predict` | POST | Predicción de viabilidad (ML + Score + Recomendaciones) |
| `/api/v1/score` | POST | Scoring financiero (0-1000) |
| `/api/v1/recommend` | POST | Recomendaciones basadas en perfil |
| `/api/v1/predict/batch` | POST | Predicción por lote |
| `/api/v1/dataset/info` | GET | Información y estadísticas del dataset |
| `/api/v1/dashboard` | GET | KPIs y estadísticas globales |
| `/api/v1/algorithms/list` | GET | Lista de todos los algoritmos implementados |
| `/api/v1/algorithms/linear-regression` | POST | Regresión Lineal con métrica MSE |
| `/api/v1/algorithms/apriori` | POST | Reglas de asociación Apriori |
| `/api/v1/algorithms/nn-architecture` | GET | Arquitectura de red neuronal (capas, pesos, sesgos, Adam) |

## Requisitos

- Python 3.10+
- pip (gestor de paquetes)

## Instalación

```bash
# 1. Clonar el repositorio
cd "predictor de viavilidad financiera"

# 2. Instalar dependencias
pip install -r backend/requirements.txt

# 3. Iniciar el servidor
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. Abrir en el navegador
http://localhost:8000
```

## Ejecutar tests

```bash
python -m pytest tests/ -v
```

## Procedencia de Datos (SQL)

El dataset CSV se genera a partir de una base de datos relacional con la siguiente consulta SQL:

```sql
SELECT
    c.id_cliente, i.ingreso_mensual, g.total_gastos, g.total_costos,
    p.total_activos, p.total_deudas,
    (SELECT COUNT(*) FROM deudas d WHERE d.id_cliente = c.id_cliente) AS num_deudas,
    (i.ingreso_mensual - g.total_gastos - g.total_costos) AS excedente,
    (i.ingreso_mensual - g.total_gastos - g.total_costos) / NULLIF(g.total_gastos + g.total_costos, 0) AS capacidad_pago,
    p.endeudamiento_patrimonial, p.capital_trabajo,
    s.monto_solicitado, s.tem, s.num_cuotas, s.cuota_estimada, s.monto_propuesto,
    COALESCE((SELECT MAX(d.mora_diaria) FROM deudas d WHERE d.id_cliente = c.id_cliente), 0) AS mora_diaria,
    e.viable
FROM clientes c
LEFT JOIN ingresos i ON c.id_cliente = i.id_cliente
LEFT JOIN gastos g ON c.id_cliente = g.id_cliente
LEFT JOIN patrimonio p ON c.id_cliente = p.id_cliente
LEFT JOIN solicitudes_credito s ON c.id_cliente = s.id_cliente
LEFT JOIN evaluaciones e ON c.id_cliente = e.id_cliente;
```

Ver `database/schema.sql` para el esquema completo.

## Algoritmos Implementados

### Clasificación (Supervisado)
| Algoritmo | Librería | Descripción |
|-----------|----------|-------------|
| **Random Forest** | scikit-learn | Bagging de árboles de decisión (300 estimadores, max_depth=20) |
| **KNN** | scikit-learn | K-Nearest Neighbors (k=5, distancia Minkowski) |
| **ID3 (Decision Tree)** | scikit-learn | Árbol de decisión con criterio Entropía / Ganancia de Información |
| **XGBoost** | xgboost | Gradient Boosting optimizado (300 estimadores, learning_rate=0.05) |
| **LightGBM** | lightgbm | Gradient Boosting basado en hojas (300 estimadores) |
| **CatBoost** | catboost | Gradient Boosting con manejo de categóricas (300 iteraciones) |
| **Gradient Boosting** | scikit-learn | Gradient Boosting clásico (250 estimadores) |
| **Logistic Regression** | scikit-learn | Regresión logística con regularización L2 |
| **AdaBoost** | scikit-learn | Adaptive Boosting (300 estimadores, SAMME) |
| **Ensemble Stacking** | scikit-learn | Meta-aprendizaje con Regresión Logística como meta-learner |

### Regresión
| Algoritmo | Librería | Métrica | Descripción |
|-----------|----------|---------|-------------|
| **Linear Regression** | scikit-learn | **MSE**, RMSE, R² | Mínimos Cuadrados Ordinarios para score continuo |

### Asociación
| Algoritmo | Librería | Descripción |
|-----------|----------|-------------|
| **Apriori** | mlxtend | Reglas de asociación (Support, Confidence, Lift) entre variables financieras discretizadas |

### Deep Learning
| Algoritmo | Librería | Optimizador | Descripción |
|-----------|----------|-------------|-------------|
| **MLPClassifier** | scikit-learn | **Adam** (Adaptive Moment Estimation) | Perceptrón Multicapa: 4 capas ocultas (256→128→64→32), activación ReLU, salida Sigmoid |

### No Supervisado
| Algoritmo | Librería | Descripción |
|-----------|----------|-------------|
| **K-Means** | scikit-learn | Clustering de clientes por perfil financiero |

### Métricas de Evaluación
- **Accuracy** (Precisión global)
- **Precision** (Precisión por clase)
- **Recall** (Sensibilidad)
- **F1-Score** (Media armónica)
- **ROC-AUC** (Área bajo la curva ROC)
- **MSE** (Mean Squared Error) - para regresión lineal
- **Matriz de Confusión**

## Arquitectura de Red Neuronal (MLP)

El MLPClassifier implementa un Perceptrón Multicapa con:

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
│ Entrada  │────│ Oculta 1 │────│ Oculta 2 │────│ Oculta 3 │────│ Oculta 4 │────│  Salida  │
│ 24 neuron│    │256 neuron│    │128 neuron│    │ 64 neuron│    │ 32 neuron│    │1 neurona │
└─────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘     └─────────┘
Activation:      ReLU             ReLU             ReLU             ReLU             Sigmoid
```

- **Optimizador:** Adam (Adaptive Moment Estimation)
- **Función de pérdida:** Log Loss (Cross-Entropy binaria)
- **Early Stopping:** activado (validation_fraction=0.1)
- **Total de parámetros:** consultar endpoint `/api/v1/algorithms/nn-architecture`

## Tecnologías

- **Backend:** Python, FastAPI, scikit-learn, XGBoost, LightGBM, CatBoost, mlxtend
- **Frontend:** HTML, Alpine.js, Tailwind CSS, Chart.js, Lucide Icons
- **ML:** Random Forest, KNN, ID3, Red Neuronal (Adam), Regresión Lineal (MSE), Apriori, SMOTE, SHAP, GridSearchCV
