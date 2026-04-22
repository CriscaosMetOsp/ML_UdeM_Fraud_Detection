# 🔐 Credit Card Fraud Detection - MLOps End-to-End

**Universidad de Medellín | Specialización en Data Science e IA**
**Proyecto Final - MLOps**

---

## 📋 Problema de Negocio

Las entidades financieras pierden miles de millones de dólares anuales por fraude en tarjetas de crédito. El reto es detectar transacciones fraudulentas en tiempo real con alta sensibilidad (*recall*), minimizando el impacto al cliente legítimo y el costo operativo de falsos positivos.

**Dataset:** 339,607 transacciones (2019–2020) con 0.52% de fraude.

**Métricas de éxito:**
| Métrica | Objetivo | Obtenido |
|---|---|---|
| PR-AUC | > 0.85 | **0.9087** ✅ |
| Recall | > 0.80 | **0.8483** ✅ |
| F1 | > 0.50 | **0.8543** ✅ |

---

## 🏗️ Arquitectura

```
fraud-mlops/
├── src/
│   ├── data/           # Carga + feature engineering
│   ├── models/
│   │   ├── train.py    # Entrenamiento baseline + MLflow
│   │   ├── hpo.py      # Optimización con Optuna
│   │   └── pipeline.py # Orquestación con Prefect
│   ├── api/            # FastAPI REST service
│   └── monitoring/
│       ├── evidently_monitor.py  # Drift con Evidently AI
│       ├── prometheus.yml        # Config scraping
│       └── grafana/              # Dashboard JSON + provisioning
├── tests/unit/         # pytest — 38 tests
├── configs/config.yaml
├── docker-compose.yml  # Prometheus + Grafana + MLflow + API
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## ⚡ Quickstart

### 1. Instalar dependencias

**Con uv (recomendado — usa el lockfile incluido):**
```bash
pip install uv
uv sync
```

**Con pip (alternativa):**
pip install pandas numpy scikit-learn xgboost mlflow prefect fastapi uvicorn \
            pydantic imbalanced-learn joblib pyyaml scipy \
            optuna "optuna-integration[mlflow]" evidently prometheus-client
```

### 2. Dataset
El dataset ya está incluido en `data/raw/credit_card_frauds.csv`. No se requiere ningún paso adicional.

### 3. Explorar datos
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Entrenar baseline (MLflow tracking)
```bash
python src/models/train.py
mlflow ui --backend-store-uri mlruns   # http://localhost:5000
```

### 5. HPO con Optuna
```bash
# XGBoost — 30 trials (recomendado para producción)
python src/models/hpo.py --model xgboost --trials 30

# Comparar ambos modelos
python src/models/hpo.py --model both --trials 20
```

### 6. Pipeline completo con Prefect
```bash
# Con HPO (recomendado)
python src/models/pipeline.py --model xgboost --trials 30

# Sin HPO (solo baseline)
python src/models/pipeline.py --no-hpo
```

---

## 📦 Pipeline Completo con Prefect (`pipeline.py`)

El script **`src/models/pipeline.py`** es el orquestador principal que automatiza todo el flujo de ML:

**Datos → Feature Engineering → Split & Scale → Baseline → [Opcional] HPO → Comparar & Registrar**

### Tasks (Unidades Independientes)

| Task | Función | Output |
|---|---|---|
| **load-data** | Carga CSV + validación | DataFrame |
| **feature-engineering** | Construye X, y | (X, y) |
| **split-and-scale** | Train/test split + normalización | (X_train, X_test, y_train, y_test) |
| **baseline-training** | Entrena RF + XGB con config.yaml | (best_model, metrics, run_id, name) |
| **hpo-optuna** | Optimización con Optuna (30 trials default) | (best_model, params, metrics, run_id) |
| **compare-and-register** | Compara baseline vs HPO, registra ganador | (winner_name, winner_metrics) |

### Flujo Completo

```
Prefect Flow: fraud-detection-full-pipeline
│
├─ task_load_data()           # Carga 339K transacciones
│
├─ task_feature_engineering()  # X: 339K×45 | y: 0.52% fraud
│
├─ task_split_scale()          # Train: 270K | Test: 69K
│                              # Scaler guardado en models/scaler.pkl
│
├─ task_baseline()             # Entrena 2 modelos baseline
│   ├─ Random Forest (config)  → PR-AUC: 0.8834
│   └─ XGBoost (config)        → PR-AUC: 0.8919 ✅ (ganador)
│
├─ [IF run_hpo] task_hpo()     # Optuna: 30 trials (default)
│   └─ XGBoost optimizado      → PR-AUC: 0.9087 ✅ (MEJOR)
│
├─ task_compare_register()      # Compara baseline vs HPO
│   └─ Improvement: +1.89%
│   └─ Ganador registrado en MLflow Model Registry
│       models/best_model.pkl salvado
│
└─ ✅ Pipeline completo
   Modelo: HPO_optimizado
   PR-AUC: 0.9087 | Recall: 0.8483 | F1: 0.8543
```

### Uso

```bash
# Baseline + HPO XGBoost (30 trials — recomendado para producción)
python src/models/pipeline.py --hpo --model xgboost --trials 30

# Baseline + HPO RandomForest (50 trials)
python src/models/pipeline.py --hpo --model random_forest --trials 50

# Solo baseline (sin optimización)
python src/models/pipeline.py --no-hpo

# Baseline + HPO en ambos modelos
python src/models/pipeline.py --hpo --model both --trials 20
```

### Parámetros

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `--hpo` / `--no-hpo` | flag | True | Ejecutar optimización con Optuna |
| `--model` | str | xgboost | Qué modelo optimizar: `xgboost`, `random_forest`, `both` |
| `--trials` | int | 30 | # de trials en Optuna |

### Integraciones

- **Prefect 2.x:** Orquestación, logging, retries, task dependencies
- **MLflow:** Tracking de todos los runs + Model Registry
- **Optuna:** Busca bayesiana de hiperparámetros (TPE sampler)
- **Joblib:** Serialización de scaler y modelo final

### Monitoreo en MLflow

Accede a `http://localhost:5000`:
- **Experimento:** `xgboost_hyperparameter_optimization`
- **Runs:** Cada trial → `params`, `metrics`, `artifacts`
- **Comparación:** Baseline vs HPO lado-a-lado
- **Registry:** Mejor modelo registrado automáticamente

---

### 8. API de predicción
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

### 9. Stack completo de monitoreo
```bash
docker compose up -d
# Grafana:    http://localhost:3000  (admin/admin)
# Prometheus: http://localhost:9090
# MLflow:     http://localhost:5000
# API:        http://localhost:8000/docs

# En paralelo — iniciar exporter de métricas
python src/monitoring/evidently_monitor.py
```

### 10. Tests
```bash
pytest tests/unit/ -v   # 38 tests
```

---

## 🔬 Tecnologías

| Categoría | Tecnología |
|---|---|
| **ML** | scikit-learn, XGBoost, imbalanced-learn |
| **HPO** | **Optuna** (TPE sampler + MedianPruner) |
| **Experiment Tracking** | MLflow (tracking + Model Registry) |
| **Orquestación** | Prefect 2.x |
| **API** | FastAPI + Uvicorn |
| **Drift Detection** | **Evidently AI** (DataDrift + Classification presets) |
| **Métricas** | **Prometheus Client** (Gauges expuestos en /metrics) |
| **Dashboards** | **Grafana** (dashboard JSON + auto-provisioning) |
| **Testing** | pytest (38 tests) |
| **CI/CD** | GitHub Actions |
| **Contenedor** | Docker + Docker Compose |
| **Code Quality** | black, flake8, pre-commit |

---

## 🔍 HPO con Optuna — Detalle

El módulo `src/models/hpo.py` implementa:

- **Sampler:** TPE (Tree-structured Parzen Estimator) — bayesiano, converge más rápido que random search
- **Pruner:** MedianPruner — elimina trials malos temprano (con `n_warmup_steps=5`)
- **Métrica objetivo:** PR-AUC vía `StratifiedKFold(n_splits=3)` — robusta ante el desbalanceo severo
- **MLflow callback:** cada trial queda como un run individual en MLflow para análisis posterior
- **Space de búsqueda XGBoost:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`
- **Space de búsqueda RF:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`

**Importancia de hiperparámetros** (obtenida con `optuna.importance`):

| Param | Importancia |
|---|---|
| learning_rate | 0.321 |
| max_depth | 0.181 |
| min_child_weight | 0.153 |
| scale_pos_weight | 0.122 |
| gamma | 0.075 |

---

## 📊 Monitoreo con Evidently + Grafana

### Stack

```
Modelo → prometheus_client (puerto 8001)
              ↓ scrape cada 30s
         Prometheus (puerto 9090)
              ↓ datasource
         Grafana (puerto 3000)
              + dashboard auto-provisionado
```

### Qué monitorea Evidently

- **DataDriftPreset:** KS test en todas las features numéricas + score de fraude
- **ClassificationPreset:** precision, recall, F1, PR-AUC en ventana de producción
- Reportes HTML guardados en `monitoring_reports/` con timestamp

### Métricas en Grafana

| Gauge Prometheus | Descripción |
|---|---|
| `fraud_model_drift_share` | Fracción de features con drift (p<0.05) |
| `fraud_model_score_drift_pvalue` | p-value KS del score de fraude |
| `fraud_model_recall` | Recall en ventana actual |
| `fraud_model_pr_auc` | PR-AUC en ventana actual |
| `fraud_model_usd_detected` | USD en fraude detectado |
| `fraud_model_usd_missed` | USD en fraude no detectado |
| `fraud_model_false_positives` | Falsos positivos en ventana |

---

## 📈 Resultados

| Etapa | Modelo | PR-AUC | Recall | F1 | Fraudes/Total |
|---|---|---|---|---|---|
| Baseline | XGBoost | 0.8919 | 0.9073 | 0.6377 | 323/356 |
| **HPO (8 trials)** | **XGBoost** | **0.9087** | **0.8483** | **0.8543** | 302/356 |

**Impacto de negocio:** $172,476 detectados vs $1,747 perdidos → 99% de detección en USD.

---

## 🌐 API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado del modelo |
| POST | `/predict` | Puntuar 1 transacción |
| POST | `/predict/batch` | Puntuar múltiples |
| GET | `/model/info` | Info del modelo activo |

---

## 📅 Timeline

| Fase | Actividad | Días |
|---|---|---|
| 1 | EDA + feature engineering | 3 |
| 2 | Baseline + MLflow tracking | 2 |
| 3 | HPO con Optuna | 2 |
| 4 | Pipeline Prefect | 2 |
| 5 | FastAPI deployment | 2 |
| 6 | Evidently + Prometheus + Grafana | 3 |
| 7 | Tests + CI/CD + Docker | 2 |
| 8 | Documentación | 1 |

---

## 🤝 Evaluación por Pares

```bash
# 1. Instalar y poner dataset
pip install -r requirements.txt   # o poetry install
cp credit_card_frauds.csv data/raw/

# 2. Tests
pytest tests/unit/ -v

# 3. HPO + entrenamiento
python src/models/hpo.py --model xgboost --trials 10

# 4. API
uvicorn src.api.main:app --port 8000
curl http://localhost:8000/health

# 5. Stack de monitoreo
docker compose up -d
# Grafana: http://localhost:3000 → dashboard "Fraud Detection Model"
```
