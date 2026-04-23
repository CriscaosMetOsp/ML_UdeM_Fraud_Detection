# Fraud Detection MLOps

## Introducción

Este proyecto implementa una solución completa de detección de fraude en transacciones con tarjeta de crédito. Combina el entrenamiento de modelos, la optimización de hiperparámetros, la API de predicción y un frontend en Streamlit, con monitoreo en Prometheus, Grafana y MLflow.

El objetivo es desplegar una aplicación confiable que permita evaluar transacciones y detectar comportamientos sospechosos, cuidando la trazabilidad de los experimentos y la calidad del modelo.

## Dataset

### Fuente
El dataset proviene de Kaggle: **[Credit Card Fraud Dataset](https://www.kaggle.com/datasets/dhruvb2028/credit-card-fraud-dataset)**

Contiene 339,607 transacciones de tarjeta de crédito con 0.52% de casos de fraude (1,754 transacciones fraudulentas).

### Estructura del Dataset

| Variable | Tipo | Descripción | Rango/Ejemplo |
|----------|------|-------------|---------------|
| `trans_date_trans_time` | datetime | Fecha y hora de la transacción | 2019-2020 |
| `cc_num` | int | Número de tarjeta de crédito (enmascarado) | Identificador único |
| `merchant` | str | Nombre del comerciante | Texto variable |
| `category` | str | Categoría de compra | shopping_net, shopping_pos, gas_transport, etc. |
| `amt` | float | Monto de la transacción en USD | 0.01 - 28,948.62 |
| `first` | str | Nombre del titular | Texto variable |
| `last` | str | Apellido del titular | Texto variable |
| `street` | str | Dirección del cliente | Texto variable |
| `city` | str | Ciudad del cliente | Texto variable |
| `state` | str | Estado del cliente | Códigos de estado (AL, NY, CA, etc.) |
| `zip` | str | Código postal | Numérico |
| `lat` | float | Latitud del cliente | -90 a 90 |
| `long` | float | Longitud del cliente | -180 a 180 |
| `city_pop` | int | Población de la ciudad del cliente | 0 - 13,200,000 |
| `job` | str | Ocupación del cliente | Texto variable |
| `dob` | datetime | Fecha de nacimiento del cliente | Año de nacimiento |
| `merch_lat` | float | Latitud del comerciante | -90 a 90 |
| `merch_long` | float | Longitud del comerciante | -180 a 180 |
| `is_fraud` | int | Indicador de fraude (objetivo) | 0 (legítimo) o 1 (fraude) |

## Requisitos

- Python 3.10 o superior
- Docker
- Docker Compose
- `data/raw/credit_card_frauds.csv`
- Archivos de modelo:
  - `models/best_model.pkl`
  - `models/scaler.pkl`

## Instalación

### 0. Clonar el repositorio

Abre una terminal y clona el proyecto:

```bash
git clone https://github.com/usuario/ML_UdeM_Fraud_Detection.git
cd ML_UdeM_Fraud_Detection
```

### 1. Preparar el entorno local

1. Instala las dependencias principales:

```bash
uv sync
```

3. Confirma que el dataset está en `data/raw/credit_card_frauds.csv`.

### 2. Verificar el modelo

El frontend y la API dependen de los archivos:

- `models/best_model.pkl`
- `models/scaler.pkl`

Si no existen, genera el modelo siguiendo el procedimiento de entrenamiento.

## Uso

El proyecto puede ejecutarse de dos maneras:

- Localmente con Streamlit
- Con Docker y Docker Compose

### Uso local

1. Instala las dependencias y ejecuta la app con `uv`:

```bash
uv sync
uv run streamlit run app.py
```

2. Abre el navegador en:

```text
http://localhost:8501
```

### Uso con Docker

1. Construye el servicio de Streamlit:

```bash
docker compose build
```

2. Inicia el servicio:

```bash
docker compose up -d 
```

3. Verifica la aplicación en:

```text
http://localhost:8501
```

4. Consulta los logs en tiempo real:

```bash
docker compose logs -f fraud-streamlit
```

5. Detén el stack:

```bash
docker compose down
```

> Nota: en Docker Compose v2 se utiliza `docker compose` sin guión.

## Procedimiento detallado

### 1. Explorar los datos

El dataset está incluido en `data/raw/credit_card_frauds.csv`. Para inspeccionarlo, usa:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Entrenamiento básico

Ejecuta el script de entrenamiento base:

```bash
python src/models/train.py
```

Este proceso crea un modelo inicial y registra los resultados en MLflow.

### 3. Optimización con Optuna

Para mejorar el modelo, ejecuta:

```bash
python src/models/hpo.py --model xgboost --trials 30
```

Opcionalmente, compara dos modelos:

```bash
python src/models/hpo.py --model both --trials 20
```

### 4. Flujo completo con Prefect

Si deseas automatizar todas las etapas, ejecuta:

```bash
python src/models/pipeline.py --model xgboost --trials 30
```

O si solo necesitas el entrenamiento sin HPO:

```bash
python src/models/pipeline.py --no-hpo
```

### 5. Despliegue de la API

Inicia la API de FastAPI con:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints disponibles:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `GET /model/info`

### 6. Monitoreo

El proyecto incluye integración con:

- Prometheus para métricas
- Grafana para dashboards
- MLflow para seguimiento de experimentos
- Evidently para detección de drift

Principales URLs:

- `http://localhost:3000` — Grafana
- `http://localhost:9090` — Prometheus
- `http://localhost:5000` — MLflow
- `http://localhost:8501` — Streamlit
- `http://localhost:8000/docs` — API

### 7. Pruebas

Instala las dependencias de desarrollo con `uv` si están disponibles y ejecuta:

```bash
uv sync --dev
uv run pytest tests/unit/ -v
```

## Arquitectura del proyecto

La estructura del repositorio es:

```
ML_UdeM_Fraud_Detection/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Configuración de CI/CD
├── .streamlit/
│   └── config.toml                   # Configuración de Streamlit
├── configs/
│   └── config.yaml                   # Parámetros del modelo y features
├── data/
│   ├── raw/
│   │   └── credit_card_frauds.csv    # Dataset original (339K registros)
│   └── processed/                    # Datos procesados (si aplica)
├── logs/
│   └── monitoring_report.json        # Reportes de monitoreo
├── mlruns/                           # Tracking de experimentos MLflow
│   ├── 0/
│   ├── 165115422552987618/          # Experimento principal
│   ├── 550687090619340603/          # Otros experimentos
│   └── models/                       # Modelos registrados
├── models/
│   ├── best_model.pkl               # Modelo entrenado
│   └── scaler.pkl                   # StandardScaler serializado
├── monitoring_reports/
│   ├── classification_*.html         # Reportes de clasificación
│   ├── drift_*.html                 # Reportes de drift
│   └── metrics_*.json               # Métricas almacenadas
├── notebooks/
│   └── 01_eda.ipynb                 # Análisis exploratorio de datos
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI endpoints
│   │   └── schemas.py               # Modelos Pydantic
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py         # Feature engineering y preprocesamiento
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                 # Entrenamiento baseline
│   │   ├── hpo.py                   # Optimización con Optuna
│   │   └── pipeline.py              # Orquestación con Prefect
│   └── monitoring/
│       ├── __init__.py
│       ├── evidently_monitor.py     # Monitoreo de drift
│       ├── prometheus.yml           # Configuración Prometheus
│       └── grafana/
│           ├── provisioning/        # Auto-provisioning de Grafana
│           └── dashboards/          # Dashboards JSON
├── tests/
│   └── unit/
│       ├── test_*.py                # Tests unitarios (38+)
│       └── conftest.py              # Configuración pytest
├── app.py                            # Aplicación Streamlit
├── predict_example.py                # Ejemplo de predicción
├── docker-compose.yml                # Orquestación de servicios
├── Dockerfile                        # Imagen multi-servicio
├── pyproject.toml                    # Configuración de proyecto y dependencias
├── uv.lock                           # Lock file de dependencias
├── .pre-commit-config.yaml           # Pre-commit hooks
├── .gitignore                        # Archivos ignorados
├── LICENSE                           # Licencia del proyecto
├── README.md                         # Este archivo
├── QUICK_START.md                    # Guía de inicio rápido
├── FRONTEND_README.md                # Documentación del frontend
└── .dockerignore                     # Archivos ignorados en Docker
```

## Solución de problemas comunes

- Si `docker compose build` falla porque falta un archivo, revisa el `Dockerfile` y confirma que el archivo exista en la raíz.
- Si Streamlit no carga, revisa que el puerto `8501` esté libre.
- Si los modelos no aparecen, genera `best_model.pkl` y `scaler.pkl` con los scripts de entrenamiento.
- Si PowerShell no permite ejecutar scripts, ajusta la política con:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Guía Rápida

1. Clona el repositorio: `git clone https://github.com/usuario/ML_UdeM_Fraud_Detection.git && cd ML_UdeM_Fraud_Detection`.
2. Instala dependencias: `uv sync`.
3. Verifica que `data/raw/credit_card_frauds.csv` exista.
4. Entrena el modelo básico: `python src/models/train.py`.
5. Optimiza con Optuna: `python src/models/hpo.py --model xgboost --trials 30`.
6. Ejecuta la app local: `streamlit run app.py`.
7. O inicia Docker: `docker compose up -d fraud-streamlit`.
8. Accede a `http://localhost:8501`.

## Créditos

Proyecto desarrollado como trabajo final de la especialización en Data Science e IA de la Universidad de Medellín.

Última actualización: Abril 2026
