# 🔍 Frontend Streamlit - Fraud Detection Model

Interfaz web interactiva para realizar predicciones de fraude en transacciones de tarjeta de crédito utilizando el modelo entrenado.

## 📋 Características

- ✅ **Interfaz intuitiva** - Ingresa parámetros de la transacción fácilmente
- ✅ **Predicción en tiempo real** - Obtén resultados al instante
- ✅ **Visualización de probabilidades** - Ve el nivel de confianza de la predicción
- ✅ **Información de ubicación** - Cálculo automático de distancia cliente-comercio
- ✅ **Containerizado** - Ejecución fácil mediante Docker

## 🚀 Inicio Rápido

### Opción 1: Ejecutar localmente (sin Docker)

#### Prerrequisitos
- Python 3.10+
- Modelo entrenado guardado en `models/best_model.pkl`
- Scaler guardado en `models/scaler.pkl`

#### Instalación

```bash
# Instalar dependencias
pip install -e .

# O usar uv si está disponible
uv pip install -e .
```

#### Ejecutar la app

```bash
streamlit run app.py
```

La aplicación estará disponible en: **http://localhost:8501**

### Opción 2: Ejecutar con Docker (recomendado)

#### Requisitos
- Docker y Docker Compose instalados

#### Ejecutar todo el stack

```bash
# Iniciar todos los servicios
docker compose up -d

# O solo el servicio de Streamlit
docker compose up -d fraud-streamlit
```

Accede a la interfaz en: **http://localhost:8501**

#### Ver logs
```bash
docker compose logs -f fraud-streamlit
```

#### Detener los servicios
```bash
docker compose down
```

## 🎯 Cómo Usar

### 1. Ingresar Parámetros de la Transacción

**Información de la Transacción:**
- **Monto (USD)** - Cantidad de la transacción (0.01 - sin límite)
- **Hora del día** - 0-23 (0=medianoche, 12=mediodía)
- **Día de la semana** - 0-6 (0=Lunes, 6=Domingo)

**Ubicaciones:**
- **Latitud/Longitud del cliente** - Ubicación del titular de la tarjeta
- **Población de la ciudad** - Habitantes de la ciudad del cliente
- **Latitud/Longitud del comercio** - Ubicación del establecimiento

**Información del Comercio:**
- **Categoría** - Tipo de comercio:
  - shopping_net (compra en línea)
  - shopping_pos (compra en punto de venta)
  - gas_transport (gasolina/transporte)
  - grocery_pos (supermercado presencial)
  - grocery_net (supermercado en línea)
  - entertainment (entretenimiento)
  - misc_net (miscelánea en línea)
  - misc_pos (miscelánea presencial)

**Información del Cliente:**
- **Edad** - 18-100 años
- **Estado** - Código de estado (AL, NY, CA, etc.)

### 2. Realizar Predicción

1. Completa todos los campos
2. Haz clic en el botón **"🎯 Realizar Predicción"**
3. Visualiza los resultados

### 3. Interpretar Resultados

La aplicación mostrará:
- ✅ **Estado** - "FRAUDE DETECTADO" o "TRANSACCIÓN LEGÍTIMA"
- 📊 **Confianza** - Nivel de certeza de la predicción (%)
- 📈 **Probabilidades** - Desglose de probabilidades
- 📋 **Resumen** - Todos los parámetros ingresados

## 🐳 Servicios Disponibles en Docker Compose

| Servicio | Puerto | URL |
|----------|--------|-----|
| **Streamlit Frontend** | 8501 | http://localhost:8501 |
| FastAPI Backend | 8000 | http://localhost:8000/docs |
| Prometheus Metrics | 9090 | http://localhost:9090 |
| Grafana Dashboards | 3000 | http://localhost:3000 |
| MLflow UI | 5000 | http://localhost:5000 |

## 📁 Estructura de Archivos

```
app.py                          # Aplicación Streamlit
Dockerfile.streamlit            # Dockerfile para Streamlit
docker-compose.yml              # Orquestación de servicios
configs/config.yaml             # Configuración del modelo
models/
  ├── best_model.pkl           # Modelo entrenado
  └── scaler.pkl               # StandardScaler entrenado
src/
  ├── data/preprocessing.py     # Funciones de preprocesamiento
  ├── models/                   # Módulo de modelos
  └── monitoring/               # Monitoreo y métricas
```

## 🔧 Configuración

### Variables de Entorno (Docker)

```env
PYTHONUNBUFFERED=1
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Archivo de Configuración (`configs/config.yaml`)

```yaml
features:
  numerical_cols:
    - "amt"
    - "lat"
    - "long"
    - "city_pop"
    - "merch_lat"
    - "merch_long"
    - "hour"
    - "day_of_week"
    - "age"
    - "distance_km"
  categorical_cols:
    - "category"
    - "state"
```

## 📊 Ejemplo de Uso

1. **Escenario 1: Transacción Normal**
   - Monto: $50
   - Hora: 14:00 (tarde)
   - Distancia: 2 km
   - **Resultado:** ✅ Legítima (98% confianza)

2. **Escenario 2: Transacción Sospechosa**
   - Monto: $2000
   - Hora: 03:00 (madrugada)
   - Distancia: 500 km
   - **Resultado:** ⚠️ Fraude (87% confianza)

## 🆘 Solución de Problemas

### Error: "No se encontraron los archivos del modelo"
```
✅ Solución: Asegúrate de que existen:
  - models/best_model.pkl
  - models/scaler.pkl
```

### Error de conexión en Docker
```bash
# Verifica que los contenedores estén en ejecución
docker compose ps

# Reinicia el servicio
docker compose restart fraud-streamlit
```

### Streamlit muy lento
```bash
# Limpia el caché de Streamlit
streamlit cache clear

# O elimina el directorio de caché
rm -rf ~/.streamlit/
```

## 📈 Monitoreo

Para monitorear el rendimiento del modelo:
- **Grafana:** http://localhost:3000 (admin/admin)
- **MLflow:** http://localhost:5000

## 🔄 Actualizar el Modelo

Si actualizas el modelo entrenado:

1. Reemplaza los archivos en `models/`:
   - `best_model.pkl`
   - `scaler.pkl`

2. Si usas Docker, reconstruye la imagen:
   ```bash
   docker compose build fraud-streamlit
   docker compose up -d fraud-streamlit
   ```

3. Si ejecutas localmente, simplemente reinicia la app:
   ```bash
   # Presiona Ctrl+C en la terminal
   # Luego ejecuta
   streamlit run app.py
   ```

## 📝 Logs

### Localmente
Los logs se muestran en la terminal donde ejecutas Streamlit.

### Con Docker
```bash
# Ver logs en tiempo real
docker compose logs -f fraud-streamlit

# Ver últimas 100 líneas
docker compose logs --tail=100 fraud-streamlit
```

## 🔐 Consideraciones de Seguridad

- El frontend no almacena datos de entrada
- Las predicciones se generan localmente
- No se envían datos a servidores externos
- Usa variables de entorno para configuraciones sensibles

## 🛠️ Desarrollo

### Modificar la interfaz
Edita `app.py` directamente. Streamlit recargará automáticamente.

### Agregar nuevas características
1. Modifica `app.py`
2. Reconstruye la imagen Docker si es necesario
3. Redeploy

### Testing
```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar tests
pytest tests/
```

## 📞 Soporte

Para reportar problemas o sugerencias:
1. Verifica los logs
2. Revisa la configuración
3. Contacta al equipo de desarrollo

---

**Versión:** 0.1.0  
**Última actualización:** Abril 2026  
**Estado:** ✅ Producción
