# 🚀 GUÍA RÁPIDA - Frontend Streamlit

## ⚡ Inicio Rápido en 3 Pasos

### Opción 1: Ejecución Local (Rápido)

```bash
# 1. Instalar dependencias
pip install streamlit pyyaml

# 2. Ejecutar la aplicación
streamlit run app.py

# 3. Abrir en navegador
# http://localhost:8501
```

### Opción 2: Ejecución con Docker (Recomendado)

```bash
# 1. Iniciar el servicio
docker compose up -d fraud-streamlit

# 2. Abrir en navegador
# http://localhost:8501

# 3. Ver logs
docker compose logs -f fraud-streamlit

# 4. Detener
docker compose down
```

---

## 📊 Acceder a Todos los Servicios

```bash
# Iniciar stack completo
docker compose up -d

# Luego accede a:
# 🌐 Frontend Streamlit:    http://localhost:8501
# 📊 Grafana:              http://localhost:3000 (admin/admin)
# 🔍 Prometheus:           http://localhost:9090
# 🤖 MLflow:               http://localhost:5000
# 🔌 API FastAPI:          http://localhost:8000/docs
```

---

## 🎯 Cómo Usar la Interfaz

1. **Ingresa los parámetros** de la transacción
2. **Haz clic** en "🎯 Realizar Predicción"
3. **Visualiza** el resultado

### Parámetros Necesarios

| Categoría | Parámetros |
|-----------|-----------|
| 💳 Transacción | Monto, Hora, Día de la semana |
| 📍 Cliente | Latitud, Longitud, Edad, Estado |
| 🏙️ Comercio | Latitud, Longitud, Categoría |
| 🗺️ Ubicación | Población de ciudad |

---

## 🔧 Comandos Útiles

```bash
# Ver logs en tiempo real
docker compose logs -f fraud-streamlit

# Reconstruir imagen
docker compose build fraud-streamlit

# Reiniciar servicio
docker compose restart fraud-streamlit

# Acceder a contenedor
docker compose exec fraud-streamlit bash

# Ver estado de servicios
docker compose ps
```

---

## 📁 Archivos Importantes

```
app.py                    ← Aplicación Streamlit (EDITAR AQUÍ para cambios)
Dockerfile.streamlit      ← Configuración del contenedor
.streamlit/config.toml    ← Configuración de Streamlit
configs/config.yaml       ← Parámetros del modelo
models/
  ├── best_model.pkl      ← Modelo entrenado
  └── scaler.pkl          ← Escalador de features
```

---

## ⚠️ Solución de Problemas

### "ModuleNotFoundError"
```bash
# Reinstalar dependencias
pip install -e .
```

### "FileNotFoundError: models/best_model.pkl"
```bash
# Asegurate que existan los archivos del modelo
ls -la models/
```

### Contenedor no inicia
```bash
# Ver errores detallados
docker compose logs fraud-streamlit

# Reconstruir
docker compose build --no-cache fraud-streamlit
docker compose up -d fraud-streamlit
```

---

## 📈 Ejemplos de Transacciones

### ✅ Transacción Legítima
- Monto: $50
- Hora: 14:00
- Distancia: 2 km
- Categoría: shopping_pos

### ⚠️ Transacción Fraudulenta
- Monto: $2000
- Hora: 03:00
- Distancia: 500 km
- Categoría: shopping_net

---

## 📞 Contacto

Para soporte, revisa:
- [FRONTEND_README.md](FRONTEND_README.md) - Documentación completa
- [README.md](README.md) - Documentación del proyecto

---

**Versión:** 1.0  
**Fecha:** Abril 2026  
**Estado:** ✅ Listo para usar
