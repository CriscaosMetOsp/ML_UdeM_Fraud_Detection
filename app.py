"""
Streamlit Frontend para Fraud Detection Model
Interfaz interactiva para hacer predicciones de fraude en transacciones de tarjeta de crédito.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from pathlib import Path

# Configuración de Streamlit
st.set_page_config(
    page_title="Fraud Detection Model",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema y estilos
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Cargar modelo y scaler desde archivos."""
    try:
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        config = yaml.safe_load(open("configs/config.yaml"))
        return model, scaler, config
    except FileNotFoundError as e:
        st.error(f"❌ Error: No se encontraron los archivos del modelo. {e}")
        st.stop()


def load_config(path="configs/config.yaml"):
    """Cargar configuración."""
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_category_encodings():
    """Obtener los encodings de categorías y estados."""
    # Mapeo de categorías a números (basado en el dataset de entrenamiento)
    categories = {
        "shopping_net": 0,
        "shopping_pos": 1,
        "gas_transport": 2,
        "grocery_pos": 3,
        "grocery_net": 4,
        "entertainment": 5,
        "misc_net": 6,
        "misc_pos": 7,
    }
    
    # Mapeo de estados a números (basado en orden alfabético)
    states = sorted(["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA",
                     "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD",
                     "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
                     "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
                     "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"])
    
    state_mapping = {state: idx for idx, state in enumerate(states)}
    
    return categories, state_mapping


def preprocess_input(inputs: dict, scaler, config: dict, model=None) -> pd.DataFrame:
    """
    Preprocesar inputs del usuario.
    
    Parameters
    ----------
    inputs : dict con los valores ingresados
    scaler : StandardScaler fitted
    config : configuración del proyecto
    
    Returns
    -------
    X_processed : DataFrame listo para predicción
    """
    # Obtener encodings
    categories, state_mapping = get_category_encodings()
    
    # Hacer copia de inputs para no modificar el original
    data = inputs.copy()
    
    # Codificar variables categóricas
    data["category"] = categories[data["category"]]
    data["state"] = state_mapping[data["state"]]
    
    # Crear DataFrame con los inputs
    X = pd.DataFrame([data])
    
    # Reordenar columnas según lo que el modelo espera
    if model is not None and hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = config["features"]["numerical_cols"] + config["features"]["categorical_cols"]
        feature_cols = [c for c in feature_cols if c != config["features"]["target"]]

    X = X[feature_cols]
    
    # Escalar features numéricos
    numerical_cols = [c for c in config["features"]["numerical_cols"] if c in X.columns]
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X


def main():
    """Función principal de la app Streamlit."""
    
    # Cargar modelo y configuración
    model, scaler, config = load_model_artifacts()
    
    # Header
    st.title("🔍 Credit Card Fraud Detection")
    st.markdown("Predice si una transacción es fraudulenta basándose en características de la transacción")
    
    st.divider()
    
    # Crear dos columnas para mejor organización
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Información de la Transacción")
        
        # Inputs numéricos - Transacción
        amt = st.number_input(
            "Monto de la transacción (USD)",
            min_value=0.01,
            value=50.0,
            step=0.01,
            help="Cantidad en dólares de la transacción"
        )
        
        hour = st.slider(
            "Hora del día",
            min_value=0,
            max_value=23,
            value=12,
            help="Hora del día (0-23) en que se realizó la transacción"
        )
        
        day_of_week = st.slider(
            "Día de la semana",
            min_value=0,
            max_value=6,
            value=2,
            format="%d",
            help="0=Lunes, 1=Martes, ..., 6=Domingo"
        )
    
    with col2:
        st.subheader("📍 Ubicaciones")
        
        # Ubicación del cliente
        lat = st.number_input(
            "Latitud del cliente",
            min_value=-90.0,
            max_value=90.0,
            value=40.7128,
            step=0.01,
            help="Latitud de la ubicación del cliente"
        )
        
        long = st.number_input(
            "Longitud del cliente",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
            step=0.01,
            help="Longitud de la ubicación del cliente"
        )
        
        city_pop = st.number_input(
            "Población de la ciudad",
            min_value=0,
            value=8000000,
            step=1000,
            help="Población de la ciudad donde se ubica el cliente"
        )
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🏪 Información del Comercio")
        
        # Ubicación del comercio
        merch_lat = st.number_input(
            "Latitud del comercio",
            min_value=-90.0,
            max_value=90.0,
            value=40.7190,
            step=0.01,
            help="Latitud de la ubicación del comercio"
        )
        
        merch_long = st.number_input(
            "Longitud del comercio",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
            step=0.01,
            help="Longitud de la ubicación del comercio"
        )
        
        # Categoría del comercio
        category = st.selectbox(
            "Categoría del comercio",
            options=["shopping_net", "shopping_pos", "gas_transport", "grocery_pos", 
                    "grocery_net", "entertainment", "misc_net", "misc_pos"],
            help="Tipo de comercio"
        )
    
    with col4:
        st.subheader("👤 Información del Cliente")
        
        # Información del cliente
        age = st.number_input(
            "Edad del cliente",
            min_value=18,
            max_value=100,
            value=45,
            step=1,
            help="Edad del titular de la tarjeta"
        )
        
        state = st.selectbox(
            "Estado",
            options=sorted(["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA",
                           "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD",
                           "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
                           "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
                           "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]),
            help="Estado de la transacción"
        )
        
        # Calcular distancia (Haversine)
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)
            a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
            return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance_km = haversine(lat, long, merch_lat, merch_long)
        st.metric("Distancia cliente-comercio", f"{distance_km:.2f} km")
    
    st.divider()
    
    # Botón de predicción
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        if st.button("🎯 Realizar Predicción", use_container_width=True):
            # Preparar inputs
            inputs = {
                "amt": amt,
                "hour": hour,
                "day_of_week": day_of_week,
                "lat": lat,
                "long": long,
                "city_pop": city_pop,
                "merch_lat": merch_lat,
                "merch_long": merch_long,
                "age": age,
                "distance_km": distance_km,
                "category": category,
                "state": state
            }
            
            try:
                # Preprocesar
                X_processed = preprocess_input(inputs, scaler, config, model)
                
                # Predicción
                prediction = model.predict(X_processed)[0]
                probability = model.predict_proba(X_processed)[0]
                
                st.divider()
                st.subheader("📈 Resultado de la Predicción")
                
                # Mostrar resultado
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if prediction == 1:
                        st.error("⚠️ **FRAUDE DETECTADO**")
                        st.metric("Confianza", f"{probability[1]*100:.2f}%")
                    else:
                        st.success("✅ **TRANSACCIÓN LEGÍTIMA**")
                        st.metric("Confianza", f"{probability[0]*100:.2f}%")
                
                with col_res2:
                    st.write("**Probabilidades:**")
                    st.write(f"- Legítimo: {probability[0]:.4f}")
                    st.write(f"- Fraude: {probability[1]:.4f}")
                
                # Visualizar distribución de probabilidad
                st.bar_chart(
                    pd.DataFrame({
                        "Clase": ["Legítimo", "Fraude"],
                        "Probabilidad": probability
                    }).set_index("Clase")
                )
                
                # Resumen de entrada
                st.divider()
                st.subheader("📋 Resumen de Entrada")
                summary_df = pd.DataFrame({
                    "Parámetro": list(inputs.keys()),
                    "Valor": list(inputs.values())
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error durante la predicción: {e}")
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **Fraud Detection Model** | Detección de Fraude en Tarjetas de Crédito
    
    Desarrollado con MLOps | [Documentación](README.md)
    """)


if __name__ == "__main__":
    main()
