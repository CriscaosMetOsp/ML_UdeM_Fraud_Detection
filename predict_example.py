"""
Script de ejemplo para hacer predicciones sin interfaz gráfica.
Útil para integración con otros sistemas o scripts.

Uso:
    python predict_example.py
"""

import joblib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Agregar src al path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.data.preprocessing import _haversine


def load_artifacts():
    """Cargar modelo, scaler y configuración."""
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    return model, scaler, config


def predict_transaction(
    amt: float,
    hour: int,
    day_of_week: int,
    lat: float,
    long: float,
    city_pop: int,
    merch_lat: float,
    merch_long: float,
    age: int,
    category: str,
    state: str,
    model,
    scaler,
    config
) -> dict:
    """
    Hacer predicción para una transacción.
    
    Returns
    -------
    dict con predicción y probabilidades
    """
    # Calcular distancia
    distance_km = _haversine(lat, long, merch_lat, merch_long)
    
    # Crear DataFrame
    data = {
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
        "state": state,
    }
    
    X = pd.DataFrame([data])
    
    # Asegurar orden de columnas
    feature_cols = (
        config["features"]["numerical_cols"] + 
        config["features"]["categorical_cols"]
    )
    feature_cols = [c for c in feature_cols if c != config["features"]["target"]]
    X = X[feature_cols]
    
    # Escalar features numéricos
    numerical_cols = [
        c for c in config["features"]["numerical_cols"] 
        if c in X.columns
    ]
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    # Predicción
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return {
        "prediction": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "fraud_probability": float(probability[1]),
        "legitimate_probability": float(probability[0]),
        "confidence": float(max(probability) * 100),
        "input_data": data,
    }


def main():
    """Función principal con ejemplos."""
    
    print("=" * 60)
    print("FRAUD DETECTION - Prediction Examples")
    print("=" * 60)
    
    model, scaler, config = load_artifacts()
    
    # Ejemplo 1: Transacción legítima típica
    print("\n📌 EJEMPLO 1: Transacción Legítima (Normal)")
    print("-" * 60)
    
    result1 = predict_transaction(
        amt=50.0,
        hour=14,
        day_of_week=2,  # Miércoles
        lat=40.7128,
        long=-74.0060,
        city_pop=8000000,
        merch_lat=40.7190,
        merch_long=-74.0060,
        age=35,
        category="shopping_pos",
        state="NY",
        model=model,
        scaler=scaler,
        config=config,
    )
    
    print(f"Predicción: {result1['prediction']}")
    print(f"Probabilidad de Fraude: {result1['fraud_probability']:.4f}")
    print(f"Confianza: {result1['confidence']:.2f}%")
    print(f"Monto: ${result1['input_data']['amt']:.2f}")
    print(f"Hora: {result1['input_data']['hour']}:00")
    print(f"Distancia cliente-comercio: {result1['input_data']['distance_km']:.2f} km")
    
    # Ejemplo 2: Transacción sospechosa
    print("\n📌 EJEMPLO 2: Transacción Sospechosa (Potencial Fraude)")
    print("-" * 60)
    
    result2 = predict_transaction(
        amt=2000.0,
        hour=3,  # 3 AM
        day_of_week=0,  # Lunes
        lat=40.7128,
        long=-74.0060,
        city_pop=100000,  # Ciudad pequeña
        merch_lat=34.0522,  # Los Ángeles
        merch_long=-118.2437,
        age=28,
        category="shopping_net",
        state="CA",
        model=model,
        scaler=scaler,
        config=config,
    )
    
    print(f"Predicción: {result2['prediction']}")
    print(f"Probabilidad de Fraude: {result2['fraud_probability']:.4f}")
    print(f"Confianza: {result2['confidence']:.2f}%")
    print(f"Monto: ${result2['input_data']['amt']:.2f}")
    print(f"Hora: {result2['input_data']['hour']}:00 (madrugada)")
    print(f"Distancia cliente-comercio: {result2['input_data']['distance_km']:.2f} km")
    
    # Ejemplo 3: Transacción intermediaria
    print("\n📌 EJEMPLO 3: Transacción Intermediaria")
    print("-" * 60)
    
    result3 = predict_transaction(
        amt=150.0,
        hour=19,
        day_of_week=5,  # Sábado
        lat=41.8781,
        long=-87.6298,
        city_pop=2700000,
        merch_lat=41.8800,
        merch_long=-87.6300,
        age=45,
        category="grocery_pos",
        state="IL",
        model=model,
        scaler=scaler,
        config=config,
    )
    
    print(f"Predicción: {result3['prediction']}")
    print(f"Probabilidad de Fraude: {result3['fraud_probability']:.4f}")
    print(f"Confianza: {result3['confidence']:.2f}%")
    print(f"Monto: ${result3['input_data']['amt']:.2f}")
    print(f"Hora: {result3['input_data']['hour']}:00")
    print(f"Distancia cliente-comercio: {result3['input_data']['distance_km']:.2f} km")
    
    print("\n" + "=" * 60)
    print("✅ Ejemplos completados")
    print("=" * 60)


if __name__ == "__main__":
    main()
