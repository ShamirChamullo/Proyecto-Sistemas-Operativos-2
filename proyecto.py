import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Título
st.title('Análisis de Youtubers')

# Cargar archivo
uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file)

    # Mostrar las primeras filas del DataFrame
    st.write("Primeras filas del archivo:")
    st.write(data.head())

    # Limpiar datos (remover filas con valores nulos en Suscribers o Visits)
    data = data.dropna(subset=['Suscribers', 'Visits'])

    # Variables
    X = data['Suscribers'].values.reshape(-1, 1)
    y = data['Visits'].values

    # Crear y ajustar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Predicciones
    y_pred = model.predict(X)

    # Calcular R^2
    r2 = r2_score(y, y_pred)

    # Crear gráfico
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Suscribers', y='Visits', data=data, label='Datos reales')
    plt.plot(data['Suscribers'], y_pred, color='red', label='Regresión lineal')
    plt.xlabel('Suscribers')
    plt.ylabel('Visits')
    plt.title(f'Regresión Lineal entre Suscriptores y Visitas (R² = {r2:.2f})')
    plt.legend()

    # Mostrar gráfico
    st.pyplot(plt)
    st.write(f'Valor de R²: {r2:.2f}')
