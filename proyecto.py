import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
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

    # Limpiar datos (remover filas con valores nulos en Suscribers, Visits, Likes o Comments)
    data = data.dropna(subset=['Suscribers', 'Visits', 'Likes', 'Comments'])

    # Control deslizante para limitar el número de filas a mostrar
    st.write("Selecciona el número de filas a mostrar:")
    num_rows = st.slider('Número de filas', min_value=1, max_value=len(data), value=len(data))

    # Aplicar el limitador de filas a los datos
    filtered_data = data.head(num_rows)

    # Mostrar los datos filtrados
    st.write("Datos filtrados:")
    st.write(filtered_data)

    # Gráfico de torta de la distribución de YouTubers por país
    if 'Country' in filtered_data.columns:
        st.write("Distribución de YouTubers por país")
        country_counts = filtered_data['Country'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribución de YouTubers por País')
        plt.axis('equal')
        st.pyplot(plt)

    # Selección de variables para regresión lineal
    st.write("Seleccione las variables para la regresión lineal simple:")
    x_var = st.selectbox('Variable Independiente (X)', ['Suscribers', 'Likes', 'Comments'])
    y_var = st.selectbox('Variable Dependiente (Y)', ['Visits'])

    # Función para crear gráficos de regresión
    def plot_regression(x, y, x_label, y_label, title):
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        coef = model.coef_[0]
        intercept = model.intercept_

        # Gráfico de regresión
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=x.flatten(), y=y, data=filtered_data, label='Datos reales')
        plt.plot(x, y_pred, color='red', label='Regresión lineal')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{title} (R² = {r2:.2f})')
        plt.legend()

        # Gráfico de residuos
        plt.subplot(1, 2, 2)
        residuals = y - y_pred
        sns.scatterplot(x=y_pred, y=residuals, data=filtered_data)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuos')
        plt.title('Gráfico de Residuos')

        plt.tight_layout()
        st.pyplot(plt)

        # Mostrar métricas
        st.write(f'**Coeficiente de Regresión (pendiente)**: {coef:.2f}')
        st.write(f'**Intersección (intercepto)**: {intercept:.2f}')
        st.write(f'**Error Cuadrático Medio (MSE)**: {mse:.2f}')
        st.write(f'**Valor de R²**: {r2:.2f}')
    
    # Ejecutar regresión lineal basada en selección del usuario
    X = filtered_data[x_var].values.reshape(-1, 1)
    y = filtered_data[y_var].values
    plot_regression(X, y, x_var, y_var, f'Regresión Lineal entre {x_var} y {y_var}')
