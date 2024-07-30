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
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x.flatten(), y=y, data=filtered_data, label='Datos reales')
        plt.plot(x, y_pred, color='red', label='Regresión lineal')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{title} (R² = {r2:.2f})')
        plt.legend()

        # Formato de ejes para números enteros
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}'))

        st.pyplot(plt)
        st.write(f'Valor de R²: {r2:.2f}')
    
    # Ejecutar regresión lineal basada en selección del usuario
    X = filtered_data[x_var].values.reshape(-1, 1)
    y = filtered_data[y_var].values
    plot_regression(X, y, x_var, y_var, f'Regresión Lineal entre {x_var} y {y_var}')
