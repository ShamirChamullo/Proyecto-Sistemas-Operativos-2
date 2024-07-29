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
        sns.scatterplot(x=x.flatten(), y=y, data=data, label='Datos reales')
        plt.plot(x, y_pred, color='red', label='Regresión lineal')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{title} (R² = {r2:.2f})')
        plt.legend()
        st.pyplot(plt)
        st.write(f'Valor de R²: {r2:.2f}')
    
    # Ejecutar regresión lineal basada en selección del usuario
    X = data[x_var].values.reshape(-1, 1)
    y = data[y_var].values
    plot_regression(X, y, x_var, y_var, f'Regresión Lineal entre {x_var} y {y_var}')

    # Histogramas
    st.write("Histogramas de las variables numéricas")
    for column in ['Suscribers', 'Visits', 'Likes', 'Comments']:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Histograma de {column}')
        st.pyplot(plt)

    # Diagrama de torta (pastel) - Distribución de categorías
    if 'Categories' in data.columns:
        st.write("Diagrama de torta de las categorías")
        category_counts = data['Categories'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribución de Categorías')
        plt.axis('equal')
        st.pyplot(plt)

    # Diagrama de torta (pastel) - Distribución de países
    if 'Country' in data.columns:
        st.write("Diagrama de torta de los países")
        country_counts = data['Country'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribución de Países')
        plt.axis('equal')
        st.pyplot(plt)
