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

    # Gráfico de torta de la distribución de YouTubers por país
    if 'Country' in filtered_data.columns:
        st.write("Distribución de YouTubers por país")
        country_counts = filtered_data['Country'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribución de YouTubers por País')
        plt.axis('equal')
        st.pyplot(plt)

    # Gráfico de torta de la distribución de categorías
    if 'Categories' in filtered_data.columns:
        st.write("Distribución de YouTubers por categoría")
        category_counts = filtered_data['Categories'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribución de YouTubers por Categoría')
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
        
        # Gráfico de regresión
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x.flatten(), y=y, data=filtered_data, label='Datos reales')
        plt.plot(x, y_pred, color='red', label='Regresión lineal')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{title} (R² = {r2:.2f})')
        plt.legend()
        st.pyplot(plt)

        # Mostrar valor de R²
        st.write(f'**Valor de R²**: {r2:.2f}')
    
    # Ejecutar regresión lineal basada en selección del usuario
    X = filtered_data[x_var].values.reshape(-1, 1)
    y = filtered_data[y_var].values
    plot_regression(X, y, x_var, y_var, f'Regresión Lineal entre {x_var} y {y_var}')

    # Distribución de Datos
    st.write("Distribución de Datos")
    for column in ['Suscribers', 'Visits', 'Likes', 'Comments']:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(filtered_data[column], shade=True)
        plt.title(f'Distribución de {column}')
        st.pyplot(plt)

    # Tendencias Temporales (si hay una columna de fecha)
    if 'Date' in filtered_data.columns:
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
        st.write("Tendencias Temporales")
        plt.figure(figsize=(14, 7))
        for column in ['Suscribers', 'Visits', 'Likes', 'Comments']:
            plt.plot(filtered_data['Date'], filtered_data[column], label=column)
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.title('Tendencias Temporales')
        plt.legend()
        st.pyplot(plt)

    # Gráfico de Pareto de Categorías
    if 'Categories' in filtered_data.columns:
        st.write("Gráfico de Pareto de Categorías")
        category_counts = filtered_data['Categories'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.xlabel('Categorías')
        plt.ylabel('Número de YouTubers')
        plt.title('Distribución de YouTubers por Categoría')
        plt.xticks(rotation=90)
        st.pyplot(plt)
