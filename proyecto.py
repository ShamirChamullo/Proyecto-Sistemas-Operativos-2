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

    # Controles deslizantes para filtrar los datos
    st.write("Selecciona el rango de valores para filtrar los datos:")
    min_suscribers, max_suscribers = st.slider('Suscribers', min_value=int(data['Suscribers'].min()), max_value=int(data['Suscribers'].max()), value=(int(data['Suscribers'].min()), int(data['Suscribers'].max())))
    min_visits, max_visits = st.slider('Visits', min_value=int(data['Visits'].min()), max_value=int(data['Visits'].max()), value=(int(data['Visits'].min()), int(data['Visits'].max())))
    min_likes, max_likes = st.slider('Likes', min_value=int(data['Likes'].min()), max_value=int(data['Likes'].max()), value=(int(data['Likes'].min()), int(data['Likes'].max())))
    min_comments, max_comments = st.slider('Comments', min_value=int(data['Comments'].min()), max_value=int(data['Comments'].max()), value=(int(data['Comments'].min()), int(data['Comments'].max())))

    # Aplicar el filtro a los datos
    filtered_data = data[
        (data['Suscribers'] >= min_suscribers) & (data['Suscribers'] <= max_suscribers) &
        (data['Visits'] >= min_visits) & (data['Visits'] <= max_visits) &
        (data['Likes'] >= min_likes) & (data['Likes'] <= max_likes) &
        (data['Comments'] >= min_comments) & (data['Comments'] <= max_comments)
    ]

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
        st.pyplot(plt)
        st.write(f'Valor de R²: {r2:.2f}')
    
    # Ejecutar regresión lineal basada en selección del usuario
    X = filtered_data[x_var].values.reshape(-1, 1)
    y = filtered_data[y_var].values
    plot_regression(X, y, x_var, y_var, f'Regresión Lineal entre {x_var} y {y_var}')

    # Histogramas generales
    st.write("Histogramas generales de las variables numéricas")
    for column in ['Suscribers', 'Visits', 'Likes', 'Comments']:
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_data[column], kde=True)
        plt.title(f'Histograma de {column}')
        st.pyplot(plt)

    # Diagrama de torta general - Distribución de categorías
    if 'Categories' in filtered_data.columns:
        st.write("Diagrama de torta de la distribución de categorías")
        category_counts = filtered_data['Categories'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribución de Categorías')
        plt.axis('equal')
        st.pyplot(plt)

        # Mostrar la distribución general de visitas, suscriptores, likes y comentarios
        st.write("Distribución general de visitas, suscriptores, likes y comentarios por categoría")
        for metric in ['Visits', 'Suscribers', 'Likes', 'Comments']:
            plt.figure(figsize=(10, 6))
            metric_sums = filtered_data.groupby('Categories')[metric].sum()
            plt.pie(metric_sums, labels=metric_sums.index, autopct='%1.1f%%', startangle=140)
            plt.title(f'Distribución de {metric} por categoría')
            plt.axis('equal')
            st.pyplot(plt)
