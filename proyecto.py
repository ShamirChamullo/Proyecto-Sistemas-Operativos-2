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

    # Selección de rango de datos
    st.write(f"Seleccione el rango de valores para {x_var} y {y_var}:")
    x_min = st.number_input(f"Valor mínimo de {x_var}", value=float(data[x_var].min()))
    x_max = st.number_input(f"Valor máximo de {x_var}", value=float(data[x_var].max()))
    y_min = st.number_input(f"Valor mínimo de {y_var}", value=float(data[y_var].min()))
    y_max = st.number_input(f"Valor máximo de {y_var}", value=float(data[y_var].max()))

    # Filtrar datos según los rangos seleccionados
    filtered_data = data[(data[x_var] >= x_min) & (data[x_var] <= x_max) & (data[y_var] >= y_min) & (data[y_var] <= y_max)]

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

        # Diagrama de tortas para cada categoría
        for category in category_counts.index:
            st.write(f"Diagrama de tortas para la categoría: {category}")
            category_data = data[data['Categories'] == category]

            # Pie chart for Visits
            plt.figure(figsize=(10, 6))
            plt.pie(category_data['Visits'], labels=category_data['Username'], autopct='%1.1f%%', startangle=140)
            plt.title(f'Distribución de Visitas en la categoría {category}')
            plt.axis('equal')
            st.pyplot(plt)

            # Pie chart for Suscribers
            plt.figure(figsize=(10, 6))
            plt.pie(category_data['Suscribers'], labels=category_data['Username'], autopct='%1.1f%%', startangle=140)
            plt.title(f'Distribución de Suscriptores en la categoría {category}')
            plt.axis('equal')
            st.pyplot(plt)

            # Pie chart for Likes
            plt.figure(figsize=(10, 6))
            plt.pie(category_data['Likes'], labels=category_data['Username'], autopct='%1.1f%%', startangle=140)
            plt.title(f'Distribución de Likes en la categoría {category}')
            plt.axis('equal')
            st.pyplot(plt)

            # Pie chart for Comments
            plt.figure(figsize=(10, 6))
            plt.pie(category_data['Comments'], labels=category_data['Username'], autopct='%1.1f%%', startangle=140)
            plt.title(f'Distribución de Comentarios en la categoría {category}')
            plt.axis('equal')
            st.pyplot(plt)
