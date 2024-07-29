import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Configurar el estilo de los gráficos
sns.set(style="whitegrid")

# Título de la aplicación
st.title('Análisis de Ventas de Autos 2010-2020')

# Subir el archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    # Cargar el archivo CSV
    df = pd.read_csv(uploaded_file)

    # Mostrar las primeras filas del dataframe
    st.write("Primeras filas del dataset:")
    st.write(df.head())

    # Histograma de precios
    st.write("Distribución de Precios de Autos")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price (USD)'], bins=20, kde=True)
    plt.title('Distribución de Precios de Autos')
    plt.xlabel('Precio (USD)')
    plt.ylabel('Frecuencia')
    st.pyplot(plt)

    # Diagrama de torta de la cantidad de autos vendidos por marca
    st.write("Distribución de Ventas por Marca")
    plt.figure(figsize=(10, 6))
    df['Make'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.title('Distribución de Ventas por Marca')
    plt.ylabel('')
    st.pyplot(plt)

    # Identificar la marca más vendida
    top_make = df['Make'].value_counts().idxmax()

    # Filtrar datos de la marca más vendida
    top_make_data = df[df['Make'] == top_make]

    # Regresión lineal simple entre el año y el precio de la marca más vendida
    st.write(f'Regresión Lineal: Año vs Precio de {top_make}')
    
    # Calcular la correlación y R-squared
    correlation, _ = stats.pearsonr(top_make_data['Year'], top_make_data['Price (USD)'])
    slope, intercept, r_value, _, _ = stats.linregress(top_make_data['Year'], top_make_data['Price (USD)'])
    r_squared = r_value**2

    plt.figure(figsize=(10, 6))
    sns.regplot(x='Year', y='Price (USD)', data=top_make_data)
    plt.title(f'Regresión Lineal: Año vs Precio de {top_make}')
    plt.xlabel('Año')
    plt.ylabel('Precio (USD)')
    
    # Añadir texto con la correlación y R-squared
    plt.text(0.05, 0.95, f'Correlación: {correlation:.2f}\nR-squared: {r_squared:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    st.pyplot(plt)

    # Mostrar algunas estadísticas descriptivas del dataset
    st.write("Estadísticas descriptivas del dataset:")
    st.write(df.describe())

else:
    st.write("Por favor, sube un archivo CSV para analizar los datos.")