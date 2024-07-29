import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

    # Selector de marca
    marcas = df['Make'].unique()
    selected_make = st.selectbox("Selecciona la marca del auto", marcas)

    # Filtrar datos de la marca seleccionada
    selected_make_data = df[df['Make'] == selected_make]

    # Regresión lineal simple entre el año y el precio de la marca seleccionada
    st.write(f'Regresión Lineal: Año vs Precio de {selected_make}')
    
    X = selected_make_data['Year'].values.reshape(-1, 1)
    y = selected_make_data['Price (USD)'].values
    
    # Crear y ajustar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)
    
    # Predicciones
    y_pred = model.predict(X)
    
    # Calcular el coeficiente de determinación R^2
    r2 = r2_score(y, y_pred)
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Year', y='Price (USD)', data=selected_make_data, line_kws={'color': 'red'})
    plt.title(f'Regresión Lineal: Año vs Precio de {selected_make} (R^2 = {r2:.2f})')
    plt.xlabel('Año')
    plt.ylabel('Precio (USD)')
    st.pyplot(plt)

    # Mostrar algunas estadísticas descriptivas del dataset
    st.write("Estadísticas descriptivas del dataset:")
    st.write(df.describe())
else:
    st.write("Por favor, sube un archivo CSV para analizar los datos.")