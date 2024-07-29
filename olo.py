import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Título de la aplicación
st.title('Análisis Estadístico de Datos de Alimentos')

# Cargar el dataset
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Mostrar los primeros datos
    st.header('Primeros datos del dataset')
    st.write(data.head())

    # Diagrama de torta para la columna 'food'
    st.header('Distribución de Tipos de Comida')
    food_counts = data['food'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(food_counts, labels=food_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Para que sea un círculo
    st.pyplot(fig1)

    # Histograma de una columna seleccionada
    st.header('Histograma de una Columna Seleccionada')
    column = st.selectbox('Selecciona una columna para el histograma', data.columns)
    fig2, ax2 = plt.subplots()
    sns.histplot(data[column], kde=True, ax=ax2)
    st.pyplot(fig2)

    # Selección de columnas para el modelo de Random Forest
    st.header('Modelo de Bosque Aleatorio')
    target_column = st.selectbox('Selecciona la columna objetivo (Y)', data.columns)
    feature_columns = st.multiselect('Selecciona las columnas características (X)', data.columns)

    if target_column and feature_columns:
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Entrenamiento del modelo
        model = RandomForestRegressor()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Graficar predicciones vs valores reales
        fig3, ax3 = plt.subplots()
        ax3.scatter(y, y_pred, color='blue')
        ax3.set_xlabel('Valores Reales')
        ax3.set_ylabel('Predicciones')
        ax3.set_title(f'Predicciones vs Valores Reales ({target_column})')
        st.pyplot(fig3)
        
        # Mostrar la precisión del modelo
        st.write(f'R^2 Score del modelo: {model.score(X, y):.2f}')

    # Otros gráficos estadísticos
    st.header('Otros Gráficos Estadísticos')

    # Boxplot
    fig4, ax4 = plt.subplots()
    sns.boxplot(x=data[column], ax=ax4)
    ax4.set_title(f'Boxplot de {column}')
    st.pyplot(fig4)

    # Pairplot
    st.header('Pairplot de Variables Numéricas')
    sns.pairplot(data.select_dtypes(include=[np.number]).dropna())
    st.pyplot()

    # Correlation matrix heatmap
    st.header('Heatmap de la Matriz de Correlación')
    fig5, ax5 = plt.subplots()
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax5)
    st.pyplot(fig5)
else:
    st.write("Por favor, sube un archivo CSV para continuar.")
