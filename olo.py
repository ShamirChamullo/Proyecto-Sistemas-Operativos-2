import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

st.title('Regresión Lineal Simple')

# Subir archivo CSV
uploaded_file = st.file_uploader('Sube tu archivo CSV', type='csv')

if uploaded_file is not None:
    # Cargar el archivo CSV
    data = pd.read_csv(uploaded_file)

    # Convertir la columna 'food' a variables dummy
    encoder = OneHotEncoder()
    food_encoded = encoder.fit_transform(data[['food']]).toarray()
    food_encoded_df = pd.DataFrame(food_encoded, columns=encoder.get_feature_names_out(['food']))

    # Añadir las columnas codificadas al dataframe original
    data_encoded = pd.concat([data, food_encoded_df], axis=1)

    # Extraer las columnas pertinentes para Caloric Value
    X_caloric = food_encoded_df
    y_caloric = data['Caloric Value']

    # Extraer las columnas pertinentes para Saturated Fats
    X_saturated = food_encoded_df
    y_saturated = data['Saturated Fats']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train_caloric, X_test_caloric, y_train_caloric, y_test_caloric = train_test_split(X_caloric, y_caloric, test_size=0.2, random_state=42)
    X_train_saturated, X_test_saturated, y_train_saturated, y_test_saturated = train_test_split(X_saturated, y_saturated, test_size=0.2, random_state=42)

    # Crear los modelos de regresión lineal
    model_caloric = LinearRegression()
    model_saturated = LinearRegression()

    # Ajustar los modelos a los datos de entrenamiento
    model_caloric.fit(X_train_caloric, y_train_caloric)
    model_saturated.fit(X_train_saturated, y_train_saturated)

    # Realizar predicciones sobre el conjunto de prueba
    y_pred_caloric = model_caloric.predict(X_test_caloric)
    y_pred_saturated = model_saturated.predict(X_test_saturated)

    # Calcular el error cuadrático medio y el coeficiente de determinación R^2
    mse_caloric = mean_squared_error(y_test_caloric, y_pred_caloric)
    r2_caloric = r2_score(y_test_caloric, y_pred_caloric)
    mse_saturated = mean_squared_error(y_test_saturated, y_pred_saturated)
    r2_saturated = r2_score(y_test_saturated, y_pred_saturated)

    # Mostrar los resultados
    st.write(f'Caloric Value - MSE: {mse_caloric}, R^2: {r2_caloric}')
    st.write(f'Saturated Fats - MSE: {mse_saturated}, R^2: {r2_saturated}')

    # Visualizar los resultados para Caloric Value
    fig_caloric, ax_caloric = plt.subplots()
    ax_caloric.scatter(y_test_caloric, y_pred_caloric, color='blue', label='Datos reales vs Predicciones')
    ax_caloric.set_xlabel('Valores reales de Caloric Value')
    ax_caloric.set_ylabel('Valores predichos de Caloric Value')
    ax_caloric.set_title('Regresión Lineal Simple - Caloric Value')
    ax_caloric.legend()
    st.pyplot(fig_caloric)

    # Visualizar los resultados para Saturated Fats
    fig_saturated, ax_saturated = plt.subplots()
    ax_saturated.scatter(y_test_saturated, y_pred_saturated, color='blue', label='Datos reales vs Predicciones')
    ax_saturated.set_xlabel('Valores reales de Saturated Fats')
    ax_saturated.set_ylabel('Valores predichos de Saturated Fats')
    ax_saturated.set_title('Regresión Lineal Simple - Saturated Fats')
    ax_saturated.legend()
    st.pyplot(fig_saturated)
