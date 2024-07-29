import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Regresión Lineal Simple", page_icon="📈")

# Título de la aplicación
st.title("Regresión Lineal Simple: Caloric Value vs Saturated Fats")

# Cargar el archivo CSV
uploaded_file = st.file_uploader("Carga tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.write(data.head())

    # Seleccionar las columnas "Caloric Value" y "Saturated Fats"
    if 'Caloric Value' in data.columns and 'Saturated Fats' in data.columns:
        X = data[['Caloric Value']].values
        y = data['Saturated Fats'].values

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear el modelo de regresión lineal
        model = LinearRegression()

        # Entrenar el modelo
        model.fit(X_train, y_train)

        # Realizar predicciones
        y_pred = model.predict(X_test)

        # Evaluar el modelo
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f'Mean Squared Error: {mse}')
        st.write(f'R-squared: {r2}')

        # Visualizar los resultados
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
        plt.xlabel('Caloric Value')
        plt.ylabel('Saturated Fats')
        plt.title('Regresión Lineal Simple: Caloric Value vs Saturated Fats')
        plt.legend()
        st.pyplot(plt.gcf())
    else:
        st.error("El archivo CSV debe contener las columnas 'Caloric Value' y 'Saturated Fats'.")
