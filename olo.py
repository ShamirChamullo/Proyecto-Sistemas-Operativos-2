import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Cargar el dataset
file_path = 'FOOD-DATA-GROUP1.csv'
data = pd.read_csv(file_path)

# Título de la aplicación
st.title('Análisis Estadístico de Datos de Alimentos')

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

# Histograma y regresión lineal para la columna 'Caloric Value'
st.header('Histograma de Caloric Value y Regresión Lineal')
fig2, ax2 = plt.subplots()
sns.histplot(data['Caloric Value'], kde=True, ax=ax2)
st.pyplot(fig2)

# Regresión lineal simple
st.header('Regresión Lineal Simple')
x = data['Caloric Value'].values.reshape(-1, 1)
y = data['Fat'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Graficar la regresión lineal
fig3, ax3 = plt.subplots()
ax3.scatter(data['Caloric Value'], data['Fat'], color='blue')
ax3.plot(data['Caloric Value'], y_pred, color='red')
ax3.set_xlabel('Caloric Value')
ax3.set_ylabel('Fat')
st.pyplot(fig3)

# Mostrar la precisión del modelo
st.write(f'Taza de precisión del modelo: {model.score(x, y):.2f}')

# Otros gráficos estadísticos
st.header('Otros Gráficos Estadísticos')

# Boxplot
fig4, ax4 = plt.subplots()
sns.boxplot(x=data['Caloric Value'], ax=ax4)
ax4.set_title('Boxplot de Caloric Value')
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
