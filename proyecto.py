import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.ticker as mticker
from io import BytesIO

# Título
st.title('Análisis de Youtubers')

# Cargar archivo
uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file)

    # Mostrar las primeras filas del DataFrame
    st.write("LOs primeros 5 youtubers mas grandes")
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

    # Función para guardar gráficos como archivos en un buffer
    def save_plot_to_buffer(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    # Gráfico de torta de la distribución de YouTubers por país
    if 'Country' in filtered_data.columns:
        st.write("Distribución de YouTubers por país")
        country_counts = filtered_data['Country'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribución de YouTubers por País')
        plt.axis('equal')
        buf = save_plot_to_buffer(plt.gcf())
        st.pyplot(plt)
        st.download_button(label="Descargar gráfico de distribución por país", data=buf, file_name="distribucion_pais.png", mime="image/png")

    # Gráfico de torta de la distribución de categorías
    if 'Categories' in filtered_data.columns:
        st.write("Distribución de YouTubers por categoría")
        category_counts = filtered_data['Categories'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribución de YouTubers por Categoría')
        plt.axis('equal')
        buf = save_plot_to_buffer(plt.gcf())
        st.pyplot(plt)
        st.download_button(label="Descargar gráfico de distribución por categoría", data=buf, file_name="distribucion_categoria.png", mime="image/png")

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

        # Ajustar formato de los ejes
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
        plt.legend()
        buf = save_plot_to_buffer(plt.gcf())
        st.pyplot(plt)
        st.download_button(label="Descargar gráfico de regresión", data=buf, file_name="regresion_lineal.png", mime="image/png")

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
        sns.histplot(filtered_data[column], kde=True, bins=30)
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de {column}')

        # Ajustar formato de los ejes
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
        buf = save_plot_to_buffer(plt.gcf())
        st.pyplot(plt)
        st.download_button(label=f"Descargar gráfico de distribución de {column}", data=buf, file_name=f"distribucion_{column}.png", mime="image/png")

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

        # Ajustar formato de los ejes
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
        buf = save_plot_to_buffer(plt.gcf())
        st.pyplot(plt)
        st.download_button(label="Descargar gráfico de tendencias temporales", data=buf, file_name="tendencias_temporales.png", mime="image/png")

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

        # Ajustar formato de los ejes
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
        buf = save_plot_to_buffer(plt.gcf())
        st.pyplot(plt)
        st.download_button(label="Descargar gráfico de Pareto de Categorías", data=buf, file_name="pareto_categorias.png", mime="image/png")

    # Descargar archivo Excel con los datos
    st.write("Descargar datos como archivo Excel")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        filtered_data.to_excel(writer, sheet_name='Datos', index=False)
    excel_buffer.seek(0)
    st.download_button(label="Descargar datos como Excel", data=excel_buffer, file_name="datos_youtubers.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
