import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv("dataset/data.csv")  # Asegúrate de reemplazar "tu_archivo.csv" con el nombre de tu archivo de datos

# Definir la función para calcular similitud de coseno
def calcular_similitud(nombre_del_producto, data):
    # Vectorizar los nombres de los productos
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['nombre'])

    # Combinar características
    features = np.column_stack([tfidf_matrix.toarray(), data['calificacion'], data['precio_descuento'], data['precio_actual']])

    # Calcular la matriz de similitud de coseno
    similarity_matrix = cosine_similarity(features)

    # Obtener el índice del producto dado
    product_index = data[data['nombre'] == nombre_del_producto].index[0]

    # Obtener las similitudes con otros productos
    product_similarities = similarity_matrix[product_index]

    # Obtener los índices de los productos más similares
    #most_similar_products_indices = np.argsort(-product_similarities)
    most_similar_products_indices = np.argsort(-product_similarities)[:10]

    # Obtener los nombres de los productos más similares
    most_similar_products = data.loc[most_similar_products_indices, 'nombre']

    return most_similar_products

# Configurar la aplicación con título y descripción
st.title('Consulta de Recomendación de Productos')
st.write('Esta aplicación permite realizar consultas de recomendación de productos utilizando un modelo de similitud de coseno basado en nombres, calificaciones y precios.')

st.write("Tabla de Productos Disponibles:")
st.write(data)
# Interfaz para ingresar el nombre del producto
st.title("Recomendaciones")
nombre_del_producto = st.selectbox('Seleccione un producto:', data['nombre'])

if nombre_del_producto:
    # Calcular la similitud de coseno y mostrar los resultados
    st.write("Los productos más similares al producto", nombre_del_producto, "son:")
    most_similar_products = calcular_similitud(nombre_del_producto, data)
    st.write(most_similar_products)


