from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import pandas as pd
import numpy as np

EXTENSION = st.radio(
    "Escoja la extensión del archivo",
    ('CSV', 'XLS', 'JSON'))
uploaded_file = st.file_uploader("Escoja un archivo")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    if EXTENSION == 'CSV':
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
    elif EXTENSION == 'XLS':
        dataframe = pd.read_excel(uploaded_file)
        dataframe.to_csv('Convert.csv')
        dataframe = pd.read_csv('Convert.csv')
        st.write(dataframe)
    elif EXTENSION == 'JSON':
        dataframe = pd.read_json(uploaded_file)
        dataframe.to_csv('Convert.csv')
        dataframe = pd.read_csv('Convert.csv')
        st.write(dataframe)


def regresion_lineal():
    columna = st.selectbox('Seleccione X', (dataframe.columns))
    columna2 = st.selectbox('Seleccione dato de columna', (dataframe.columns))
    X = np.asarray(dataframe[columna]).reshape(-1, 1)
    Y = dataframe[columna2]

    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    Y_pred = linear_regression.predict(X)

    st.write("Error medio: ", mean_squared_error(Y, Y_pred, squared=True))
    st.write("Coef: ", linear_regression.coef_)
    st.write("R2: ", r2_score(Y, Y_pred))

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()
    val_prediccion = st.number_input('Ingrese valor')
    Y_new = linear_regression.predict([[val_prediccion]])
    st.write(Y_new)


op = st.multiselect('Escoja una opción', ['Regresion Lineal', 'Regresion Polinomial', 'Clasificador Gausiano',
                    'Clasificador de arboles de desicion', 'Redes neuronales'])

if len(op) > 0:
    if op[0] == 'Regresion Lineal':
        regresion_lineal()
    elif op[0] == 'Regresion Polinomial':
        st.write('Selecciono regresion polinomial')
    elif op[0] == 'Clasificador Gausiano':
        st.write('Selecciono Clasificador Gausiano')
    elif op[0] == 'Clasificador de arboles de desicion':
        st.write('Selecciono Arboles de desición')
    elif op[0] == 'Redes neuronales':
        st.write('Selecciono redes neuronales')