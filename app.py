from pyexpat import features
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
from streamlit_option_menu import option_menu
from distutils import extension
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Machine Learning 201901073"
)

hiden_menu_style = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden; }
    </style>
"""
st.markdown(hiden_menu_style, unsafe_allow_html=True)

st.markdown(
    """
<style>
span[data-baseweb="tag"] {
  background-color: blue !important;
}
</style>
""",
    unsafe_allow_html=True,
)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

st.write("# Machine Learning 201901073")

flag = False
extension_archivo = st.radio(
    "Seleccione el tipo de archivo",
    ('CSV', 'XLS', 'XLSX', 'JSON'))
uploaded_file = st.file_uploader("Cargar archivo")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    if extension_archivo == 'CSV':
        dataframe = pd.read_csv(uploaded_file, header=None)
        st.dataframe(dataframe)
        flag = True
    elif extension_archivo == 'XLS':
        dataframe = pd.read_excel(uploaded_file)
        st.dataframe(dataframe)
        flag = True
    elif extension_archivo == 'XLSX':
        dataframe = pd.read_excel(uploaded_file)
        st.dataframe(dataframe)
        flag = True
    elif extension_archivo == 'JSON':
        dataframe = pd.read_json(uploaded_file)
        st.dataframe(dataframe)
        flag = True

def regresion_lineal():
    columna = st.selectbox('Seleccione X', (dataframe.columns), index=0)
    columna2 = st.selectbox('Seleccione Y', (dataframe.columns), index=1)
    X = np.asarray(dataframe[columna]).reshape(-1, 1)
    Y = dataframe[columna2]

    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    Y_pred = linear_regression.predict(X)
    
    st.write(Y_pred)
    st.write("Error medio: ", mean_squared_error(Y, Y_pred, squared=True))
    st.write("Coeficiente: " + str(linear_regression.coef_))
    st.write("R2: ", r2_score(Y, Y_pred))

    st.set_option('deprecation.showPyplotGlobalUse', False)

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()
    st.pyplot()

    val_prediccion = st.number_input('Ingrese valor', value=0)
    Y_new = linear_regression.predict([[val_prediccion]])
    st.write("Predicción: " + str(Y_new))


def regresion_polinomial():
    dtf = dataframe.columns.to_list()
    index1 = st.selectbox("Seleccione X", range(len(dtf)), format_func=lambda x: dtf[x], index=0)
    index2 = st.selectbox("Seleccione Y", range(len(dtf)), format_func=lambda x: dtf[x], index=1)

    X = dataframe.iloc[:, int(index1)].values.reshape(-1, 1)
    Y = dataframe.iloc[:, int(index2)].values.reshape(-1, 1)

    grado = st.number_input('Ingrese grado', value=0)
    poly = PolynomialFeatures(degree=int(grado))
    X = poly.fit_transform(X)
    Y = poly.fit_transform(Y)

    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    Y_pred = linear_regression.predict(X)
    st.write(Y_pred)
    st.write("Error medio: ", str(mean_squared_error(Y, Y_pred, squared=False)))
    st.write("R2: ", r2_score(Y, Y_pred))

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color = 'red')
    plt.show()
    st.pyplot()

    val_prediccion = st.number_input('Ingrese valor a predecir', value=0)
    Y_new = linear_regression.predict(poly.fit_transform([[val_prediccion]]))
    st.write("Predicción: ", Y_new)

def clasificador_gauseano():
    st.write("# Clasificador gauseano")
    dataframe.columns = range(dataframe.shape[1])
    st.write(dataframe)
    X = np.array(dataframe)

    index = X.shape[1]
    X = np.delete(X, int(index) - 1, axis=1)
    play_column = np.array(dataframe)
    play_column = play_column.T[int(index) - 1]

    lista_play = []
    for i in play_column:
        lista_play.append(i)
    lt = X.T.tolist()
    features = list(zip(*lt))
    model = GaussianNB()
    model.fit(features, lista_play)
    val = st.text_input('Ingrese entrada: ')
    if val == "":
        val = ""
    else:
        cadena = val.split(',')
        entrada = [int(x) for x in cadena]
        predicted = model.predict([entrada])
        st.write("Predicción: ", predicted)


selected = option_menu(
    menu_title="Menú",
    options=["Escoja una opción", "Regresion Lineal", "Regresion Polinomial", "Clasificador Gausiano", 
    "Clasificador de arboles de desicion", "Redes neuronales"],
    icons=['list', 'bar-chart', 'graph-down', 'graph-up-arrow', 'graph-up', 'graph-down-arrow'], 
    menu_icon="clipboard-data", default_index=0,
    orientation="horizontal",
)

if flag:
    if selected == "Regresion Lineal":
        st.title(f'Regresión Lineal')
        regresion_lineal()
    if selected == "Regresion Polinomial":
        st.title(f'Regresión Polinomial')
        regresion_polinomial()
    if selected == "Clasificador Gausiano":
        clasificador_gauseano()
        st.write("Clasificador Gausiano")
    if selected == "Clasificador de arboles de desicion":
        st.title(f'Clasificador de arboles de desicion')
        st.write("Clasificador de arboles de desicion")
    if selected == "Redes neuronales":
        st.title(f'Redes neuronales')
        st.write("Redes neuronales")
else:
    st.write("No se ha cargado ningún archivo (CSV, XLS, XLSX, JSON)")