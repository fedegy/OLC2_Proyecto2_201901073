import streamlit as st
import pandas as pd

EXTENSION = st.radio(
    "Escoja la extensión del archivo",
    ('CSV', 'XLS', 'JSON'))
uploaded_file = st.file_uploader("Escoja un archivo")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    if EXTENSION == 'CSV':
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        