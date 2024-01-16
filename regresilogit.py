import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

def PreprocessingData(data_raw):
    data_raw.drop(columns = 'education', inplace = True)
    data_raw.isna().sum().sort_values(ascending = False)
    data_raw['cigsPerDay'].fillna(value=0.0, inplace = True)

    return data_raw

def SplitTrain(data_cooked):
    slide_testsize = st.slider("Masukkan jumlah test size yang digunakan: ", min_value=0.001, max_value=0.2, value=0.1, step=0.01)
    X =data_cooked.drop(columns=['TenYearCHD'])
    target = data_cooked['TenYearCHD']

    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=slide_testsize, random_state=42)

    return X_train, X_test, y_train, y_test