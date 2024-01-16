import streamlit as st
import pandas as pd
import regresilogit as rl
import joblib 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay , classification_report , accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier

def main():
    st.write("## Computer vision basic")
    model = st.file_uploader("Masukkan model logistik")
    dataset = st.file_uploader("Masukkan dataset yang akan digunakan")

    try:
        df = pd.read_csv(dataset)
        df = rl.PreprocessingData(df)

        X_train, X_test, y_train, y_test = rl.SplitTrain(df)

        loaded_model = joblib.load('RegressionlogHeartDisease.joblib')
        y_test_pred = loaded_model.predict(X_test)

        acc_test = accuracy_score(y_test, y_test_pred)
        st.write(f"Ke akuratan: {acc_test}")

        dataset2 = st.file_uploader("Masukkan dataset yang ingin diklasifikasi")
        new_data = pd.read_excel(dataset2)
        st.write(new_data)
        result = loaded_model.predict(new_data)
        new_data['class'] = result
        st.write(new_data)
    except:
        st.write("Silahkan input data dengan benar")
    # st.write("Prediction: ")
    # gender = st.checkbox("male", value=1)
    # age = st.slider("Age", min_value=0.1, max_value=100.0, step=0.5)
    # isSmoke = st.checkbox("Saya merokok ")
    # if isSmoke is True:
    #     smoke_perday = st.slider("Ngudud berapa batang", min_value=1, max_value=40, step=1)
    # isMedication = st.checkbox("Mengonsumsi obat tekanan darah ")
    # isStroke = st.checkbox("Saya pernah stroke ")
    # isDiabetes = st.checkbox("Saya mengalami diabetes ")

    # Cholesterol = st.number_input("Berapa kadar kolesterol dalam darah? ", value=None, placeholder="input angka disini")
    # bloodPressureSistol = st.number_input("Nilai tekanan darah saat ini saat berkontraksi", value=None, placeholder="input angka disini")
    # bloodPressureDiastol = st.number_input("Nilai tekanan darah saat ini saat relaksasi", value=None, placeholder="input angka disini")
    # bmi = st.number_input("body mass index", value=None, placeholder="input angka disini")
    # hr = st.number_input("Detak jantung", value=None, placeholder="input angka disini")
    # glucose = st.number_input("Kadar gula dalam darah", value=None, placeholder="input angka disini")
    
    # st.write(gender)

main()