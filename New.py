import numpy as np
from PIL import Image
from keras.models import load_model
import streamlit as st

def main():
    model = load_model('mnist.h5')
    st.write("## Computer vision basic")

    photo_test = st.file_uploader('Model saat ini mampu mendeteksi angka dalam foto', type=['jpg', 'png'])
    if photo_test is not None:
        image_test = Image.open(photo_test)
        image_test = np.array(image_test).reshape(-1, 28, 28, 1)

        result = model.predict({'conv2d_8_input' : image_test})
        result = np.argmax(result)

        st.write("Hasil pembacaan dari komputer: ")
        st.header(f'{result}')

main()