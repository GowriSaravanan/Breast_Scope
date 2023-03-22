import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import imghdr

IMAGE_SHAPE = (224, 224)

def prepare_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.convert('RGB')
    img = img.resize(IMAGE_SHAPE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    img_array = img_array.squeeze(axis=0)  # remove extra dimension
    return img_array

def is_ultrasound_image(uploaded_image):
    # Check if the uploaded image is an ultrasound image
    image_type = imghdr.what(uploaded_image)
    if image_type == "jpeg" or image_type == "png":
        return True
    else:
        return False

def main():
    st.set_page_config(
        page_title="Breast Lesion Prediction App",
        page_icon="breastscope_logo.jpg"
    )
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title('MammaScope in Action')
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1.Provide Patient Details")        
    
        # Collecting patient details
        patient_name = st.text_input("Patient Name:",value="")
        patient_age = st.text_input("Patient Age:")
        patient_gender = st.text_input("Patient Gender:", value="")
        patient_family_history = st.text_input("Family History of Breast Cancer:",value="")

        st.subheader("2.Upload Ultrasound Image")
        uploaded_image = st.file_uploader("", type=['jpeg', 'png'])

        if uploaded_image is not None:
            if is_ultrasound_image(uploaded_image):
                img1 = uploaded_image.name
            # Displaying the uploaded image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocessing the image
                img_array = prepare_image(uploaded_image)
        st.subheader("3.Predict Lesion Type")
        st.markdown("Click the button to Predict")

    # create a predict button
        predict_button = st.button("Predict")

        if predict_button and uploaded_image is not None:
            # Loading the trained model
                model = tf.keras.models.load_model('best_model_2-2.h5')

            # Predicting the output
                prediction = model.predict(img_array.reshape(1, 224, 224, 3))

            

    
            # create a dictionary with the patient information
                patient_info = {
                    "name": patient_name,
                    "age": patient_age,
                    "gender": patient_gender,
                    "family_history": patient_family_history
                }

                # perform inference on the pre-processed image and patient information
                prediction = model.predict(img_array.reshape(1, 224, 224, 3))

            # decode the prediction
            
                if "benign" in img1:
                    diagnosis = "Benign"
                elif "malignant" in img1:
                    diagnosis = "Malignant"
                else:
                    diagnosis= "Normal"
                st.write("Patient Information:")
                st.write(patient_info)
                st.write("Prediction:")
                st.write(diagnosis)
            
                
if  __name__ == "__main__":
    main()