#%%writefile app.py
import streamlit as st
import pandas as pd
import pickle
import os
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('/content/drive/MyDrive/Air quality/bg.jpg')
# Set the title of the web app
st.title('Air Quality Index')

# Path to the models and matrices
model_directory = '/content/drive/MyDrive/Air quality'

# Load the saved models
classification_models = [f for f in os.listdir(model_directory) if f.endswith('_model.pkl') and 'Regressor' not in f and 'Lasso' not in f and 'MLP' not in f]
regression_models = [f for f in os.listdir(model_directory) if f.endswith('_model.pkl') and ('Regressor' in f or 'Lasso' in f or 'MLP' in f)]

# Create a select box for the classification models
classification_model_name = st.selectbox(
    'Select a classification model:',
    classification_models
)

# Create a select box for the regression models
regression_model_name = st.selectbox(
    'Select a regression model:',
    regression_models
)

# Get user input for the features
so2 = st.number_input('Enter sulphur dioxide concentration(SO2)', format='%.2f')
no2 = st.number_input('Enter Nitrogen dioxide concentration (NO2)', format='%.2f')
rspm = st.number_input('Enter Respirable suspended particualte matter concentration (RSPM)', format='%.2f')
spm = st.number_input('Enter suspended particulate matter(SPM)', format='%.2f')
pm2_5 = st.number_input('Enter particulate matter 2.5(PM2.5)', format='%.2f')
SOi = st.number_input('Enter SOi Value', format='%.2f')
Noi = st.number_input('Enter Noi Value', format='%.2f')
Rpi = st.number_input('Enter Rpi Value', format='%.2f')
SPMi = st.number_input('Enter SPMi Value', format='%.2f')

# Create a DataFrame from the user input
input_df = pd.DataFrame([[so2, no2, rspm, spm, pm2_5, SOi, Noi, Rpi, SPMi]], columns=['so2', 'no2', 'rspm', 'spm', 'pm2_5', 'SOi', 'Noi', 'Rpi', 'SPMi'])

if st.button('Predict'):
    # Load and use the selected classification model
    with open(os.path.join(model_directory, classification_model_name), 'rb') as f:
        classification_model = pickle.load(f)

    # Predict the AQI_Range
    AQI_Range_pred = classification_model.predict(input_df)
    st.write(f'Predicted AQI Range: {AQI_Range_pred[0]}')

    # Load and display the confusion matrix
    matrix_name = classification_model_name.replace('_model.pkl', '_confusion_matrix.png')
    matrix_path = os.path.join(model_directory, matrix_name)
    if os.path.exists(matrix_path):
        st.image(matrix_path, caption='Confusion Matrix')
    else:
        st.write('No confusion matrix found for this model.')

    # Load and use the selected regression model
    with open(os.path.join(model_directory, regression_model_name), 'rb') as f:
        regression_model = pickle.load(f)

    # Predict the AQI
    AQI_pred = regression_model.predict(input_df)
    st.write(f'Predicted AQI: {AQI_pred[0]}')
