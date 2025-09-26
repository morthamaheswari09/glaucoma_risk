# %%
# %%
import streamlit as st # streamlit is an py lib to create web applications
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Glaucoma Risk  Prediction App")

# Load the trained logistic regression model
try:
    with open('log_reg_model2.pkl', 'rb') as file:
        log_reg_model2 = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'log_reg_model2.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the label encoder for decoding predictions
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Primary Open-Angle Glaucoma', 'Juvenile Glaucoma' ,'Congenital Glaucoma',
 'Normal-Tension Glaucoma' ,'Angle-Closure Glaucoma', 'Secondary Glaucoma'])  # Adjust based on notebook's encoding

# Sidebar for user inputs
st.sidebar.header("Enter Patient Details")

# Numerical features
age = st.sidebar.slider("Age", min_value=18, max_value=59, value=42)
Intraocular_Pressure= st.sidebar.slider("Intraocular Pressure (IOP)", min_value=10.0, max_value=25.0 ,value=15.0)
Cup_to_Disc_Ratio= st.sidebar.slider("Cup-to-Disc Ratio (CDR)", min_value=0.0, max_value=1.0, value=0.5)
Pachymetry = st.sidebar.slider("Pachymetry", min_value=500.0, max_value=600.0, value=550.0)


# Categorical features
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
diagnosis = st.sidebar.selectbox("Diagnosis", options=[
   "No Glaucoma","Glaucoma"
])
vam=st.sidebar.selectbox("Visual Acuity Measurements",options=['LogMAR 0.1', '20/40', 'LogMAR 0.0', '20/20'])
family_history = st.sidebar.selectbox("Family History", options=["Yes","No"])
Cataract_Status= st.sidebar.selectbox("Cataract Status", options=['Present', 'Absent'])
Angle_Closure_Status= st.sidebar.selectbox("Angle Closure Status", options=['Open', 'Closed'])

# Function to preprocess input data
def preprocess_input(age,Intraocular_Pressure,Cup_to_Disc_Ratio,Pachymetry,gender,diagnosis,vam,family_history,Cataract_Status,Angle_Closure_Status):
    # Create a DataFrame with numerical features
    data = {
        'Age': age,
        'Intraocular Pressure (IOP)':Intraocular_Pressure,
        'Cup-to-Disc Ratio (CDR)':Cup_to_Disc_Ratio,
        'Pachymetry':Pachymetry,
        'Gender':gender,
        'Visual Acuity Measurements':vam, 
        'Family History':family_history, 
        'Cataract Status':Cataract_Status, 
        'Angle Closure Status':Angle_Closure_Status, 
        'Diagnosis':diagnosis
    }
    df = pd.DataFrame([data])

    # One-hot encode categorical features
    df['Gender_Female'] = 1 if gender == 'Female' else 0
    df['Gender_Male'] = 1 if gender == 'Male' else 0
    
    df['Diagnosis_Galucoma'] = 1 if diagnosis == 'Glaucoma' else 0
    df['Diagnosis_No Galucoma'] = 1 if diagnosis == 'No Glaucoma' else 0

    df['Family History_Yes'] = 1 if family_history == 'Yes' else 0
    df['Family History_No'] = 1 if family_history == 'No' else 0

    df['Cataract Status_Present'] = 1 if Cataract_Status == 'Present' else 0
    df['Cataract Status_absent'] = 1 if Cataract_Status == 'Absent' else 0

    df['Angle Closure Status_Open'] = 1 if Angle_Closure_Status == 'Open' else 0
    df['Angle Closure Status_Closed'] = 1 if Angle_Closure_Status == 'Closed' else 0

    vam_cat=['LogMAR 0.1', '20/40', 'LogMAR 0.0', '20/20']
    for i in vam_cat:
        df[f' Visual Acuity Measurements_{i}'] = 1 if vam == i else 0

    # Ensure all expected columns are present in the correct order
    expected_columns = [
        'Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)',
       'Pachymetry',  'Gender_Female', 'Gender_Male',
       'Visual Acuity Measurements_20/20', 'Visual Acuity Measurements_20/40',
       'Visual Acuity Measurements_LogMAR 0.0',
       'Visual Acuity Measurements_LogMAR 0.1', 'Family History_No',
       'Family History_Yes', 'Cataract Status_Absent',
       'Cataract Status_Present', 'Angle Closure Status_Closed',
       'Angle Closure Status_Open', 'Diagnosis_Glaucoma',
       'Diagnosis_No Glaucoma'
    ]
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

# Button to make prediction
if st.sidebar.button("Predict"):
    # Preprocess the input
    input_df = preprocess_input(
        age,Intraocular_Pressure,Cup_to_Disc_Ratio,Pachymetry,gender,diagnosis,vam,family_history,Cataract_Status,Angle_Closure_Status
    )
    
    # Make prediction
    try:
        prediction = log_reg_model2.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Display result
        st.subheader("Prediction Result")
        st.write(f"The predicted Glaucoma Risk is: **{predicted_label}**")
        if predicted_label == "None":
            st.write("No Glaucoma Risk predicted.")
        else:
            st.write(f"The patient may have {predicted_label}.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Display instructions
st.write("""
### Instructions
1. Use the sidebar to enter the patient's details.
2. Adjust the sliders for numerical features like Age, IOP , etc.
3. Select appropriate options for Gender, Diagnosis, Angle Closure Status.
4. Click the 'Predict' button to see the predicted sleep disorder.
""")




# %%





