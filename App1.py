import streamlit as st# type: ignore
import numpy as np# type: ignore
import pandas as pd# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.ensemble import GradientBoostingRegressor# type: ignore
from sklearn.metrics import r2_score# type: ignore

# Set page config
st.set_page_config(page_title='Medical Insurance Prediction ', page_icon=':hospital:', layout='wide')

# Load and preprocess the data
medical_df = pd.read_csv('insurance 22.csv')
medical_df.replace({'sex':{'male':1,'female':0}},inplace=True)
medical_df.replace({'smoker':{'yes':1,'no':0}},inplace=True)
medical_df.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)
medical_df.replace({'Diabetes':{'Yes':1,'No':0}}, inplace=True)
medical_df.replace({'BloodPressureProblems':{'Yes':1,'No':0}}, inplace=True)
medical_df.replace({'AnyTransplants':{'Yes':1,'No':0}}, inplace=True)
medical_df.replace({'AnyChronicDiseases':{'Yes':1,'No':0}}, inplace=True)
medical_df.replace({'KnownAllergies':{'Yes':1,'No':0}}, inplace=True)
medical_df.replace({'HistoryOfCancerInFamily':{'Yes':1,'No':0}}, inplace=True)


# Split the data
X = medical_df.drop('charges',axis=1)
y = medical_df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Train the model
GR = GradientBoostingRegressor()
GR.fit(X_train,y_train)

# App title
st.title("CareOptima-Medical Insurance Premium Prediction Model For Self")

# User inputs
st.header('Enter Person Details')
age = st.number_input('Age', min_value=15, max_value=100)
sex = st.selectbox('Sex', options=['Male', 'Female'])
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0)
smoker = st.selectbox('Smoker', options=['Yes', 'No'])
region = st.selectbox('Region', options=['Southeast', 'Southwest', 'Northwest', 'Northeast'])
diabetes = st.selectbox('Diabetes', options=['Yes', 'No'])
blood_pressure_problems = st.selectbox('Blood Pressure Problems', options=['Yes', 'No'])
any_transplants = st.selectbox('Any Transplants', options=['Yes', 'No'])
any_chronic_diseases = st.selectbox('Any Chronic Diseases', options=['Yes', 'No'])
known_allergies = st.selectbox('Known Allergies', options=['Yes', 'No'])
history_of_cancer_in_family = st.selectbox('History of Cancer in Family', options=['Yes', 'No'])
number_of_major_surgeries = st.number_input('Number of Major Surgeries', min_value=0, max_value=10)

# Convert  features to binary
sex = 1 if sex == 'Male' else 0
smoker = 1 if smoker == 'Yes' else 0
region = {'Southeast':0, 'Southwest':1, 'Northwest':2, 'Northeast':3}[region]
diabetes = 1 if diabetes == 'Yes' else 0
blood_pressure_problems = 1 if blood_pressure_problems == 'Yes' else 0
any_transplants = 1 if any_transplants == 'Yes' else 0
any_chronic_diseases = 1 if any_chronic_diseases == 'Yes' else 0
known_allergies = 1 if known_allergies == 'Yes' else 0
history_of_cancer_in_family = 1 if history_of_cancer_in_family == 'Yes' else 0

# Predict button
if st.button('Predict'):
    # Check for history of cancer in family
    if history_of_cancer_in_family == 1:
        st.warning('Health insurance premium cannot be predicted for cancer.')
    else:
        # Make prediction
        input_data = np.array([age, sex, bmi, smoker, region, diabetes, blood_pressure_problems, any_transplants, any_chronic_diseases, known_allergies, history_of_cancer_in_family, number_of_major_surgeries])
        prediction = GR.predict(input_data.reshape(1,-1))

        # Determine coverage based on prediction
        if prediction[0] < 5000:
            coverage = '2 Lakh'
        elif prediction[0] < 10000:
            coverage = '5 Lakh'
        elif prediction[0] < 20000:
            coverage = '10 Lakh'
        else:
            coverage = '15 Lakh'

        # Display prediction and coverage
        st.success(f'Medical Insurance for this person is: {prediction[0]}')
        st.success(f'Coverage amount is: {coverage}')





