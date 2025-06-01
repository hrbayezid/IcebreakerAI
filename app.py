import streamlit as st 
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('titanic_rf_model.pkl')

# Streamlit UI
st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details below:")

# 1. Inputs for all features
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Female", "Male"])  # Show names
sex = 0 if sex == "Female" else 1              # Convert to numeric

age = st.slider("Age", 0, 80, 25)

sibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)

embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_map[embarked]

# Family size = sibSp + parch + 1 (you)
family_size = sibSp + parch + 1

title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Other"])
title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
title = title_map[title]

# 2. Assemble input with all expected features in order
input_df = pd.DataFrame([{
    'pclass': pclass,
    'sex': sex,
    'age': age,
    'sibSp': sibSp,
    'parch': parch,
    'fare': fare,
    'embarked': embarked,
    'family_size': family_size,
    'title': title
}])

# 3. Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_df)
    result = "🟢 Survived!" if prediction[0] == 1 else "🔴 Did not survive."
    st.subheader(result)
