import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("dataset.csv")

X = data[["StudyHours", "Attendance", "PreviousMarks", "Assignments"]]
y = data["FinalMarks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Website title
st.title("🎓 Student Performance Predictor")

st.write("Enter student details below:")

study = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous = st.number_input("Previous Marks", min_value=0.0, max_value=100.0)
assignments = st.number_input("Assignments Completed", min_value=0.0)

if st.button("Predict"):

    prediction = model.predict(
        [[study, attendance, previous, assignments]]
    )

    st.success(
        f"Predicted Final Marks: {prediction[0]:.2f}"
    )