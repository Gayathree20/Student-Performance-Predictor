import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset.csv")

X = data[["StudyHours", "Attendance", "PreviousMarks", "Assignments"]]
y = data["FinalMarks"]

model = LinearRegression()
model.fit(X, y)

study = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance: "))
previous = float(input("Enter Previous Marks: "))
assignments = float(input("Enter Assignments Completed: "))

prediction = model.predict([[study, attendance, previous, assignments]])

print("Predicted Marks:", prediction[0])