from predict import predict_student

student = {

    "Marital status": 1,
    "Application mode": 17,
    "Application order": 1,
    "Course": 9500,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Previous qualification (grade)": 130,
    "Nationality": 1,
    "Mother's qualification": 19,
    "Father's qualification": 19,
    "Mother's occupation": 5,
    "Father's occupation": 5,
    "Admission grade": 135,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 1,
    "Age at enrollment": 19,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8,
    "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 14,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 8,
    "Curricular units 2nd sem (approved)": 6,
    "Curricular units 2nd sem (grade)": 14,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 12.4,
    "Inflation rate": 1.4,
    "GDP": 1.74

}

prediction, confidence = predict_student(student)

print("Prediction :", prediction)
print("Confidence :", confidence, "%")
