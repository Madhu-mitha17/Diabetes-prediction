import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Separate features and target
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Target (Outcome)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
model = SVC(kernel='linear', random_state=42)  # You can try other kernels like 'rbf'
model.fit(X_train, y_train)

# Function to take user input and predict
def predict_diabetes():
    print("\nEnter patient details:")
    pregnancies = float(input("Pregnancies: "))
    glucose = float(input("Glucose Level: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin Level: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))

    # Prepare the input in the same format as the model expects
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)  # Standardize input data

    # Make a prediction
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        print("\nPrediction: The patient is diabetic.")
    else:
        print("\nPrediction: The patient is not diabetic.")

# Main program
if __name__ == "__main__":
    # Model evaluation
    y_pred = model.predict(X_test)
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

    # Allow user to input patient details
    predict_diabetes()
