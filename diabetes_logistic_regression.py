import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the data
# Assuming you have a 'diabetes.csv' file with the Pima Indians Diabetes dataset
data = pd.read_csv('pima-indians-diabetes.csv')

# Step 2: Explore the data
print(data.head())  # View first few rows
print(data.info())  # Check data types and missing values

# Step 3: Preprocess the data
# Separate features and target
X = data.drop('Outcome', axis=1)  # Features (all columns except 'Outcome')
y = data['Outcome']               # Target (diabetes outcome: 0 or 1)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
