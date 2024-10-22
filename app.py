from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and prepare the dataset
dataset = pd.read_csv(r"C:\Users\CHARAN\Datascience-programs\MachineLeaarningPrograms\LinearRegression\SalaryPrediction\Salary_dataset.csv")

# Separate features and target variable
x = dataset[['YearsExperience']].values
y = dataset['Salary'].values

# Correct the train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=0)

# Train the Linear Regression model with the correct split
regressor = LinearRegression()
regressor.fit(x_train, y_train)

app = FastAPI()

# Define the request model
class PredictionRequest(BaseModel):
    years_of_experience: float

@app.post("/salaryPrediction/")
async def predict_salary(request: PredictionRequest):
    # Make prediction using the request data
    prediction = regressor.predict([[request.years_of_experience]])
    
    # Return the predicted salary as a JSON response
    return {"predicted_salary": float(prediction[0])}
