from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and prepare the dataset
dataset = pd.read_csv(r"C:\Users\CHARAN\Datascience-programs\MachineLeaarningPrograms\LinearRegression\SalaryPrediction\Salary_dataset.csv")
x = dataset[['YearsExperience']].values
y = dataset['Salary'].values

# Train the model
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Initialize FastAPI app
app = FastAPI()

# Mount static files (for CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates (for HTML files)
templates = Jinja2Templates(directory="templates")

# Define request model
class PredictionRequest(BaseModel):
    years_of_experience: float

@app.get("/")
async def get_form(request: Request):
    # Render the index.html page
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_salary(request: Request):
    form_data = await request.form()
    years_of_experience = float(form_data['years_of_experience'])
    
    # Make prediction
    prediction = regressor.predict([[years_of_experience]])
    
    # Return the prediction in the response
    return templates.TemplateResponse("index.html", {"request": request, "predicted_salary": round(prediction[0], 2), "years_of_experience": years_of_experience})
