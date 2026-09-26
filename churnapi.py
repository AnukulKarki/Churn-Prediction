from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd

app = FastAPI()

#load models
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_name"]


# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

input_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float




@app.get("/a")
def home():
    return {"message": "Welcome to the backend" }

@app.post("/predict")
def get_churn(customer: CustomerData):

    input_data = customer.model_dump()

    #load the saved model and the label encoders
    
    
   

    input_data_df = pd.DataFrame([input_data])

    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])

    #make a prediction
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)

    return {
        "prediction":"Churn" if prediction[0] == 1 else "No Churn",
        "churn_probability":float(pred_prob[0][1])
    }






    
