from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class Input(BaseModel):
   city : object
   city_development_index : float
   gender : object
   relevent_experience : object
   enrolled_university : object
   education_level : object
   major_discipline : object
   experience : object
   company_size : object
   company_type : object
   last_new_job : object
   training_hours : float

class Output(BaseModel):
   target:int


@app.get("/")
def read_root():
   return {"Hello":"World"}

@app.post("/predict")
def pr(input:Input) -> Output:
   model = joblib.load("jobchg_pipeline_model.pkl")
   X_input = pd.DataFrame([Input.dict()])
   prediction = model.predict(X_input)
   return Output(target=prediction[0])