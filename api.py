from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import numpy as np
import uvicorn
import joblib


app = FastAPI()

modele = joblib.load("best_model.joblib")

class type_var(BaseModel):
    age_months : float
    weight_kg  : float
    height_cm  : float
    muac_cm    : float
    bmi        : float

@app.get("/")
def accueil():
    return {"message": "Hello pagui, t'es prêt pour les prédictions ?"}

def message(classe):
    if classe == 1:
        nom = "NORMALE"
        cas = "Cet enfant est en bonne forme"
    elif classe == 0:
        nom = "MODERE"
        cas = "Ce cas est a surveiller"
    else:
        nom = "SEVERE"
        cas = "A diagnostiquer le plus tot possible"
    return f"Le taux de manutrition de cet enfant est : {nom}, {cas}"

@app.post("/prediction")
def predire(Data: type_var):
    data = pd.DataFrame([Data.model_dump()])   # crochets = une seule ligne
    prediction = int(modele.predict(data)[0])       # [0] = la valeur, pas le tableau    # JSON, pas une phrase
    status = message(prediction)
    return {
            "Prediction": prediction ,
            "Status"    : status
        }      

if __name__ == "__main__":
    uvicorn.run(app)
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# Lien : https://fr.pornhub.com/view_video.php?viewkey=6792551517f17