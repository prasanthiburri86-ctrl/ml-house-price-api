from fastapi import FastAPI
import joblib

app = FastAPI()

# load model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API running"}

@app.get("/predict")
def predict(size: int):
    prediction = model.predict([[size]])
    return {
        "house_size": size,
        "predicted_price": int(prediction[0])
    }
