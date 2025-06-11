from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from pydantic import BaseModel

# Load the trained model
model = joblib.load(r"wine_quality.joblib")

# Create FastAPI instance
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    with open(r"index.html") as file:
        return file.read()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)  # Change port here

# Define request structure
class WineFeatures(BaseModel):
    features: list  # Example: [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]

# API route for prediction
@app.post("/predict")
def predict_wine_quality(data: WineFeatures):
    try:
        # Convert input to numpy array and reshape for prediction
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"wine_quality": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

# Run with: uvicorn main:app --reload
