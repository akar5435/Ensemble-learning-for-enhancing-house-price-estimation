from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import joblib
import re

# Initialise FastAPI app
app = FastAPI(
    title="UK House Price Predictor",
    description="Predict house prices using TF-IDF embeddings.",
    version="1.0.0"
)

# Mount static files directory to serve images
app.mount("/images_main", StaticFiles(directory="images_main"), name="images_main")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load pre-trained model and artifacts

lgbm_model = joblib.load('lgbm_model_tuned_tfidf.pkl')  # Default model
scaler = joblib.load('scaler_tfidf.pkl')
type_columns = joblib.load('type_columns_tfidf.pkl')
grouped_descriptions = joblib.load('grouped_descriptions_tfidf.pkl')
numerical_columns = joblib.load('numerical_columns_tfidf.pkl')
tfidf = joblib.load('tfidf_vectoriser.pkl')


# Define input data model using Pydantic (for API endpoint)
class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: int
    area: float = 0.0
    description: str = ""
    location: str = ""


# Clean text function
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower().replace(u'\xa0', u' ')
    text = re.sub(r'(gbp[^s])|(gbp)|(£)', ' gbp', text)
    text = re.sub(r'(m2)|(m²)', ' m²', text)
    text = re.sub(
        r'(?:(?:\+|00)33[\s.-]{0,3}(?:\(0\)[\s.-]{0,3})?|0)[1-9](?:(?:[\s.-]?\d{2}){4}|\d{2}(?:[\s.-]?\d{3}){2})|(\d{2}[ ]\d{2}[ ]\d{3}[ ]\d{3})',
        '', text)
    text = re.sub(
        r'(?:(?!.*?[.]{2})[a-zA-Z0-9](?:[a-zA-Z0-9.+!%-]{1,64}|)|\"[a-zA-Z0-9.+!% -]{1,64}\")@[a-zA-Z0-9][a-zA-Z0-9.-]+(.[a-z]{2,}|.[0-9]{1,})',
        '', text)
    text = re.sub(
        r'fr\d{2}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{2}|fr\d{20}|fr[ ]\d{2}[ ]\d{3}[ ]\d{3}[ ]\d{3}[ ]\d{5}', '',
        text)
    text = re.sub(r'(\(*)(ref|réf)(\.|[ ])\d+(\)*)', '', text)
    text = re.sub(r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*', '', text)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    return text


# Clean area function
def clean_area(area: float) -> float:
    if not area:
        return 0.0 #for flask compatibility
    area_str = str(area).replace('sq. ft', '').replace(',', '').strip()
    if 'From' in area_str or '-' in area_str:
        range_vals = re.findall(r'\d+', area_str)
        if range_vals:
            return sum(float(val) for val in range_vals) / len(range_vals)
    return float(area_str) if area_str.replace('.', '').isdigit() else 0.0


# Prediction function
def predict_house_price(bedrooms: int, bathrooms: int, area: float, description: str, location: str,
                        model=lgbm_model) -> float:
    bedrooms = int(bedrooms)
    bathrooms = int(bathrooms)
    area = clean_area(area)
    latitude, longitude = 0.0, 0.0
    if location:
        try:
            lat, lon = map(float, location.split(','))
            latitude, longitude = lat, lon
        except ValueError:
            pass

    numeric_input = np.array([[bedrooms, bathrooms, 0, area, latitude, longitude]], dtype=float)
    numeric_scaled = scaler.transform(numeric_input)

    cleaned_description = clean_text(description)
    embedding = tfidf.transform([cleaned_description]).toarray().reshape(1, -1)

    binary_input = np.array([[0] * len(grouped_descriptions)], dtype=float)
    type_input = np.array([[0] * len(type_columns)], dtype=float)

    input_features = np.hstack([embedding, numeric_scaled, binary_input, type_input])
    predicted_log_price = model.predict(input_features)[0]
    predicted_price = np.expm1(predicted_log_price)
    return predicted_price


# front end HTML
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "error": None})


@app.post("/", response_class=HTMLResponse)
async def predict_form(request: Request):
    form_data = await request.form()
    bedrooms = form_data.get("bedrooms")
    bathrooms = form_data.get("bathrooms")
    area = form_data.get("area", 0.0)
    description = form_data.get("description", "")
    location = form_data.get("location", "")

    if not bedrooms or not bathrooms:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": None,
            "error": "Bedrooms and Bathrooms are required."
        })

    try:
        price = predict_house_price(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            area=float(area) if area else 0.0,
            description=description,
            location=location
        )
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Predicted Price: £{price:,.2f}",
            "error": None
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": None,
            "error": f"Error: {str(e)}"
        })


# API prediction endpoint (optional)
@app.post("/predict")
async def predict_api(input_data: HouseInput):
    try:
        price = predict_house_price(
            bedrooms=input_data.bedrooms,
            bathrooms=input_data.bathrooms,
            area=input_data.area,
            description=input_data.description,
            location=input_data.location
        )
        return {"predicted_price": f"£{price:,.2f}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}