from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from utils import mask_pii

router = APIRouter()

# Load model and vectorizer
model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/tfidf.pkl")

class EmailInput(BaseModel):
    email_body: str

@router.post("/classify-email")
async def classify_email(data: EmailInput):
    input_text = data.email_body
    masked_email, entities = mask_pii(input_text)
    X_input = vectorizer.transform([masked_email])
    predicted_category = model.predict(X_input)[0]
    return {
        "input_email_body": input_text,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": predicted_category
    }
