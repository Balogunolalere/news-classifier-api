from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize FastAPI app
app = FastAPI(title="Financial News Classifier API")

# Load the model and vectorizer
def load_model(filename_prefix='financial_news'):
    model_filename = f'{filename_prefix}_model.pkl'
    vectorizer_filename = f'{filename_prefix}_vectorizer.pkl'
    
    try:
        loaded_model = pickle.load(open(model_filename, 'rb'))
        loaded_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model files not found. Please train the model first.")
    
    return loaded_model, loaded_vectorizer

model, vectorizer = load_model()

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Pydantic model for request body
class NewsArticle(BaseModel):
    text: str

# Pydantic model for response
class Prediction(BaseModel):
    category: str
    confidence: float

@app.post("/predict", response_model=Prediction)
async def predict_category(article: NewsArticle):
    # Preprocess the input text
    processed_text = preprocess_text(article.text)
    
    # Vectorize the text
    text_vector = vectorizer.transform([processed_text])
    
    # Make prediction
    category = model.predict(text_vector)[0]
    confidence = max(model.predict_proba(text_vector)[0])
    
    return Prediction(category=category, confidence=float(confidence))

@app.get("/")
async def root():
    return {"message": "Welcome to the Financial News Classifier API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)