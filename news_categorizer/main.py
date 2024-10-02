from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Load the model
model_filename = 'bbc_news_model.pkl'
tfidf_filename = 'tfidf_vectorizer.pkl'
id_to_category_filename = 'id_to_category.pkl'

try:
    model = pickle.load(open(model_filename, 'rb'))
    tfidf = pickle.load(open(tfidf_filename, 'rb'))
    id_to_category = pickle.load(open(id_to_category_filename, 'rb'))
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model files not found. Please ensure the .pkl files are in the correct directory.")

class NewsItem(BaseModel):
    title: str
    description: str

@app.post("/predict")
async def predict_category(news_item: NewsItem):
    # Combine title and description
    combined_text = f"{news_item.title} {news_item.description}"
    
    # Transform the text using the loaded TfidfVectorizer
    text_features = tfidf.transform([combined_text])
    
    # Predict using the loaded model
    prediction = model.predict(text_features)
    
    # Get the category name
    category = id_to_category[prediction[0]]
    
    return {"category": category}

@app.get("/")
async def root():
    return {"message": "Welcome to the BBC News Classifier API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)