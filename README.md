# BBC News Classifier API

This FastAPI application provides an API for classifying news articles into categories using a pre-trained machine learning model. The model is trained on BBC News articles and can categorize articles based on their title and description.

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone this repository or download the source code.

2. Navigate to the project directory:
   ```
   cd news-classifier-api
   ```

3. Install the required dependencies:
   ```
   pip install fastapi uvicorn scikit-learn
   ```

4. Ensure you have the following files in the project directory:
   - `bbc_news_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `id_to_category.pkl`

   These files contain the trained model, TF-IDF vectorizer, and category mapping, respectively.

## Usage

1. Start the FastAPI server:
   ```
   python main.py
   ```
   Alternatively, you can use uvicorn directly:
   ```
   uvicorn main:app --reload
   ```

2. The API will be available at `http://localhost:8000`.

3. Use the `/predict` endpoint to classify news articles. Send a POST request with a JSON payload containing the `title` and `description` of the news article.

Example using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"title": "Manchester United signs new striker", "description": "The Premier League club has announced a record-breaking transfer deal for the young talent."}'
```

Example response:
```json
{"category": "sport"}
```

## API Endpoints

- `GET /`: Welcome message
- `POST /predict`: Predict the category of a news article

## Input Format

The `/predict` endpoint expects a JSON payload with the following structure:

```json
{
  "title": "Article title",
  "description": "Article description or content"
}
```

## Output Format

The API returns a JSON object with the predicted category:

```json
{
  "category": "predicted_category"
}
```

## Error Handling

The API will return appropriate HTTP status codes and error messages for invalid requests or server errors.

## Customization

You can modify the `main.py` file to add more endpoints, change the input/output format, or integrate additional functionality as needed.

