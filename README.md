# Financial News Classifier API

## Overview

This project implements a FastAPI-based REST API for classifying financial news articles into categories such as 'crypto', 'stocks', and 'other'. It uses a machine learning model trained on a large dataset of financial news articles.

## Features

- Fast and efficient classification of financial news articles
- RESTful API built with FastAPI
- Preprocessing of input text to improve classification accuracy
- Returns both the predicted category and the confidence level of the prediction

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone this repository:
   ```
   git clone repo_url
   cd financial-news-classifier
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have trained the model and generated the following files:
   - `financial_news_model.pkl`
   - `financial_news_vectorizer.pkl`

   These files should be in the same directory as the `main.py` file.

2. Start the FastAPI server:
   ```
   python main.py
   ```
   Or use uvicorn directly:
   ```
   uvicorn main:app --reload
   ```

3. The API will be available at `http://localhost:8000`. You can access the auto-generated API documentation at `http://localhost:8000/docs`.

4. To make a prediction, send a POST request to `http://localhost:8000/predict` with a JSON body:
   ```json
   {
     "text": "Your financial news article text here"
   }
   ```

5. The API will return a JSON response with the predicted category and confidence:
   ```json
   {
     "category": "predicted_category",
     "confidence": 0.95
   }
   ```

## API Endpoints

- `GET /`: Welcome message
- `POST /predict`: Predict the category of a financial news article

## Model Training

If you need to train or retrain the model:

1. Ensure you have the necessary data files.
2. Run the training script:
   ```
   python train_model.py
   ```
3. This will generate new `financial_news_model.pkl` and `financial_news_vectorizer.pkl` files.


## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [NLTK](https://www.nltk.org/) for natural language processing

