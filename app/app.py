from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from schemas import EmailInput, HealthResponse, PredictionOutput
from utility import load_model_from_mr, load_nlp, load_vectorizer, preprocessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the API starts.
    Loads expensive resources into memory.

    """

    app.state.model = load_model_from_mr(model_name="MailGuard-API", model_version=1)

    app.state.vectorizer = load_vectorizer()

    app.state.nlp = load_nlp()

    print("✓ Model Loaded")
    print("✓ Vectorizer Loaded")
    print("✓ NLP Model Loaded")

    yield

    print("Application shutting down...")


app = FastAPI(
    title="Production MailGuard-API",
    description="Classifies the email as Spam or Not Spam",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Spam Email Classifier API",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "predict": "/predict", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint"""

    return {
        "status": "healthy",
        "model_loaded": app.state.model is not None,
        "vectorizer_loaded": app.state.vectorizer is not None,
        "nlp_loaded": app.state.nlp is not None,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(request: EmailInput):
    """Predict if a single email is spam or ham"""

    if app.state.model is None or app.state.vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess
        processed_email = preprocessor(text=request.text, nlp=app.state.nlp)

        if not processed_email:
            raise HTTPException(
                status_code=400, detail="Email text is empty after preprocessing."
            )

        # Vectorize
        X = app.state.vectorizer.transform([processed_email])

        prediction = app.state.model.predict(X)[0]

        probabilities = app.state.model.predict_proba(X)[0]

        # Fromat response
        label = "Spam" if prediction == 1 else "Not Spam"
        confidence = float(probabilities[prediction] * 100)

        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "probabilities": {
                "ham": round(float(probabilities[0]), 4),
                "spam": round(float(probabilities[1]), 4),
            },
            "cleaned_text": processed_email[:200] + "..."
            if len(processed_email) > 200
            else processed_email,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.get("/model/info", tags=["Model Info"])
async def model_info():
    """Get information about the loaded model."""

    if app.state.model is None or app.state.vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "Logistic Regression",
        "vectorizer_type": "TF-IDF",
        "max_features": app.state.vectorizer.max_features,
        "vocabulary_size": len(app.state.vectorizer.vocabulary_),
        "ngram_range": app.state.vectorizer.ngram_range,
        "preprocessing": [
            "Lowercase conversion",
            "URL removal",
            "Email address removal",
            "HTML tag removal",
            "Special character removal",
            "Stopwords removal",
            "Lemmatization",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
