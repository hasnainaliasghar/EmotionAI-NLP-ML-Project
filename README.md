# EmotionAI 🧠

A machine-learning web app that detects emotions from text using a TF-IDF + Logistic Regression pipeline.

**Emotions detected:** sadness 😢 · anger 😠 · love ❤️ · surprise 😲 · fear 😨 · joy 😄

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model
- Vectorizer: `TfidfVectorizer` (bigrams, sublinear TF)
- Classifier: `LogisticRegression` (C=10, liblinear solver)
- Trained with scikit-learn 1.7.1
