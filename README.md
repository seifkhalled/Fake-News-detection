# Fake News Detection
**MAIM NLP Track Final Project**

**Author:** Seif Eldeen Khalid Nabil  
**Course:** MAIM NLP Track  
**Date:** 9/4/2025

---

## Project Overview

This project presents a complete **Natural Language Processing (NLP) pipeline** for detecting fake news and analyzing sentiment in news articles and social media posts. The system integrates **classical machine learning models, deep learning sequence models, and Transformer-based architectures**, providing an end-to-end solution for real-time fake news classification.

The project includes:  
1. **Classical Machine Learning Models:** Naïve Bayes, Logistic Regression, and SVM.  
2. **Sequential Models with Pretrained Embeddings:** RNN, GRU, LSTM with Word2Vec, GloVe, and FastText.  
3. **Transformer Models:** BERT, fine-tuned for fake news detection.  
4. **Flask Deployment:** Provides a web interface for users to input a news headline or article and get a real-time prediction.

---

## Dataset

The dataset consists of two CSV files:

- **Fake.csv:** 23,502 fake news articles  
- **True.csv:** 21,417 real news articles

Each file contains the following columns:  
- `title`: The headline of the news article  
- `text`: The main body of the article  
- `subject`: The news category (e.g., politics, world news)  
- `date`: The publish date

The datasets were combined and labeled (`0` for fake, `1` for real) for modeling.

---

---

## Preprocessing

- Lowercased all text  
- Removed URLs, emails, punctuation, and non-alphabetic characters  
- Tokenized text into words  
- Removed stopwords  
- Applied **lemmatization with part-of-speech tagging** to preserve semantic meaning  
- Processed both article bodies (`text`) and titles (`title`)  

---

## Modeling

### 1. Classical ML
- Feature extraction: Bag of Words (BoW) and TF-IDF  
- Models: Naïve Bayes, Logistic Regression, SVM  

### 2. Sequential Models
- Pretrained embeddings: Word2Vec, GloVe, FastText  
- Models: RNN, GRU, LSTM  

### 3. Transformer Models
- Fine-tuned **BERT** for fake news classification  
- Attention mechanisms used for interpretability  

---

## Evaluation Metrics

All models were evaluated using:  
- Accuracy  
- Precision  
- Recall  
- F1-score  

Comparisons between classical ML, sequential models, and Transformers show that **BERT achieved the highest performance**, while classical ML models provide lightweight and fast baselines.

---

## Deployment

- **Flask app** allows users to input a news headline or article  
- Returns a real-time prediction: **Fake** or **Real**  
- Supports practical application for news verification and content moderation  

**To run the app:**

```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py



