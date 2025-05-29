# NLP-project-
# TEXT SIMILARITY NLP
## 📝 Text Similarity Checker using NLP
## 👤 Author
## Name: Karim Gaber

# 📌 Project Overview
This project detects whether two questions are semantically similar using Natural Language Processing techniques. The model is trained and evaluated on the Quora Question Pairs dataset to identify duplicate questions.

# 🎯 Objectives
Preprocess and clean text data for model training.

Convert text into numerical representations using TF-IDF, Word2Vec, and SBERT.

Train and evaluate classification models: Logistic Regression, Support Vector Machine (SVM), Random Forest, and compare them.

Select the best model based on evaluation metrics.

# 📁 Dataset
Source: Quora Question Pairs on Kaggle

Description: Contains pairs of questions with a binary label (1 for duplicate, 0 for not duplicate).

Size: ~400,000 question pairs.

Preprocessing:

Lowercasing

Punctuation removal

Stopword removal

Tokenization

# 🧠 Models Used
Model	Description
Logistic Regression	Simple linear classifier
SVM	Margin-based binary classifier
Random Forest	Ensemble of decision trees
SBERT (Sentence-BERT)	Transformer-based embedding model

# ⚙️ Tools & Libraries
Python

scikit-learn

NLTK

Gensim

Sentence-Transformers (SBERT)

Matplotlib / Seaborn / Plotly

# 📊 Evaluation Metrics
Accuracy

Precision

Recall

F1-score

Confusion Matrix

Visual comparisons (interactive bar chart using Plotly)

# ✅ Results
All models were evaluated and compared visually.

The best-performing model based on accuracy and F1-score was [Insert your best model here, e.g., SBERT or SVM].

