# 📰 Fake News Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)

> A machine learning project to classify news articles as **real or fake** using text preprocessing, NLP features, and supervised ML models.

---

## 📚 Table of Contents

- [📖 Introduction](#-introduction)
- [📂 Project Structure](#-project-structure)
- [🛠️ Features Extracted](#️-features-extracted)
- [🤖 Models Used](#-models-used)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [🚀 How to Run](#-how-to-run)
- [🧪 Requirements](#-requirements)
- [📈 Sample Results](#-sample-results)
- [📝 License](#-license)

---

## 📖 Introduction

Fake news has emerged as a significant challenge in the information age. This project focuses on using NLP and machine learning to build a model capable of distinguishing between real and fake news articles.

---

## 📂 Project Structure

```bash
.
├── fakenewdetection.ipynb     # Main Jupyter notebook
├── requirements.txt           # Required dependencies
├── README.md                  # Project documentation
└── LiarDataset.csv                      # Dataset (not included)
```

---

## 🛠️ Features Extracted

- Lowercasing, tokenization, stopword removal, stemming
- TF-IDF vectorization
- Sentence and word count
- Punctuation and character analysis
- POS tag ratios
- Polarity & subjectivity using `TextBlob`
- Named entity counts using `spaCy`

---

## 🤖 Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Naive Bayes
- K-Nearest Neighbors
- XGBoost

Hyperparameter tuning via:
- `GridSearchCV`
- `RandomizedSearchCV`

---

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Curve
- SHAP value-based feature importance visualization

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Manishcicada/FakeNewsDetectionModel
   cd FakeNewsDetectionModel
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download necessary resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')

   import spacy
   spacy.cli.download("en_core_web_sm")
   ```

4. Open and run the notebook:
   ```bash
   jupyter notebook fakenewdetection.ipynb
   ```

---

## 🧪 Requirements

```
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
xgboost
textblob
spacy
shap
scipy
```

---

## 📈 Sample Results

| Model             | Accuracy | F1 Score |
|------------------|----------|----------|
| Random Forest     | 0.70     | 0.69     |
| SVM               | 0.64     | 0.63     |
| Decision Tree     | 0.69     | 0.69     |
| XGBoost           | 0.74     | 0.73     |

SHAP plots were used for model interpretability.

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [spaCy](https://spacy.io/)
- [SHAP](https://shap.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/)

---

> Made with ❤️ by **Manish Sharma**
