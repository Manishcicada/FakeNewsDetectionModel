📰 Fake News Detection using Machine Learning
This project aims to detect fake news articles using a variety of natural language processing (NLP) techniques and machine learning models. It explores text preprocessing, feature extraction, and model evaluation to build an effective fake news classifier.

📂 Project Structure
The notebook contains the following key sections:

Data Loading & Exploration

Text Preprocessing

Feature Engineering

Model Training

Hyperparameter Tuning

Model Evaluation

Visualization (Confusion Matrix, ROC Curve, SHAP Interpretations)

📌 Features Used
TF-IDF Vectorization

Lexical features (average sentence length, punctuation count)

Syntactic features (POS tags, stopwords count)

Semantic features (using TextBlob for polarity/subjectivity)

🧠 Models Trained
Logistic Regression

Decision Tree

Random Forest

Naive Bayes

Support Vector Machine (SVM)

XGBoost

K-Nearest Neighbors

Hyperparameter tuning is performed using:

GridSearchCV

RandomizedSearchCV

📊 Libraries Used
pandas, numpy for data handling

matplotlib, seaborn for visualization

nltk, TextBlob, spaCy for NLP

scikit-learn for ML models and metrics

xgboost

shap for model interpretability

📈 Evaluation Metrics
Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

ROC-AUC Score

SHAP value explanations

📌 How to Run
Clone the repository or download the notebook.

Make sure you have Python 3.x installed.

Install dependencies using:

pip install -r requirements.txt

Run the notebook using Jupyter or VS Code.

📝 Conclusion
This project showcases how different ML models perform on fake news classification and highlights the importance of feature engineering and explainability in NLP pipelines.