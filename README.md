# Toxic Comments Detection

## 📌 Project Description
This project focuses on predicting the toxicity of comments using Natural Language Processing (NLP) and Machine Learning (ML) techniques. Various approaches, including TF-IDF and Sentence BERT, were tested, along with several classification algorithms.
Metric: F1

## 🔧 Technologies
- **Programming Languages**: Python
- **Libraries**:
  - pandas, numpy, seaborn, matplotlib (data analysis and visualization)
  - nltk (text preprocessing)
  - scikit-learn (machine learning)
  - lightgbm (gradient boosting)
  - sentence-transformers (SBERT)

## 📊 Dataset
Link for downloading data: https://code.s3.yandex.net/datasets/toxic_comments.csv
The dataset `toxic_comments.csv` is used, containing comment texts and a binary toxicity label:
- `text` — comment text
- `toxic` — label (0 — non-toxic, 1 — toxic)

## 📌 Approach
1. **Data Preprocessing**
   - Text cleaning
   - Lemmatization
   - Stopword removal

2. **Text Vectorization**
   - TF-IDF
   - Sentence BERT (SBERT)

3. **Model Training**
   - Logistic Regression
   - Naive Bayes (MultinomialNB, GaussianNB)
   - Gradient Boosting (LightGBM)

4. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Model comparison using TF-IDF and SBERT

## 📈 Results
The best F1-score was achieved using **TF-IDF + Gradient Boosting**:
- **F1-score**: 0.7703
- **Accuracy**: 0.9596
- **ROC-AUC**: 0.9642

## 🚀 Running the Project
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open Jupyter Notebook and launch `toxic_comments_detection.ipynb`
3. Execute the cells in order

## 📌 Conclusions
- Lemmatization and stopword removal improve model performance
- TF-IDF + Gradient Boosting provides the best results
- Using Sentence BERT for creating vector representation opens new opportunities for further improvements

---
**Author:** [vadzimsmirnou]



