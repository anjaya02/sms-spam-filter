# 📩 SMS/Email Spam Classifier

This project is a machine learning-based web app that classifies SMS or Email messages as **Spam** or **Not Spam** using **Natural Language Processing (NLP)** techniques and a **Multinomial Naive Bayes** classifier. It is built using **Python**, **Scikit-learn**, and **Streamlit** for web deployment.

---

## 🚀 Demo

Launch the app locally:

```bash
streamlit run app.py
```

## 🔍 Features

- Text preprocessing (lowercasing, punctuation removal, stopword removal, stemming)
- TF-IDF vectorization
- Multiple model evaluation (Naive Bayes, SVM, Random Forest, etc.)
- Word cloud & frequency visualization for spam and ham messages
- Interactive Streamlit app for live predictions

---

## 🧠 Tech Stack

- **Python** 🐍
- **NLTK** – Natural language preprocessing
- **Scikit-learn** – ML model training & evaluation
- **Streamlit** – Web interface
- **Matplotlib/Seaborn** – Data visualization

---

## 📁 Project Structure

```
sms_filter/
├── app.py                 # Streamlit frontend app
├── spam.csv              # Dataset
├── vectorizer.pkl        # Saved TF-IDF vectorizer
├── model.pkl             # Trained Naive Bayes model
├── train_and_save.py     # Model training and export script
└── README.md             # Project documentation
```

---

## 📦 Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/anjaya02/sms-spam-filter.git
cd sms-spam-classifier
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

---

## 📊 Model Performance

| Model            | Accuracy | Precision |
|------------------|----------|-----------|
| Multinomial NB   | 97.2%    | 100.0%    |
| Random Forest    | 97.4%    | 98.3%     |
| SVM (sigmoid)    | 97.3%    | 97.4%     |
| Ensemble Voting  | 98.2%    | 99.1%     |

---

## ✍️ Author

**Anjaya Induwara**  
[GitHub](https://github.com/anjaya02)

---

## 📘 License

This project is licensed under the MIT License.
```
