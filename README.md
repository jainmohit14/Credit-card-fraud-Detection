# Credit Card Fraud Detection
This project analyzes and predicts fraudulent transactions using the popular Credit Card Fraud Detection Dataset. The dataset is highly imbalanced, containing genuine and fraudulent transactions.

The goal is to build a machine learning model that can classify transactions as:
0 = Not Fraud
1 = Fraud

# 🚀 Features
- Data exploration and visualization of transaction patterns
- Preprocessing (feature scaling, handling imbalance)
- Logistic Regression model for classification
- Comparison between normal training and class-balanced training
- Evaluation using confusion matrix, classification report, and accuracy

## 📦 Installation
Clone the repo and install dependencies:
git clone https://github.com/RishiSrivastava17/Credit-Card-Fraud-.git
cd Credit-Card-Fraud-

pip install -r requirements.txt

requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn

## 📊 Dataset
The dataset used is Credit Card Fraud Detection Dataset
Place the CSV file (creditcard.csv) in the project folder before running the notebook/script.

## ▶️ Usage
Run the Jupyter Notebook or Python script step by step:
jupyter notebook fraud_detection.ipynb

OR

python fraud_detection.py

## 📈 Results
Class Distribution

The dataset is highly imbalanced:
- Majority class → Not Fraud (0)
- Minority class → Fraud (1)

Model Performance

- Without class balancing:
  - High accuracy (due to imbalance), but poor fraud detection

- With class balancing (class_weight='balanced'):
  - Better recall and precision for fraud transactions

- Accuracy Score: 97%

## 📂 Project Structure
├── fraud_detection.ipynb   # Jupyter notebook version
├── fraud_detection.py      # Python script version
├── creditcard.csv          # Dataset (not included, download from Kaggle)
├── requirements.txt
└── README.md

## 📜 License

MIT License – free to use and modify.
