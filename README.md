# 🧠 Breast Cancer Prediction using Logistic Regression

This project builds and deploys a machine learning model that predicts whether a breast tumor is **benign** or **malignant** using the Breast Cancer Wisconsin dataset. The model is trained using **logistic regression**, and a **Streamlit** web interface allows for real-time predictions with adjustable sliders for all features.

---

## 🚀 Demo

![ DEMO ](https://github.com/Shaun-Roy/BreastCancerPredUsingLogisticRegression/blob/main/Screenshot%202025-07-07%20104416.png)

---

## 📌 Features

- Built using **scikit-learn** and **logistic regression**
- Predicts breast cancer diagnosis based on **30 clinical features**
- Interactive **Streamlit app** with sliders for user input
- Real-time prediction with **confidence scores**
- Clean and responsive layout

---

## 🗂️ Dataset

- Source: [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- 569 samples
- 30 numeric features per sample
- Target classes: `Malignant` (1), `Benign` (0)

---

## 📊 Model Training Overview

The notebook includes:
- Data preprocessing and cleaning
- Feature scaling with `StandardScaler`
- Binary encoding of the diagnosis column
- Model training with `LogisticRegression`
- Saving the model and scaler using `joblib`

---

## 🖥️ Streamlit Web App

The app:
- Loads the trained model and scaler
- Uses sliders for all 30 features
- Makes real-time predictions
- Displays classification and confidence

### Run it locally:

```bash
git clone https://github.com/Shaun-Roy/BreastCancerPredUsingLogisticRegression.git
cd BreastCancerPredUsingLogisticRegression
pip install -r requirements.txt
streamlit run streamlit_app.py
