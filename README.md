# 👁️ Diabetic Retinopathy Detection System

This project implements a machine learning-based diagnostic tool to assist in the early detection of diabetic retinopathy using clinical ocular features.

## 🚀 Live Application
Access the tool here: **[Diabetic Retinopathy Detection App](https://diabeticretinbpathy.streamlit.app/)**

## 📖 Overview
Diabetic retinopathy is a complication of diabetes that affects the eyes. A **LinearSVC** model, trained on a clinical dataset, is implemented to classify patients as "Healthy" or "Patient". To maximize diagnostic sensitivity, the decision threshold was optimized to **-0.2**.

## 📊 Dataset Information
The model was trained on a clinical dataset consisting of 921 entries. The data includes various ocular metrics used for diagnostic purposes:

* **Key Features:** **Microaneurysms (ma1-ma6):** Indicators of early-stage retinopathy.
    * **Exudates (exudate1-exudate8):** Deposits of protein and lipids.
    * **Morphological Data:** Macula-optic disc distance, optic disc diameter, and AM/FM classification.

## 📈 Model Performance
Several standard machine learning classifiers were evaluated to determine the most effective approach for the dataset. The **LinearSVC** model was selected for its balanced performance, and its predictive capability was further enhanced by adjusting the decision threshold to -0.2.

| Classifier | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **LinearSVC** | **0.717** | **0.817** | **0.598** | **0.690** |
| LGBMClassifier | 0.696 | 0.753 | 0.629 | 0.685 |
| GradientBoosting | 0.663 | 0.688 | 0.660 | 0.674 |
| XGBClassifier | 0.658 | 0.693 | 0.629 | 0.659 |

---
*Disclaimer: This project is for educational purposes and does not substitute professional medical diagnosis.*
