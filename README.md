# Household Energy Consumption Prediction

## 📄 Project Overview
This project was developed after a thorough analysis of the `Household_Energy.py` file. The file includes a complete end-to-end pipeline for data loading, cleaning, preprocessing, modeling, evaluation, and visualization using Streamlit. The goal is to provide insights into household energy usage and build various regression models to predict future energy consumption patterns.

## 📌 Problem Statement
In the modern world, energy management is a critical issue for both households and energy providers. Predicting energy consumption accurately enables better planning, cost reduction, and optimization of resources. The goal of this project is to develop a machine learning model that can predict household energy consumption based on historical data. Using this model, consumers can gain insights into their usage patterns, while energy providers can forecast demand more effectively.

By the end of this project, learners should provide actionable insights into energy usage trends and deliver a predictive model that can help optimize energy consumption for households or serve as a baseline for further research into energy management systems.

---

## 📁 Repository Contents
This repository contains:
- `Household_Energy.py` – Python script with full machine learning pipeline and Streamlit app
- `household_power_consumption.zip` – Zipped dataset file (must be extracted to access `.txt` file)

---

## 📦 Required Packages
This project uses the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `streamlit`
- `scikit-learn`

### 🛠 Install Packages with pip
```bash
pip install pandas numpy matplotlib seaborn streamlit scikit-learn
```

---

## 🚀 How to Run the App
1. Make sure the `household_power_consumption.txt` file is available (extract from the zip).
2. Run the Streamlit application with the command:
```bash
streamlit run Household_Energy.py
```

---

## 🔄 Project Workflow (Step-by-Step)
1. **Load Dataset** – Read the household energy consumption data
2. **Clean Data** – Remove missing values and invalid records
3. **Skew Analysis** – Analyze and visualize distribution of numeric features
4. **Outlier Handling** – Use IQR capping to limit extreme values
5. **Correlation Analysis** – Identify relationships between features
6. **Feature Selection & Splitting** – Select inputs and divide into training and testing sets
7. **Model Training & Evaluation**:
   - Linear Regression
   - KNN Regression
   - Decision Tree Regression
   - Random Forest Regression
   - Ridge Regression
8. **Visualization** – Display R², MAE, MSE, RMSE scores in Streamlit dashboard

---

## 🔓 License
This project is provided under a **free and open license** for educational and non-commercial use.

---

## 🙌 Acknowledgements
- UCI Machine Learning Repository: Household Power Consumption Dataset
- Streamlit for the interactive web interface
- scikit-learn for machine learning models

---

Happy Coding! ✨

