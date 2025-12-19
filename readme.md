This project uses supervised machine learning on a real Kaggle dataset to:

-Clean and preprocess flight data

-Engineer meaningful numerical features

-Train and compare ML regressors

-Store pipelines using Joblib

-Provide an interactive web-based prediction UI

-The dataset contains 10,683 flights, sourced from Kaggle, with price as the target variable 

| Model                  | Type                      | Performance             |
| ---------------------- | ------------------------- | ----------------------- |
| **Random Forest**      | Ensemble (Parallel Trees) | ⭐ Best overall          |
| **XGBoost**            | Ensemble (Boosted Trees)  | ⭐ Tied Best             |
| **MLP Neural Network** | Feed-Forward ANN          | Weaker for tabular data |

| Model         | R² Score   | MAE (₹)  | RMSE (₹) |
| ------------- | ---------- | -------- | -------- |
| Random Forest | **0.8905** | 635.55   | 1,489.56 |
| XGBoost       | **0.8905** | 635.55   | 1,489.56 |
| MLP           | 0.8456     | 1,078.97 | 1,768.80 |


Dataset

Source: Kaggle — Flight Fare Prediction

Link: https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh

Size: 10,683 rows, 11 raw features

Major preprocessing included:

Handling missing values

Extracting day/month/year/hour/minutes from date-time strings

Label & one-hot encoding

Dropping redundant Route & Additional_Info columns

Calculating duration and numeric stop counts



To run this program use commands

git clone https://github.com/Aditya-Gupta23/EDA_and_Flight_Price_Prediciton.git
cd Flight-Prediction

To install all the required libraries
pip install -r requirements.txt

To train and ceate the models run all the cells in Flight_Prediction_AI_Project.ipynb it is a jupyter notebook

after modles are trained enter command 
streamlit app.py
