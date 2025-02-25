Project completed on November 6, 2024.

# Walmart Store Sales Forecast

## Project Overview

This project focuses on **forecasting weekly sales for Walmart stores** using historical sales data across multiple stores and departments. Given sales records from **2010 to 2012**, the goal is to predict sales for the next two months based on past patterns.

The challenge includes:
* Handling time-series forecasting for multiple store-department combinations.
* Implementing dimensionality reduction to improve model performance.
* Designing a robust regression-based approach.

## Repository Contents

* `walmart_sales_model.py` – The main script for training models and generating predictions
* `Report.pdf` – Summary of data preprocessing, modeling techniques, and performance evaluation
* `Instructions.pdf` – Project description and dataset details
* `Proj2_Data/` - Directory for datasets
* `mypred.csv` - Generated output file containing predicted weekly sales for each store-department combination

## Dataset
The dataset comes from the Walmart Recruiting - Store Sales Forecasting [Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/overview) competition but has been modified for this project. 
It consists of:
* 10 different training/testing folds, each with:
  * `train.csv` – Contains weekly sales data for different stores and departments
  * `test.csv` – Contains the same store-department structure but without sales values
 * A final test file (`test_with_label.csv`) for evaluation

## Key Variables:
* **Store** – Store ID
* **Dept** – Department ID
* **Date** – Weekly timestamp
* **Weekly_Sales** – Target variable (only available in training data)
* **IsHoliday** – Boolean flag indicating a holiday week

## Exploratory Data Analysis & Preprocessing
The data preprocessing and transformation steps included:
* **Singular Value Decomposition (SVD)** to reduce noise in the sales data
* **Feature engineering**:
  * Extracting Week (Wk) and Year (Yr) from Date
  * Adding Yr² to capture nonlinear time effects
* **Filtering training data** to ensure each store-department pair appears in both train and test sets
* **Handling missing values** and ensuring full-rank design matrices for regression

## Model Implementation & Performance
1. **Regression-Based Sales Prediction**
* A linear regression model was trained for each Store-Department combination
* Dimensionality reduction (SVD) helped capture key sales patterns
* Feature adjustments were applied to avoid rank-deficient matrices

2. **Post-Processing for Holiday Adjustments**
To improve forecast accuracy during holiday periods, a correction method was applied based on historical sales patterns. The approach follows ideas discussed in:
* [Kaggle Walmart Sales Forecasting Discussion](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/discussion/8028)
* [GitHub: Walmart Sales Forecasting Post-Processing](https://github.com/davidthaler/Walmart_competition_code/blob/master/postprocess.R)

The method involved:
* **Baseline Sales Calculation**: Averaging weekly sales from Weeks 48 & 52 (non-holiday weeks).
* **Surge Sales Adjustment**: Detecting sales spikes in Weeks 49, 50, and 51.
* **Redistribution of Holiday Sales**: If the sales increase exceeded 10% compared to baseline, sales were redistributed across surrounding weeks using a circular shift technique.

This adjustment reduced Weighted Absolute Error (WAE) and improved forecasting accuracy for holiday periods.

## Results Summary
The model’s performance was evaluated using Weighted Absolute Error (WAE), where holiday weeks had higher importance.
1. **Baseline Linear Regression:** Average WAE = 1658
2. **\+ SVD Noise Reduction:** Average WAE = 1613
3. **\+ Yr² Feature:** Average WAE = 1585
4. **\+ Holiday Adjustments:** Average WAE = 1559.998 (best)
  
## How to Run the Code
1.	**Clone the repository and navigate to one of the project directories:**

`git clone https://github.com/SaniyaAbushakimova/Walmart-Store-Sales-Forecast.git`

`cd Walmart-Store-Sales-Forecast/proj2/foldX`

2. **Run the model script:**

`python walmart_sales_model.py --train train.csv --test test.csv`





