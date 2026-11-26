# 🚗 Brand New Car Price Prediction using Machine Learning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CodeAlpha Internship](https://img.shields.io/badge/CodeAlpha-Data%20Science%20Internship-purple.svg)](https://www.codealpha.tech/)

A complete, beginner-friendly end-to-end Machine Learning project developed as **Task 2** for the **CodeAlpha Data Science Internship**. 

Unlike used-car depreciation models, this project builds a machine learning regression pipeline to predict the **MSRP / Showroom Price of Brand New Cars** directly based on their engineering specifications, engine power, vehicle dimensions, brand valuation, and fuel efficiency.

---

## 📌 Table of Contents
- [Project Overview: New Cars vs. Used Cars](#-project-overview-new-cars-vs-used-cars)
- [Dataset Details & Source](#-dataset-details--source)
  - [Dataset Source](#dataset-source)
  - [Features Description](#features-description)
- [Machine Learning Architecture](#-machine-learning-architecture)
  - [Preventing Data Leakage with Pipelines](#preventing-data-leakage-with-pipelines)
  - [Models Evaluated](#models-evaluated)
- [Model Evaluation & Real Results](#-model-evaluation--real-results)
  - [Test Set Performance Table](#test-set-performance-table)
  - [Sample Actual vs. Predicted Prices](#sample-actual-vs-predicted-prices)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
  - [1. Interactive Mode](#1-interactive-mode)
  - [2. Automated / Demonstration Mode](#2-automated--demonstration-mode)
- [Loading the Saved Model (Joblib)](#-loading-the-saved-model-joblib)
- [Technologies Used](#-technologies-used)
- [Author & Acknowledgements](#-author--acknowledgements)

---

## 📖 Project Overview: New Cars vs. Used Cars

| Factor | Used Car Price Prediction | **Brand New Car Price Prediction (This Project)** |
| :--- | :--- | :--- |
| **Pricing Principle** | Based on **Depreciation** | Based on **Engineering & Vehicle Tier** |
| **Odometer Mileage** | Changes price drastically (e.g. 50,000 km) | **0 km** (Fresh from factory / showroom) |
| **Vehicle Age** | Strong negative impact | **0 Years** (Brand new manufacturing) |
| **Past Owners** | Number of past owners matters | **0 Owners** (First direct buyer) |
| **Key Price Drivers** | Wear and tear, maintenance history | **Horsepower, Engine Displacement, Weight, Brand, Body Style, Fuel Efficiency** |

This project provides automobile manufacturers, dealerships, and buyers with an AI-driven valuation tool to determine fair new vehicle MSRP.

---

## 📊 Dataset Details & Source

### Dataset Source
* **Origin**: Publicly available **Automobile New Car Price Dataset** (`CarPrice_Assignment.csv`) compiled from automotive market data and Geely Auto's market entry study. Widely distributed on Kaggle and open-source GitHub repositories under open educational terms.
* **Direct File**: Included in this repository as `CarPrice_Assignment.csv`.
* **Dimensions**: 205 new car models, 26 vehicle specifications.

### Features Description

| Feature | Type | Description |
| :--- | :--- | :--- |
| `brand` | Categorical | Manufacturer (Toyota, Honda, BMW, Audi, Porsche, Nissan, etc.) |
| `carbody` | Categorical | Body style (`sedan`, `hatchback`, `wagon`, `hardtop`, `convertible`) |
| `fueltype` | Categorical | Fuel consumed (`gas` / `diesel`) |
| `aspiration` | Categorical | Induction system (`std` naturally aspirated vs. `turbo` turbocharged) |
| `drivewheel` | Categorical | Drivetrain layout (`fwd` front-wheel, `rwd` rear-wheel, `4wd` all-wheel) |
| `horsepower` | Numerical | Engine power output in **Horsepower (HP)** |
| `enginesize` | Numerical | Engine displacement in **cubic inches** |
| `curbweight` | Numerical | Total unloaded vehicle weight in **pounds (lbs)** |
| `highwaympg` | Numerical | Fuel efficiency on highways in **Miles per Gallon (MPG)** |
| `carwidth` | Numerical | Vehicle width in inches |
| `wheelbase` | Numerical | Distance between front and rear axle centers in inches |
| **`price`** | **Target ($y$)** | **Brand New Car Selling / Showroom Price (USD and INR equivalent)** |

---

## 🛠 Machine Learning Architecture

```
Raw New Car Specs (CarPrice_Assignment.csv)
       │
       ▼
Data Cleaning & Brand Normalization (Fixing typos: 'toyouta' -> 'toyota', 'maxda' -> 'mazda')
       │
       ▼
Train/Test Split (80% Train, 20% Test)
       │
       ▼
Scikit-Learn ColumnTransformer Pipeline
 ├── Numerical: StandardScaler() ───────► ['horsepower', 'enginesize', 'curbweight', 'highwaympg', 'carwidth', 'wheelbase']
 └── Categorical: OneHotEncoder(drop='first') ──► ['brand', 'carbody', 'fueltype', 'aspiration', 'drivewheel']
       │
       ▼
RandomForestRegressor(n_estimators=100, random_state=42)
       │
       ▼
Evaluation & Persistence (new_car_price_model.pkl)
```

### Preventing Data Leakage with Pipelines
All preprocessing (scaling and one-hot encoding) is packaged directly into a Scikit-Learn `Pipeline`. The transformer fits only on the training set (`X_train`) and transforms the test set (`X_test`), guaranteeing that no test distribution insights leak into model training.

---

## 📈 Model Evaluation & Real Results

### Test Set Performance Table

| Model | Holdout MAE | Holdout RMSE | Holdout $R^2$ Score | 5-Fold CV Mean $R^2$ (± Std) | 5-Fold CV Mean MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | $2,277 USD (~₹1.90L) | $3,284 USD (~₹2.74L) | 0.863 (86.3%) | 0.871 (± 0.048) | $1,952 USD (~₹1.63L) |
| **Random Forest Regressor (Primary)** | **$1,287 USD (~₹1.08L)** | **$1,856 USD (~₹1.55L)** | **0.956 (95.6%)** | **0.924 (± 0.045)** | **$1,495 USD (~₹1.25L)** |

*Note: USD values converted to INR at standard baseline rate ($1 USD ≈ ₹83.50 INR).*

#### Metric Explanations:
* **$R^2$ Score (0.956 / 95.6%)**: The Random Forest model explains **95.6% of the variance** in new car prices based strictly on vehicle specs and brand tier.
* **5-Fold Cross-Validation (0.924)**: Proves that the model generalizes robustly across all 5 unseen folds without overfitting.
* **MAE ($1,287 USD)**: On a new vehicle priced between $10,000 and $35,000, average error is ~90%+ accurate to actual sticker price.

### Sample Actual vs. Predicted Prices

| Actual Price ($) | Model Prediction ($) | Abs Error ($) | Equivalent in INR | Prediction Accuracy |
| :---: | :---: | :---: | :---: | :--- |
| **$31,400** | **$33,267** | $1,867 | Rs. 26.22 Lakhs vs Rs. 27.78 Lakhs | Accurate luxury tier capture |
| **$17,859** | **$18,194** | **$335** | Rs. 14.91 Lakhs vs Rs. 15.19 Lakhs | **~₹28,000 difference!** |
| **$9,549** | **$9,015** | $534 | Rs. 7.97 Lakhs vs Rs. 7.53 Lakhs | Accurate economy hatchback |
| **$28,248** | **$28,476** | **$228** | Rs. 23.59 Lakhs vs Rs. 23.78 Lakhs | **Near perfect (<1% error)** |
| **$7,788** | **$7,827** | **$39** | Rs. 6.50 Lakhs vs Rs. 6.54 Lakhs | **Spot-on (~₹3,200 difference)** |

---

## 📉 Exploratory Data Analysis (EDA)

The script automatically generates `eda_new_car_plots.png`, featuring 6 charts:
1. **New Car Price Distribution**: Highlights entry-level economy ($5k–$12k), mid-tier ($12k–$22k), and luxury tiers ($22k+).
2. **Horsepower vs. Price**: Powerful positive correlation showing that higher output engines command exponential price premiums.
3. **Engine Size vs. Price**: Engine displacement directly correlates with showroom cost.
4. **Price by Body Style**: Convertibles and Hardtops demand higher pricing than standard Sedans and Hatchbacks.
5. **Top Luxury Brands**: Highlights premium valuations commanded by Porsche, Jaguar, Buick, and BMW.
6. **Feature Correlation Matrix**: Confirms strong relationships between curb weight (+0.84), engine size (+0.87), horsepower (+0.81), and price.

---

## 📁 Project Structure

```
CodeAlpha_CarPricePrediction/
│
├── car_price_prediction.py       # Main end-to-end Python script (New Car Model)
├── CarPrice_Assignment.csv       # Brand new car dataset (205 rows, 26 specs)
├── new_car_price_model.pkl       # Saved trained pipeline (Joblib)
├── eda_new_car_plots.png         # Generated 6-panel high-resolution charts
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git exclusions
└── README.md                     # Comprehensive project documentation
```

---

## 💻 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/CodeAlpha_CarPricePrediction.git
   cd CodeAlpha_CarPricePrediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### 1. Interactive Mode
Run the script to train the model, save charts, and open the interactive valuation prompt:
```bash
python car_price_prediction.py
```

Sample interaction:
```
Please provide the new car specifications:
Enter Brand (e.g. Toyota, Honda, BMW, Audi, Hyundai): Toyota
Select Body Style (Sedan/Hatchback/Wagon/Hardtop/Convertible): Sedan
Select Fuel Type (Gas/Diesel): Gas
Select Aspiration (Std/Turbo): Std
Select Drive Wheel (FWD/RWD/4WD): FWD
Enter Horsepower in HP (e.g. 110): 110
Enter Engine Size in cu in (e.g. 120): 122
Enter Curb Weight in lbs (e.g. 2500): 2400
Enter Highway Mileage in MPG (e.g. 32): 34

=======================================================
         NEW CAR VALUATION RESULT
=======================================================
Vehicle Model        : Brand New Toyota Sedan (Gas)
Specifications       : 110 HP | 122 cu in Engine | 2400 lbs
Estimated Price (USD): $9,664.98 USD
Estimated Price (INR): Rs. 8.07 Lakhs (Rs. 807,026)
=======================================================
```

### 2. Automated Demonstration Mode
```bash
python car_price_prediction.py --non-interactive
```

---

## 💾 Loading the Saved Model (Joblib)

```python
import joblib
import pandas as pd

# 1. Load the trained new car pipeline
model = joblib.load("new_car_price_model.pkl")

# 2. Define custom vehicle specs
new_car = pd.DataFrame([{
    'brand': 'Honda',
    'carbody': 'sedan',
    'fueltype': 'gas',
    'aspiration': 'std',
    'drivewheel': 'fwd',
    'horsepower': 105,
    'enginesize': 115,
    'curbweight': 2300,
    'highwaympg': 36,
    'carwidth': 65.0,
    'wheelbase': 97.5
}])

# 3. Predict showroom price
pred_usd = model.predict(new_car)[0]
pred_inr_lakhs = (pred_usd * 83.50) / 100000

print(f"Estimated MSRP (USD): ${pred_usd:,.2f}")
print(f"Estimated INR Price : Rs. {pred_inr_lakhs:.2f} Lakhs")
```

---

## 🧰 Technologies Used

- **Language**: Python 3.12
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
- **Numerical Computing**: [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (`Pipeline`, `ColumnTransformer`, `RandomForestRegressor`, `LinearRegression`, `StandardScaler`, `OneHotEncoder`)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)

---

## 👤 Author & Acknowledgements

- **Developer**: Vedant
- **Role**: Data Science Intern at [CodeAlpha](https://www.codealpha.tech/)
- **Project**: Internship Task 2 - Brand New Car Price Prediction

*Special thanks to CodeAlpha for this practical machine learning project!*
