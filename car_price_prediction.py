"""
=============================================================================
Project: Brand New Car Price Prediction using Machine Learning
Organization: CodeAlpha Internship (Task 2)
Author: Vedant (Data Science Intern)
Domain: Automobile Engineering & New Vehicle Pricing (MSRP / Ex-Showroom)
Technologies: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib
=============================================================================
"""

import os
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd

# Configure Matplotlib backend and temporary directory to prevent cache warnings
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib_new_car_cache')
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for headless and script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure stdout supports standard output encoding safely
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# USD to INR conversion rate for currency representation
USD_TO_INR = 83.50


def print_banner(title):
    """Prints a styled terminal banner for distinct project sections."""
    print("\n" + "=" * 70)
    print(f" {title.upper()} ")
    print("=" * 70)


def load_and_inspect_dataset(filepath="CarPrice_Assignment.csv"):
    """
    Step 1: Load the New Car dataset using Pandas and display:
    - Dimensions (rows and columns)
    - Column names and data types
    - First 5 rows
    - Missing values count
    - Statistical summary of numerical features
    """
    print_banner("Step 1: Loading & Inspecting New Car Dataset")
    
    if not os.path.exists(filepath):
        print(f"[!] Error: Dataset file '{filepath}' was not found in the current directory.")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    print(f"[+] Dataset successfully loaded from: '{filepath}'")
    print(f"[+] Dataset Dimensions: {df.shape[0]} Rows (New Car Models), {df.shape[1]} Columns\n")
    
    print("--- [Column Names & Data Types] ---")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"  * {col:18s} : {str(dtype)}")
        
    print("\n--- [First 5 Rows of New Car Specifications] ---")
    preview_cols = ['CarName', 'carbody', 'fueltype', 'horsepower', 'enginesize', 'curbweight', 'price']
    print(df[preview_cols].head().to_string(index=False))
    
    print("\n--- [Missing Values Count per Column] ---")
    null_counts = df.isnull().sum()
    print(f"Total null values across all columns: {null_counts.sum()}")
    if null_counts.sum() == 0:
        print("[+] Good news: Zero missing values found in the new car dataset.")
        
    print("\n--- [Statistical Summary of Key Numerical Features] ---")
    num_summary_cols = ['enginesize', 'horsepower', 'curbweight', 'highwaympg', 'price']
    print(df[num_summary_cols].describe().round(2).to_string())
    
    return df


def clean_and_prepare_data(df):
    """
    Step 2: Clean the dataset and perform feature normalization.
    - Extract brand name from 'CarName'
    - Correct typo variations in brand names (e.g. 'toyouta' -> 'toyota', 'maxda' -> 'mazda')
    - Verify deduplication
    """
    print_banner("Step 2: Data Cleaning & Brand Normalization")
    
    initial_rows = len(df)
    duplicate_count = df.duplicated().sum()
    print(f"[*] Detected duplicate rows: {duplicate_count}")
    
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[+] Removed {duplicate_count} duplicate rows. Remaining rows: {len(df)}")
    else:
        print("[+] Dataset contains no duplicate rows.")
        
    # Extract brand name (the first word in 'CarName')
    def clean_brand(name):
        b = str(name).strip().split()[0].lower()
        typo_mapping = {
            'maxda': 'mazda',
            'vw': 'volkswagen',
            'vokswagen': 'volkswagen',
            'toyouta': 'toyota',
            'porcshce': 'porsche'
        }
        return typo_mapping.get(b, b).title()

    df['brand'] = df['CarName'].apply(clean_brand)
    
    print("\n--- [Unique Categories in New Car Features] ---")
    print(f"  * Total Car Brands  : {df['brand'].nunique()} brands {sorted(df['brand'].unique())[:6]}...")
    print(f"  * Body Styles       : {list(df['carbody'].unique())}")
    print(f"  * Fuel Types        : {list(df['fueltype'].unique())}")
    print(f"  * Aspiration Types  : {list(df['aspiration'].unique())}")
    print(f"  * Drive Wheel Types : {list(df['drivewheel'].unique())}")
    print(f"  * Horsepower Range  : {df['horsepower'].min()} HP to {df['horsepower'].max()} HP")
    print(f"  * Price Range       : ${df['price'].min():,.0f} to ${df['price'].max():,.0f}")
    
    return df


def perform_eda(df, output_image="eda_new_car_plots.png"):
    """
    Step 3: Exploratory Data Analysis (EDA) and Visualization with Matplotlib/Seaborn.
    Generates and saves a high-resolution 6-panel chart tailored for new cars:
    1. Distribution of New Car Prices (MSRP)
    2. Horsepower vs. Price (Engine power correlation)
    3. Engine Size vs. Price (Displacement impact)
    4. Price by Body Style (Sedan, Hatchback, SUV/Wagon, Convertible)
    5. Top 8 Luxury & High-Value Brands by Average Price
    6. Correlation Heatmap of Vehicle Engineering Specs
    """
    print_banner("Step 3: Exploratory Data Analysis & Visualizations")
    print(f"[*] Generating exploratory charts and saving to '{output_image}'...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("New Car Price Prediction - Exploratory Data Analysis", fontsize=16, fontweight='bold')
    
    # Chart 1: Distribution of New Car Prices
    sns.histplot(df['price'], kde=True, ax=axes[0, 0], color='#2b5c8f', bins=25)
    axes[0, 0].set_title("1. New Car Price Distribution (MSRP)", fontweight='bold')
    axes[0, 0].set_xlabel("Price in USD ($)")
    axes[0, 0].set_ylabel("Frequency")
    
    # Chart 2: Horsepower vs Price
    sns.scatterplot(x='horsepower', y='price', data=df, hue='fueltype', ax=axes[0, 1], palette='Set1', s=55)
    sns.regplot(x='horsepower', y='price', data=df, scatter=False, ax=axes[0, 1], color='darkred', line_kws={'linestyle': '--'})
    axes[0, 1].set_title("2. Horsepower vs. New Car Price", fontweight='bold')
    axes[0, 1].set_xlabel("Engine Horsepower (HP)")
    axes[0, 1].set_ylabel("Price ($)")
    
    # Chart 3: Engine Size vs Price
    sns.scatterplot(x='enginesize', y='price', data=df, hue='drivewheel', ax=axes[0, 2], palette='Dark2', s=55)
    sns.regplot(x='enginesize', y='price', data=df, scatter=False, ax=axes[0, 2], color='navy', line_kws={'linestyle': '--'})
    axes[0, 2].set_title("3. Engine Size vs. New Car Price", fontweight='bold')
    axes[0, 2].set_xlabel("Engine Size (cubic inches)")
    axes[0, 2].set_ylabel("Price ($)")

    # Chart 4: Price by Body Style
    body_order = df.groupby('carbody')['price'].median().sort_values(ascending=False).index
    sns.boxplot(x='carbody', y='price', data=df, order=body_order, ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title("4. New Car Price by Body Style", fontweight='bold')
    axes[1, 0].set_xlabel("Body Style")
    axes[1, 0].set_ylabel("Price ($)")
    
    # Chart 5: Top Brands by Average Price
    top_brands = df.groupby('brand')['price'].mean().sort_values(ascending=False).head(8)
    sns.barplot(x=top_brands.values, y=top_brands.index, ax=axes[1, 1], palette='mako')
    axes[1, 1].set_title("5. Top 8 Highest Valuation Car Brands", fontweight='bold')
    axes[1, 1].set_xlabel("Average MSRP ($)")
    axes[1, 1].set_ylabel("Brand")
    for p in axes[1, 1].patches:
        axes[1, 1].annotate(f"${p.get_width():,.0f}", (p.get_width() / 2., p.get_y() + p.get_height() / 2.),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # Chart 6: Correlation Heatmap of Vehicle Engineering Specs
    spec_cols = ['price', 'horsepower', 'enginesize', 'curbweight', 'carwidth', 'highwaympg']
    corr_matrix = df[spec_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 2], square=True)
    axes[1, 2].set_title("6. Feature Correlation Heatmap", fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_image, dpi=300)
    plt.close()
    
    print(f"[+] EDA plots generated and saved as '{output_image}'.")


def build_machine_learning_pipeline():
    """
    Step 4: Build a Scikit-learn Pipeline with ColumnTransformer.
    - Numerical Features: Scaled using StandardScaler
    - Categorical Features: Encoded using OneHotEncoder(drop='first', handle_unknown='ignore')
    - Regressor: RandomForestRegressor(n_estimators=100, random_state=42)
    This isolates preprocessing inside the pipeline, completely preventing data leakage.
    """
    numerical_features = ['horsepower', 'enginesize', 'curbweight', 'highwaympg', 'carwidth', 'wheelbase']
    categorical_features = ['brand', 'carbody', 'fueltype', 'aspiration', 'drivewheel']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return rf_pipeline, lr_pipeline, numerical_features, categorical_features


def train_and_evaluate_models(rf_pipeline, lr_pipeline, X, y):
    """
    Step 5: Train and evaluate models on 80/20 train/test split and 5-fold cross validation.
    Reports real test metrics without fabricating results:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R-squared score (R²)
    """
    print_banner("Step 5: Model Training & Evaluation")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    print(f"[*] Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"[*] Testing samples : {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)\n")
    
    results = {}
    
    for name, pipeline in [("Linear Regression (Baseline)", lr_pipeline),
                           ("Random Forest Regressor (Primary)", rf_pipeline)]:
        # Fit pipeline strictly on training data
        pipeline.fit(X_train, y_train)
        
        # Predict on holdout test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate real metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 5-Fold Cross Validation across the entire dataset with shuffling
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
        cv_mae_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error')
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2_Mean': cv_r2_scores.mean(),
            'CV_R2_Std': cv_r2_scores.std(),
            'CV_MAE_Mean': cv_mae_scores.mean(),
            'pipeline': pipeline,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"--- [{name}] ---")
        print(f"  * Holdout Test MAE  : ${mae:,.2f} USD (~Rs. {mae * USD_TO_INR / 100000:.2f} Lakhs)")
        print(f"  * Holdout Test RMSE : ${rmse:,.2f} USD (~Rs. {rmse * USD_TO_INR / 100000:.2f} Lakhs)")
        print(f"  * Holdout Test R2   : {r2:.3f} ({r2*100:.1f}% of price variance explained)")
        print(f"  * 5-Fold CV Mean R2 : {cv_r2_scores.mean():.3f} (+/- {cv_r2_scores.std():.3f})")
        print(f"  * 5-Fold CV Mean MAE: ${cv_mae_scores.mean():,.2f} USD\n")
        
    # Display actual vs predicted samples for the primary Random Forest model
    rf_res = results["Random Forest Regressor (Primary)"]
    comp_df = pd.DataFrame({
        'Actual ($)': [f"${val:,.0f}" for val in rf_res['y_test'].values[:8]],
        'Predicted ($)': [f"${val:,.0f}" for val in rf_res['y_pred'][:8]],
        'Abs Error ($)': [f"${abs(a - p):,.0f}" for a, p in zip(rf_res['y_test'].values[:8], rf_res['y_pred'][:8])],
        'Actual (INR)': [f"Rs. {val * USD_TO_INR / 100000:.2f}L" for val in rf_res['y_test'].values[:8]],
        'Predicted (INR)': [f"Rs. {val * USD_TO_INR / 100000:.2f}L" for val in rf_res['y_pred'][:8]]
    })
    
    print("--- [Sample Real vs Predicted Prices (First 8 Test Set Cars)] ---")
    print(comp_df.to_string(index=False))
    
    return results["Random Forest Regressor (Primary)"]['pipeline'], results


def save_trained_model(pipeline, model_filepath="new_car_price_model.pkl"):
    """
    Step 6: Persist the trained model and preprocessing pipeline to disk using joblib.
    """
    print_banner("Step 6: Saving Model & Pipeline")
    joblib.dump(pipeline, model_filepath)
    print(f"[+] Trained pipeline successfully saved to '{model_filepath}'")
    
    # Verify reload
    loaded_pipeline = joblib.load(model_filepath)
    print(f"[+] Verified: Successfully reloaded model from '{model_filepath}'")
    return loaded_pipeline


def validate_float(prompt, min_val, max_val=None, unit_label=""):
    """Helper to validate numeric float input from the user."""
    while True:
        try:
            val_str = input(f"{prompt} ").strip()
            val = float(val_str)
            if val < min_val:
                print(f"  [!] Value must be greater than or equal to {min_val} {unit_label}.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Value must be less than or equal to {max_val} {unit_label}.")
                continue
            return val
        except ValueError:
            print("  [!] Invalid numeric format. Please enter a valid number (e.g. 120).")


def validate_choice(prompt, valid_options):
    """Helper to validate categorical choice from a set of allowed options."""
    options_str = "/".join(valid_options)
    options_lower = {opt.lower(): opt for opt in valid_options}
    while True:
        val_str = input(f"{prompt} ({options_str}): ").strip().lower()
        if val_str in options_lower:
            return options_lower[val_str]
        print(f"  [!] Invalid choice '{val_str}'. Please choose one of: {', '.join(valid_options)}")


def predict_new_car_price(pipeline, car_data_dict):
    """
    Formats car details into a Pandas DataFrame, makes prediction via the pipeline,
    and returns results in USD and Indian Rupees (Lakhs).
    """
    input_df = pd.DataFrame([car_data_dict])
    pred_usd = pipeline.predict(input_df)[0]
    pred_usd = max(3000.0, float(pred_usd))
    
    pred_inr = pred_usd * USD_TO_INR
    pred_lakhs = pred_inr / 100000.0
    
    return pred_usd, pred_inr, pred_lakhs


def interactive_prediction_menu(pipeline):
    """
    Step 7: User-friendly interactive prediction console with validation and unit explanations.
    """
    print_banner("Step 7: Interactive New Car Price Valuation")
    print("Welcome to the Brand New Car Price Valuation Estimator!")
    print("New Car Specifications explained:")
    print("  * Brand        : Manufacturer (e.g. Toyota, Honda, BMW, Audi, Hyundai, Nissan)")
    print("  * Body Style   : Sedan / Hatchback / Wagon / Hardtop / Convertible")
    print("  * Fuel Type    : Gas (Petrol) / Diesel")
    print("  * Aspiration   : Std (Naturally Aspirated) / Turbo")
    print("  * Drive Wheel  : FWD (Front-Wheel) / RWD (Rear-Wheel) / 4WD (All-Wheel)")
    print("  * Horsepower   : Engine power in HP (e.g., 100 HP to 250 HP)")
    print("  * Engine Size  : Engine displacement in cubic inches (e.g., 110 to 200)")
    print("  * Curb Weight  : Total vehicle weight in pounds (lbs, e.g., 2200 to 3800)")
    print("  * Highway MPG  : Mileage / Fuel Economy in Miles per Gallon (e.g., 25 to 45)\n")
    
    while True:
        print("-" * 50)
        print("OPTIONS: [1] Estimate Custom New Car Price   [2] Run Sample Test Car   [3] Exit")
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            print("\nPlease provide the new car specifications:")
            brand = input("Enter Brand (e.g. Toyota, Honda, BMW, Audi, Hyundai): ").strip().title()
            if not brand:
                brand = "Toyota"
                
            carbody = validate_choice("Select Body Style", ["Sedan", "Hatchback", "Wagon", "Hardtop", "Convertible"]).lower()
            fueltype = validate_choice("Select Fuel Type", ["Gas", "Diesel"]).lower()
            aspiration = validate_choice("Select Aspiration", ["Std", "Turbo"]).lower()
            drivewheel = validate_choice("Select Drive Wheel", ["FWD", "RWD", "4WD"]).lower()
            
            horsepower = validate_float("Enter Horsepower in HP (e.g. 110):", min_val=40, max_val=500, unit_label="HP")
            enginesize = validate_float("Enter Engine Size in cu in (e.g. 120):", min_val=50, max_val=400, unit_label="cu in")
            curbweight = validate_float("Enter Curb Weight in lbs (e.g. 2500):", min_val=1200, max_val=6000, unit_label="lbs")
            highwaympg = validate_float("Enter Highway Mileage in MPG (e.g. 32):", min_val=10, max_val=65, unit_label="MPG")
            
            # Default average dimensional values for convenience
            carwidth = 65.9
            wheelbase = 98.9
            
            car_dict = {
                'brand': brand,
                'carbody': carbody,
                'fueltype': fueltype,
                'aspiration': aspiration,
                'drivewheel': drivewheel,
                'horsepower': horsepower,
                'enginesize': enginesize,
                'curbweight': curbweight,
                'highwaympg': highwaympg,
                'carwidth': carwidth,
                'wheelbase': wheelbase
            }
            
            pred_usd, pred_inr, pred_lakhs = predict_new_car_price(pipeline, car_dict)
            
            print("\n" + "=" * 55)
            print("         NEW CAR VALUATION RESULT")
            print("=" * 55)
            print(f"Vehicle Model        : Brand New {brand} {carbody.title()} ({fueltype.title()})")
            print(f"Specifications       : {horsepower:.0f} HP | {enginesize:.0f} cu in Engine | {curbweight:.0f} lbs")
            print(f"Estimated Price (USD): ${pred_usd:,.2f} USD")
            print(f"Estimated Price (INR): Rs. {pred_lakhs:.2f} Lakhs (Rs. {pred_inr:,.0f})")
            print("=" * 55)
            
        elif choice == '2':
            # Predefined sample brand new family sedan
            sample_new_car = {
                'brand': 'Toyota',
                'carbody': 'sedan',
                'fueltype': 'gas',
                'aspiration': 'std',
                'drivewheel': 'fwd',
                'horsepower': 110,
                'enginesize': 122,
                'curbweight': 2400,
                'highwaympg': 34,
                'carwidth': 65.5,
                'wheelbase': 98.0
            }
            print("\n--- Running Sample Test: Brand New Toyota Sedan ---")
            print("Specs: 110 HP, 122 cu in Gas Engine, FWD, 34 Highway MPG, 2400 lbs Curb Weight")
            pred_usd, pred_inr, pred_lakhs = predict_new_car_price(pipeline, sample_new_car)
            print(f"[+] Estimated New Car Price (USD): ${pred_usd:,.2f}")
            print(f"[+] Estimated Ex-Showroom (INR)  : Rs. {pred_lakhs:.2f} Lakhs (Rs. {pred_inr:,.0f})\n")
            
        elif choice == '3':
            print("\nThank you for using the New Car Price Prediction System! Happy Learning!")
            break
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")


def main():
    """Main program execution workflow."""
    print("=" * 70)
    print("      CODEALPHA DATA SCIENCE INTERNSHIP - TASK 2")
    print("       BRAND NEW CAR PRICE PREDICTION MACHINE LEARNING PIPELINE")
    print("=" * 70)
    
    # 1. Load and inspect
    dataset_file = "CarPrice_Assignment.csv"
    raw_df = load_and_inspect_dataset(dataset_file)
    
    # 2. Clean and feature engineer
    clean_df = clean_and_prepare_data(raw_df)
    
    # 3. Exploratory Data Analysis & Visualization
    perform_eda(clean_df, output_image="eda_new_car_plots.png")
    
    # 4. Pipeline construction
    num_features = ['horsepower', 'enginesize', 'curbweight', 'highwaympg', 'carwidth', 'wheelbase']
    cat_features = ['brand', 'carbody', 'fueltype', 'aspiration', 'drivewheel']
    target = 'price'
    
    X = clean_df[num_features + cat_features]
    y = clean_df[target]
    
    rf_pipeline, lr_pipeline, num_cols, cat_cols = build_machine_learning_pipeline()
    
    # 5. Model training and evaluation
    trained_rf, all_results = train_and_evaluate_models(rf_pipeline, lr_pipeline, X, y)
    
    # 6. Save model to disk
    save_trained_model(trained_rf, model_filepath="new_car_price_model.pkl")
    
    # 7. Check if non-interactive (automated test or batch mode)
    if "--non-interactive" in sys.argv or "--batch" in sys.argv or not sys.stdin.isatty():
        print("\n[*] Running in batch / non-interactive demonstration mode:")
        sample_new_car = {
            'brand': 'Toyota',
            'carbody': 'sedan',
            'fueltype': 'gas',
            'aspiration': 'std',
            'drivewheel': 'fwd',
            'horsepower': 110,
            'enginesize': 122,
            'curbweight': 2400,
            'highwaympg': 34,
            'carwidth': 65.5,
            'wheelbase': 98.0
        }
        pred_usd, pred_inr, pred_lakhs = predict_new_car_price(trained_rf, sample_new_car)
        print("Sample Specs: Brand New Toyota Sedan | 110 HP | Gas Engine | 34 MPG")
        print(f"Predicted Price : ${pred_usd:,.2f} USD (~Rs. {pred_lakhs:.2f} Lakhs | Rs. {pred_inr:,.0f})")
        print("[+] Execution completed successfully.")
    else:
        interactive_prediction_menu(trained_rf)


if __name__ == "__main__":
    main()
