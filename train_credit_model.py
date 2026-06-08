import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def generate_financial_data(n_samples=2000):
    """Generates a realistic synthetic banking dataset."""
    print("Generating synthetic financial data...")
    np.random.seed(42)
    
    # Base Features
    age = np.random.randint(21, 70, n_samples)
    income = np.random.randint(30000, 150000, n_samples)
    debt = np.random.randint(1000, 60000, n_samples)
    years_employed = np.random.randint(0, 30, n_samples)
    credit_util = np.random.uniform(0.05, 0.95, n_samples)
    late_payments = np.random.poisson(0.5, n_samples) # Most have 0, some have 1, 2, etc.
    
    # Feature Engineering: Debt-to-Income (DTI) Ratio
    dti_ratio = debt / income
    
    # Target Logic: Complex simulated risk calculation
    risk_score = (dti_ratio * 3) + (credit_util * 2) + (late_payments * 1.5) - (years_employed * 0.05)
    # 1 = High Risk/Default, 0 = Low Risk/Approved
    target = (risk_score > 2.0).astype(int) 
    
    df = pd.DataFrame({
        'Age': age,
        'Annual_Income': income,
        'Total_Debt': debt,
        'DTI_Ratio': dti_ratio,
        'Years_Employed': years_employed,
        'Credit_Utilization': credit_util,
        'Late_Payments': late_payments,
        'Risk_Flag': target
    })
    return df

def main():
    # 1. Prepare Data
    df = generate_financial_data()
    X = df.drop('Risk_Flag', axis=1)
    y = df['Risk_Flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Train Model
    print("Training Random Forest Credit Assessor...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 3. Evaluate Model
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Financial Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # 4. Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/credit_rf_model.pkl')
    
    # Save a sample CSV for the user to test the dashboard batch upload
    df.head(20).to_csv('sample_credit_applications.csv', index=False)
    print("\nModel saved to 'models/credit_rf_model.pkl'")
    print("Sample dataset saved as 'sample_credit_applications.csv'")

if __name__ == "__main__":
    main()