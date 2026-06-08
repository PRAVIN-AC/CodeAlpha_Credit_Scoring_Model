# Nexus Bank: AI Credit Scoring Engine 🏦

A dynamic, production-ready FinTech application that evaluates financial risk profiles and underwrites credit applications in real-time. Built during my Machine Learning Internship at CodeAlpha, this project moves beyond standard Jupyter Notebooks by integrating synthetic data engineering, predictive modeling, and a highly interactive UI/UX dashboard.

## 🚀 Live FinTech Dashboard Preview
The application deploys a sleek, single-page Streamlit interface branded as a Neo-Banking platform (Nexus Bank), featuring:
1. **Interactive Underwriting Form:** Capture applicant data including age, income, existing debt, and derogatory marks.
2. **Real-Time Feature Engineering:** Automatically calculates critical banking metrics like the Debt-to-Income (DTI) ratio on the fly.
3. **Dynamic FICO-Style Gauge:** Uses Plotly to render a responsive visual gauge that translates default probabilities into a traditional 300–850 credit score.
4. **Interpretability:** Outputs explicit reasoning for application approval or denial based on the Random Forest decision logic.

---

## 🛠️ System Architecture & Engineering

### Core Technologies
* **Frameworks:** Python, Streamlit, Scikit-Learn, Joblib.
* **Visualizations:** Plotly Graph Objects.
* **Algorithm:** `RandomForestClassifier` optimized for robust tabular financial data.

### Data Engineering Pipeline
To ensure a reliable testing environment, the system utilizes a custom synthetic data generation pipeline (`generate_financial_data`). This engine generates 2,000 realistic banking records featuring non-linear relationships between income, credit utilization, and default risk, ensuring the Random Forest model learns realistic underwriting thresholds.

---

## 📂 Project Directory Structure

```text
CodeAlpha_Credit_Scoring_Model/
│
├── models/                     # Saved binary model artifacts (.pkl)
│   └── credit_rf_model.pkl
│
├── train_credit_model.py       # Data generation and Random Forest training script
├── credit_app.py               # Streamlit FinTech UI/UX Application
├── sample_credit_applications.csv # Auto-generated sample dataset
├── requirements.txt            # System dependencies
└── README.md                   # System documentation
