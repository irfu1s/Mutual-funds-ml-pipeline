# Mutual Funds ML Pipeline  
An end-to-end machine learning system using AMFI NAV data for SIP simulation, feature engineering, model training, SHAP explainability, and fund recommendation.

---

## 📌 Overview  
This project builds a complete financial ML workflow that processes real AMFI mutual fund NAV history and produces personalized investment recommendations.  

The system performs:

- Robust NAV ingestion using a **90-day sliding window method**  
- SIP simulation for every mutual fund  
- Feature engineering (returns, drawdowns, volatility, recent performance)  
- Model training using **XGBoost**  
- Explainability using **SHAP**  
- Intelligent recommendations based on user inputs such as:
  - SIP amount  
  - Duration  
  - Holding period  
  - Risk level  
  - Fund category  

---

## 🧠 Pipeline Architecture  

1. **Data Ingestion**  
   - Downloads AMFI NAV data in 90-day blocks  
   - Handles retry logic, timeouts, and continuity validation  
   - Produces `nav_history_raw.txt`

2. **Preprocessing**  
   - Cleans NAV data  
   - Converts dates, removes duplicates  
   - Produces `nav_daily_clean_filtered.csv`

3. **SIP Engine**  
   - Performs monthly SIP simulation for each scheme  
   - Computes:
     - Total invested  
     - Final value  
     - Profit  
     - CAGR  
   - Produces `sip_results_active_clean.csv`

4. **Feature Engineering**  
   - Adds SIP-based & NAV-based statistical features  
   - Computes recent 3-year CAGR, max drawdown, activity flag  
   - Saves `sip_features.csv`

5. **Model Training**  
   - Trains an XGBoost regressor to predict future CAGR  
   - Saves model + scaler + feature metadata into `/models`

6. **Explainability**  
   - SHAP summary, dependence, and feature impact plots  
   - Saved in `/shap_exports`

7. **Fund Recommendation Engine**  
   - Produces top-3 recommended funds for input parameters  
   - Filters by:
     - Risk level (derived from volatility)  
     - Category (large/mid/small cap)  
     - Active status  
   - Computes expected returns based on predicted CAGR  

---

## 📊 Example Output  
Top 3 BEST recommendations for:
₹5000/month, SIP 6 years, hold 4 years, risk=medium, category=large_cap

1. Nippon India Large Cap Fund - Direct Plan Growth Plan - Bonus Option (Code: 118633)
  Base Fund: NIPPON INDIA LARGE CAP FUND
  Category: large_cap
  Risk Label: medium
  Pred CAGR: 9.20%
  Invested: ₹360000.00
  Final Value: ₹671897.09
  Profit: ₹311897.09 (86.64% total)

2. ICICI Prudential Large Cap Fund (erstwhile Bluechip Fund) - Direct Plan - Growth (Code: 120586)
   Base Fund: ICICI PRUDENTIAL LARGE CAP FUND (ERSTWHILE BLUECHIP FUND)
   Category: large_cap
   Risk Label: medium
   Pred CAGR: 8.92%
   Invested: ₹360000.00
   Final Value: ₹659454.87
   Profit: ₹299454.87 (83.18% total)

3. CANARA ROBECO LARGE CAP FUND - DIRECT PLAN - GROWTH OPTION (Code: 118269)
Base Fund: CANARA ROBECO LARGE CAP FUND
   Category: large_cap
   Risk Label: medium
   Pred CAGR: 8.70%
   Invested: ₹360000.00
   Final Value: ₹650214.04
   Profit: ₹290214.04 (80.62% total)


---

## 📁 Project Structure  

  mutual-funds-ml-pipeline/
  │── data/
  │     ├── sample_nav.csv
  │     ├── sample_sip_results.csv
  │     └── scheme_metadata.csv
  │
  │── docs/
  │     ├── shap_summary.png
  │     └── dependence_years.png
  │
  │── models/
  │     ├── final_xgb_model.json
  │     ├── scaler.pkl
  │     └── feature_columns.json
  │
  │── src/
  │     ├── data_ingestion.py
  │     ├── nav_preprocessing.py
  │     ├── history_sanity_check.py
  │     ├── sip_calculator.py
  │     ├── feature_engineering.py
  │     ├── model_training.py
  │     ├── model_inference.py
  │     ├── shap_explainability.py
  │     └── recommender.py
  │
  │── requirements.txt
  └── README.md


---

## 🔍 SHAP Explainability  

SHAP visualizations help understand:  
- Which features increase or decrease predicted CAGR  
- How SIP duration, final value, return ratios, or drawdowns affect predictions  
- Model transparency for financial decision-making  

Examples include:  
- `shap_summary.png`  
- `dependence_years.png`  
- `dependence_total_invested.png`

---

## 🛠️ Technologies Used  

- Python  
- Pandas, NumPy  
- Scikit-Learn  
- XGBoost  
- SHAP  
- Matplotlib, Seaborn  
- Requests  
- Joblib  

---

## 📌 Next Steps (Future Work)

- Fully interactive **Streamlit UI**  
- NLP-powered **investment assistant** (`agent.py`)  
- Portfolio optimization  
- Risk-adjusted scoring  
- Real-time NAV ingestion automation (cron)

---

## 👤 Author  
Irfan — Data Science & Machine Learning  
Building practical AI systems from scratch with real financial data
=======
