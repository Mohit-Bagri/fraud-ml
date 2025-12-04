# 🛡️ Credit Card Fraud Detection (Imbalanced Classification)

This project builds a complete fraud detection pipeline using a **synthetic imbalanced dataset** created with scikit-learn. Fraud represents only **1%** of the data, making this a realistic challenge for machine learning systems.

The goal of this project is to practice:

- Handling severe class imbalance  
- Using SMOTE correctly  
- Class weighting  
- Comparing Random Forest, XGBoost, and LightGBM  
- Hyperparameter tuning with GridSearchCV  
- Precision–Recall evaluation  
- Feature importance analysis  
- Model explainability with SHAP  

---

## 📂 Project Structure

```
fraud-ml/
│
├── main.py            # End-to-end fraud detection pipeline
├── requirements.txt   # Dependencies
└── README.md
```

---

## 📊 Dataset

We generate a synthetic fraud dataset using:

```
from sklearn.datasets import make_classification
```

### Dataset properties:

- 100,000 samples  
- 20 features  
- 10 informative, 5 redundant  
- **1% fraud cases** (highly imbalanced)  
- Moderate class separation (`class_sep=2`)  

**Target variable**
```
Class → 0 (legitimate), 1 (fraud)
```

---

## 🤖 Models Used

### **1. Baseline Random Forest**
- Trained on raw imbalanced data  
- Evaluates natural model behavior without handling imbalance  

---

### **2. Weighted Random Forest**
- Uses `class_weight="balanced"`  
- Compensates for minority under-representation  

---

### **3. SMOTE + RandomForest Pipeline**
- SMOTE oversampling expands minority class in **training only**  
- Combined with RandomForest in a Pipeline  
- Evaluated using **Stratified K-Fold PR-AUC**  
- Prevents data leakage and gives reliable metrics  

---

### **4. XGBoost Classifier**
- Handles imbalance using `scale_pos_weight`  
- Excellent performance on tabular datasets  

---

### **5. LightGBM Classifier**
- Uses `class_weight="balanced"`  
- Very fast and accurate  
- Often top performer in fraud detection competitions  

---

## 🔧 Hyperparameter Tuning

We use:

```
GridSearchCV
```

to tune RandomForest hyperparameters inside the SMOTE pipeline:

- n_estimators  
- max_depth  
- min_samples_split  

Scored using:

```
average_precision (PR-AUC)
```

---

## 📈 Evaluation

All models are evaluated using:

- **PR-AUC** (primary metric)  
- Classification Report  
- Confusion Matrix  
- Precision–Recall Curve  

PR-AUC is preferred because accuracy and ROC-AUC are misleading for rare-event detection.

---

## 🔍 Explainability (SHAP)

SHAP is used to:

- Show global feature importance  
- Understand individual fraud predictions  
- Provide transparency for audits and model trust  

The project includes SHAP summary plots for interpretability.

---

## ▶️ How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the project:

```
python main.py
```

---

## 📊 Example Output (Varies by run)

```
Baseline PR-AUC: 0.69
Weighted PR-AUC: 0.69
Cross-val PR-AUC (SMOTE + RF): 0.68

=== XGBOOST RESULTS ===
PR-AUC: ~0.72

=== LIGHTGBM RESULTS ===
PR-AUC: ~0.73

=== BEST GRIDSEARCH MODEL ===
{'model__n_estimators': 400, ...}
```

Boosted models (XGBoost, LightGBM) generally perform best.

---

## ✅ Summary

- Fraud datasets require special handling due to extreme imbalance  
- SMOTE + Pipeline + Stratified CV is the correct approach  
- PR-AUC is the most reliable metric for rare classes  
- Boosted models outperform classical models  
- SHAP provides model transparency  

---
