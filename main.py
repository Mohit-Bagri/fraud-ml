from sklearn.datasets import make_classification
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


X, y = make_classification(
    n_samples=100_000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    weights=[0.99, 0.01],  # 1% fraud
    class_sep=2,
    random_state=42,
)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
df["Class"] = y

print(df.head())


from sklearn.model_selection import train_test_split

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)

rf_base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)

# Predict probabilities
y_proba_base = rf_base.predict_proba(X_test)[:, 1]
y_pred_base = (y_proba_base >= 0.5).astype(int)

# PR AUC (very important metric)
ap_base = average_precision_score(y_test, y_proba_base)

print("Baseline PR AUC:", ap_base)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_base))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_base))


rf_weighted = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)

rf_weighted.fit(X_train, y_train)

y_proba_w = rf_weighted.predict_proba(X_test)[:, 1]
y_pred_w = (y_proba_w >= 0.5).astype(int)

ap_w = average_precision_score(y_test, y_proba_w)

print("Weighted PR AUC:", ap_w)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_w))


smote_rf = ImbPipeline(
    [
        ("smote", SMOTE(sampling_strategy=0.1, random_state=42)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
            ),
        ),
    ]
)
scores = cross_val_score(
    smote_rf,
    X,
    y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="average_precision",
    n_jobs=-1,
)
print("Cross-val PR-AUC:", scores.mean())
smote_rf.fit(X_train, y_train)
y_pred = smote_rf.predict(X_test)
y_proba_sm = smote_rf.predict_proba(X_test)[:, 1]


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

prec, rec, thres = precision_recall_curve(y_test, y_proba_sm)

plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve — SMOTE + RF")
plt.grid()
plt.show()
