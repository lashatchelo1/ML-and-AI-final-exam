"""
Binary classifier for network traffic:
- Label 0 = normal
- Label 1 = abnormal (attack/anomaly)

Requirements:
    pip install numpy pandas scikit-learn matplotlib

You can run this script as-is (it uses synthetic data),
then replace the synthetic data block with loading your own CSV.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Create synthetic "network traffic" dataset (example only)
# ------------------------------------------------------------
# Features could represent:
# - duration
# - src_bytes, dst_bytes
# - packets_in, packets_out
# - src_port, dst_port (encoded)
# - flags, protocol, etc.
# Here we create 10 numeric features and a binary label.

X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=2,
    weights=[0.8, 0.2],  # 80% normal, 20% abnormal
    flip_y=0.01,
    random_state=42
)

# Put into a DataFrame just to show column naming
feature_names = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "packets_in",
    "packets_out",
    "src_port_enc",
    "dst_port_enc",
    "protocol_enc",
    "tcp_flags_enc",
    "flow_rate"
]
df = pd.DataFrame(X, columns=feature_names)
df["label"] = y

print("Dataset shape:", df.shape)
print(df.head())

# ------------------------------------------------------------
# 2. (Optional) How YOU would load real data instead
# ------------------------------------------------------------
# Uncomment and adapt this part when you have a CSV:
#
# df = pd.read_csv("network_traffic.csv")
# feature_names = [
#     "duration",
#     "src_bytes",
#     "dst_bytes",
#     "packets_in",
#     "packets_out",
#     "src_port_enc",
#     "dst_port_enc",
#     "protocol_enc",
#     "tcp_flags_enc",
#     "flow_rate"
# ]
# X = df[feature_names].values
# y = df["label"].values  # 0 = normal, 1 = abnormal

# For now we reuse the synthetic X, y defined above
X = df[feature_names].values
y = df["label"].values

# ------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ------------------------------------------------------------
# 4. Build pipeline: StandardScaler + RandomForest
# ------------------------------------------------------------
# StandardScaler normalizes numeric features.
# RandomForest handles non-linear relationships and is robust.

pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",  # handle class imbalance
        random_state=42,
        n_jobs=-1
    ))
])

# ------------------------------------------------------------
# 5. Train model
# ------------------------------------------------------------
pipeline.fit(X_train, y_train)
print("\nModel training completed.")

# ------------------------------------------------------------
# 6. Evaluate model
# ------------------------------------------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of class 1 (abnormal)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC AUC:", roc_auc)

# Plot ROC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve - Normal vs Abnormal Network Traffic")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 7. Example: using the trained model for new flows
# ------------------------------------------------------------
example_flows = np.array([
    # Example "normal-looking" flow
    [0.2, 1500, 2300, 10, 12, 0.1, 0.2, 1.0, 0.0, 50],
    # Example "abnormal-looking" flow (fake values)
    [10.0, 50000, 80000, 2000, 2100, 0.9, 0.95, 2.0, 1.0, 3000]
], dtype=float)

pred_classes = pipeline.predict(example_flows)
pred_proba = pipeline.predict_proba(example_flows)[:, 1]

for i, (c, p) in enumerate(zip(pred_classes, pred_proba)):
    label = "ABNORMAL (attack/anomaly)" if c == 1 else "NORMAL"
    print(f"\nFlow #{i+1}: predicted = {label}, probability_attack={p:.3f}")
