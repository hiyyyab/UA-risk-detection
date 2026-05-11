import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load data
data = pd.read_csv("d_dset.csv")

X = data.drop(columns=["Diabetes_binary"])
y = data["Diabetes_binary"].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# ================= BEFORE =================
y_pred_before = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_before)
tn, fp, fn, tp = cm.ravel()

print("\nLOGISTIC REGRESSION - BEFORE DEFERRAL")
print(classification_report(y_test, y_pred_before))
print("Error rate:", (fp + fn) / len(y_test))
print("False negatives:", fn)

# ================= DEFERRAL =================
probs = model.predict_proba(X_test_scaled)
threshold = 0.7

final_preds = []
for p in probs:
    if max(p) < threshold:
        final_preds.append("DEFER")
    else:
        final_preds.append(np.argmax(p))

final_preds = np.array(final_preds, dtype=object)

mask = final_preds != "DEFER"
y_after = y_test.values[mask]
pred_after = final_preds[mask].astype(int)

cm = confusion_matrix(y_after, pred_after)
tn, fp, fn, tp = cm.ravel()

print("\nAFTER DEFERRAL")
print("Deferred:", np.sum(~mask))
print(classification_report(y_after, pred_after))
print("Error rate:", (fp + fn) / len(y_after))
print("False negatives:", fn)