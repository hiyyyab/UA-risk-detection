import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("d_dset.csv")

X = data.drop(columns=["Diabetes_binary"])
y = data["Diabetes_binary"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# BEFORE
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nRANDOM FOREST - BEFORE DEFERRAL")
print(classification_report(y_test, y_pred))
print("Error rate:", (fp + fn) / len(y_test))
print("False negatives:", fn)

# DEFERRAL
probs = model.predict_proba(X_test)
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