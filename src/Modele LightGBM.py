import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

# =========================================================
# 1️⃣ CHARGEMENT DU DATASET
# =========================================================
print("Chargement du dataset...")

df = pd.read_csv(r"C:\Users\user\Documents\predection heart failure\nouvelle dataset equilibrée.csv")

print("Dataset chargé")
print("Nombre de patients :", len(df))
print(df.head())

# =========================================================
# 2️⃣ PREPARATION DES DONNEES
# =========================================================
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 3️⃣ ENTRAINEMENT XGBOOST
# =========================================================
print("\nEntrainement du modèle XGBoost...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("Modèle entraîné")

# =========================================================
# 4️⃣ EVALUATION
# =========================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

print("\nRESULTATS DU MODELE")
print("---------------------")
print("Accuracy :", round(accuracy*100, 2), "%")
print("F1 Score :", round(f1, 2))
print("ROC AUC :", round(roc, 2))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# =========================================================
# 5️⃣ IMPORTANCE DES VARIABLES
# =========================================================
importance = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nImportance des variables :\n")
print(importance_df)

# =========================================================
# 6️⃣ SAUVEGARDE DU MODELE
# =========================================================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/heart_model_xgb.pkl")

print("\nModele sauvegardé : models/heart_model_xgb.pkl")

# =========================================================
# 7️⃣ FONCTION DE PREDICTION
# =========================================================
def predict_patient(data):
    model = joblib.load("models/heart_model_xgb.pkl")

    patient = pd.DataFrame([data])

    prob = model.predict_proba(patient)[0][1]
    risk = round(prob*100, 2)

    if risk >= 70:
        diag = "RISQUE ELEVE"
    elif risk >= 40:
        diag = "RISQUE MODERE"
    else:
        diag = "FAIBLE RISQUE"

    print("\n---------------------------")
    print("Probabilité de décès :", risk, "%")
    print("Diagnostic :", diag)
    print("---------------------------")

# =========================================================
# 8️⃣ EXEMPLE DE PREDICTION
# =========================================================
patient_test = {
    "age": 30,
    "anaemia": 0,
    "creatinine_phosphokinase": 150,
    "diabetes": 0,
    "ejection_fraction": 70,
    "high_blood_pressure": 1,
    "platelets": 400000,
    "serum_creatinine": 2,
    "serum_sodium": 140,
    "sex": 0,
    "smoking": 0,
    "time": 100
}

predict_patient(patient_test)