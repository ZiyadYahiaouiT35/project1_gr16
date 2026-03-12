import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

# =========================================================
# 1️⃣ CHARGEMENT DU DATASET
# =========================================================
print("Chargement du dataset...")

df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")

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
# 3️⃣ ENTRAINEMENT LOGISTIC REGRESSION
# =========================================================
print("\nEntrainement du modèle...")

# On utilise solver 'liblinear' pour les petits datasets
model = LogisticRegression(
    solver='liblinear',
    max_iter=1000,
    random_state=42
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
# 5️⃣ COEFFICIENTS DES VARIABLES (importance)
# =========================================================
coefficients = model.coef_[0]
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nCoefficients des variables :\n")
print(importance_df)

# =========================================================
# 6️⃣ SAUVEGARDE DU MODELE
# =========================================================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/heart_model_logreg.pkl")
print("\nModele sauvegardé : models/heart_model_logreg.pkl")

# =========================================================
# 7️⃣ FONCTION DE PREDICTION
# =========================================================
def predict_patient(data):
    model = joblib.load("models/heart_model_logreg.pkl")
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
    print("⚡ Note : Vérifiez que le F1-score et la ROC-AUC du modèle sont élevés pour confirmer la fiabilité de la prédiction.")

# =========================================================
# 8️⃣ EXEMPLE DE PREDICTION
# =========================================================
patient_test = {
    "age": 65,
    "anaemia": 0,
    "creatinine_phosphokinase": 160,
    "diabetes": 1,
    "ejection_fraction": 20,
    "high_blood_pressure": 0,
    "platelets": 327000,
    "serum_creatinine": 2.7,
    "serum_sodium": 116,
    "sex": 0,
    "smoking": 0,
    "time": 8
}

predict_patient(patient_test)

patient_sain = {
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

predict_patient(patient_sain)