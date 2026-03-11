import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ═══════════════════════════════════════════════════
#  ÉTAPE 1 — ENTRAÎNEMENT DU MODÈLE
# ═══════════════════════════════════════════════════

def entrainer_modele():
    """
    Charge le dataset, entraîne le Random Forest,
    affiche les résultats et sauvegarde le modèle.
    """

    # Charger les données
    df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
    print(f"✅ Dataset chargé : {len(df)} patients")
    print(f"   Survivants : {(df['DEATH_EVENT']==0).sum()}")
    print(f"   Décédés    : {(df['DEATH_EVENT']==1).sum()}")

    # Séparer features et cible
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    # Split 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Créer et entraîner le modèle
    modele = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    modele.fit(X_train, y_train)

    # Évaluer le modèle
    y_pred       = modele.predict(X_test)
    y_pred_proba = modele.predict_proba(X_test)[:, 1]

    print(f"\n📊 Résultats du modèle :")
    print(f"   Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"   ROC-AUC   : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"   F1-Score  : {f1_score(y_test, y_pred):.4f}")

    # Sauvegarder le modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(modele, 'models/random_forest.pkl')
    print(f"\n💾 Modèle sauvegardé : models/random_forest.pkl")

    return modele


# ═══════════════════════════════════════════════════
#  ÉTAPE 2 — FONCTION DE PRÉDICTION
# ═══════════════════════════════════════════════════

def prediction(
    age,
    anaemia,
    creatinine_phosphokinase,
    diabetes,
    ejection_fraction,
    high_blood_pressure,
    platelets,
    serum_creatinine,
    serum_sodium,
    sex,
    smoking,
    time
):
    """
    Prédit le risque d'insuffisance cardiaque d'un patient.

    Paramètres :
    ─────────────────────────────────────────────────
    age                      : âge du patient (ex: 65)
    anaemia                  : anémie ? 1=Oui, 0=Non
    creatinine_phosphokinase : taux CPK dans le sang (ex: 582)
    diabetes                 : diabète ? 1=Oui, 0=Non
    ejection_fraction        : % de sang pompé (ex: 38)
    high_blood_pressure      : hypertension ? 1=Oui, 0=Non
    platelets                : plaquettes (ex: 265000)
    serum_creatinine         : créatinine sérique (ex: 1.1)
    serum_sodium             : sodium sérique (ex: 136)
    sex                      : 1=Homme, 0=Femme
    smoking                  : fumeur ? 1=Oui, 0=Non
    time                     : durée de suivi en jours (ex: 60)

    Retourne :
    ─────────────────────────────────────────────────
    pourcentage_risque : float  → ex: 87.5 (%)
    diagnostic         : str   → "RISQUE ÉLEVÉ" ou "FAIBLE RISQUE"
    """

    # Charger le modèle sauvegardé
    modele = joblib.load('models/random_forest.pkl')

    # Construire les données du patient
    patient = pd.DataFrame([{
        'age'                      : age,
        'anaemia'                  : anaemia,
        'creatinine_phosphokinase' : creatinine_phosphokinase,
        'diabetes'                 : diabetes,
        'ejection_fraction'        : ejection_fraction,
        'high_blood_pressure'      : high_blood_pressure,
        'platelets'                : platelets,
        'serum_creatinine'         : serum_creatinine,
        'serum_sodium'             : serum_sodium,
        'sex'                      : sex,
        'smoking'                  : smoking,
        'time'                     : time
    }])

    # Calculer les probabilités
    probabilites       = modele.predict_proba(patient)[0]
    pourcentage_risque = round(probabilites[1] * 100, 2)

    # Déterminer le diagnostic
    if pourcentage_risque >= 70:
        diagnostic = "⚠️  RISQUE ÉLEVÉ"
    elif pourcentage_risque >= 40:
        diagnostic = "⚡ RISQUE MODÉRÉ"
    else:
        diagnostic = "✅ FAIBLE RISQUE"

    # Afficher le résultat
    print(f"\n{'═'*45}")
    print(f"  🫀 RÉSULTAT DE PRÉDICTION")
    print(f"{'═'*45}")
    print(f"  Probabilité de survie       : {round(probabilites[0]*100, 2)}%")
    print(f"  Probabilité d'insuffisance  : {pourcentage_risque}%")
    print(f"  Diagnostic                  : {diagnostic}")
    print(f"{'═'*45}")

    return pourcentage_risque, diagnostic


# ═══════════════════════════════════════════════════
#  ÉTAPE 3 — PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════

if _name_ == "_main_":

    # --- Entraîner le modèle ---
    print("=" * 45)
    print("  ENTRAÎNEMENT DU MODÈLE")
    print("=" * 45)
    entrainer_modele()

    # --- Tester la fonction de prédiction ---
    print("\n\n" + "=" * 45)
    print("  EXEMPLES DE PRÉDICTIONS")
    print("=" * 45)

    # Patient 1 : profil à risque élevé
    print("\n👤 Patient 1 — Profil critique")
    prediction(
        age                      = 65,
        anaemia                  = 0,
        creatinine_phosphokinase = 160,
        diabetes                 = 1,
        ejection_fraction        = 20,
        high_blood_pressure      = 0,
        platelets                = 327000,
        serum_creatinine         = 2.7,
        serum_sodium             = 116,
        sex                      = 0,
        smoking                  = 0,
        time                     = 8
    )

    # Patient 2 : profil sain
    print("\n👤 Patient 2 — Profil sain")
    prediction(
        age                      = 45,
        anaemia                  = 0,
        creatinine_phosphokinase = 582,
        diabetes                 = 0,
        ejection_fraction        = 38,
        high_blood_pressure      = 0,
        platelets                = 265000,
        serum_creatinine         = 1.1,
        serum_sodium             = 136,
        sex                      = 1,
        smoking                  = 0,
        time                     = 60
    )

    # Patient 3 : profil très critique
    print("\n👤 Patient 3 — Profil très critique")
    prediction(
        age                      = 80,
        anaemia                  = 1,
        creatinine_phosphokinase = 123,
        diabetes                 = 0,
        ejection_fraction        = 35,
        high_blood_pressure      = 1,
        platelets                = 388000,
        serum_creatinine         = 9.4,
        serum_sodium             = 133,
        sex                      = 1,
        smoking                  = 1,
        time                     = 10
    )