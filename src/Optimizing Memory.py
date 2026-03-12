# --- ÉTAPE 2 : LA FONCTION D'OPTIMISATION ---
import pandas as pd

def optimize_memory(df):
    """
    Réduit la taille des données en ajustant les types numériques.
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimisation pour les nombres entiers
        if str(col_type).startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Optimisation pour les nombres décimaux
        elif str(col_type).startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast='float')
            
    return df