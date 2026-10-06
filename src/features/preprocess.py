"""Module de feature engineering et de preparation des donnees."""

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_raw_dataframe(
    df: pd.DataFrame, id_col: str, target_col: str
) -> pd.DataFrame:
    """Supprime l'identifiant technique et convertit la cible en binaire.

    Args:
        df: DataFrame brut d'entree.
        id_col: Nom de la colonne identifiant technique a supprimer.
        target_col: Nom de la colonne cible metier (ex: Churn).

    Returns:
        pd.DataFrame: DataFrame nettoye sans effet de bord.
    """
    data = df.copy()  # Regle de purete : aucune mutation de la variable d'entree
    if id_col in data.columns:
        data = data.drop(columns=[id_col])
    if target_col in data.columns:
        data[target_col] = data[target_col].map({"Yes": 1, "No": 0})
    return data


def encode_features(
    df: pd.DataFrame,
    binary_cols: List[str],
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> pd.DataFrame:
    """Applique l'encodage binaire, One-Hot et la normalisation Min-Max.

    Args:
        df: DataFrame nettoye.
        binary_cols: Liste des colonnes binaires (Yes/No, Male/Female).
        categorical_cols: Liste des variables categorielles nominales.
        numerical_cols: Liste des variables quantitatives continues.

    Returns:
        pd.DataFrame: Jeu de donnees entierement transforme et encode.
    """
    data = df.copy()

    # 1. Encodage binaire deterministe
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map(binary_map).fillna(0)

    # 2. Encodage One-Hot des colonnes categorielles
    existing_cat = [c for c in categorical_cols if c in data.columns]
    if existing_cat:
        data = pd.get_dummies(data, columns=existing_cat, drop_first=True, dtype=int)

    # 3. Imputation et Normalisation Min-Max des variables numeriques
    for col in numerical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
            c_min = data[col].min()
            c_max = data[col].max()
            if c_max > c_min:
                data[col] = (data[col] - c_min) / (c_max - c_min)
            else:
                data[col] = 0.0

    return data


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Decoupe le dataset en ensembles train/test avec stratification.

    Args:
        df: DataFrame pret pour la modelisation.
        target_col: Nom de la variable cible binaire.
        test_size: Proportion de l'echantillon de test.
        random_state: Graine aleatoire garantissant la stricte reproductibilite.

    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )