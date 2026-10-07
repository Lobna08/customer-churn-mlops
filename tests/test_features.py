"""Tests unitaires du module de feature engineering."""

import pandas as pd

from src.features.preprocess import clean_raw_dataframe, encode_features, split_data


def test_clean_raw_dataframe():
    """Vérifie la suppression de la clé technique et la binarisation de la cible."""
    df = pd.DataFrame(
        {
            "customerID": ["001", "002"],
            "gender": ["Male", "Female"],
            "Churn": ["Yes", "No"],
        }
    )

    cleaned = clean_raw_dataframe(
        df,
        id_col="customerID",
        target_col="Churn",
    )

    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].tolist() == [1, 0]


def test_encode_features_bounds():
    """Vérifie que la normalisation produit des valeurs dans [0, 1]."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "MonthlyCharges": [20.0, 70.0, 120.0],
        }
    )

    encoded = encode_features(
        df,
        binary_cols=["gender"],
        categorical_cols=["Contract"],
        numerical_cols=["MonthlyCharges"],
    )

    assert encoded["MonthlyCharges"].min() >= 0.0
    assert encoded["MonthlyCharges"].max() <= 1.0
    assert not encoded.isnull().values.any()


def test_split_data_proportions():
    """Vérifie le respect du ratio de test et de la stratification."""
    df = pd.DataFrame(
        {
            "feat_1": range(100),
            "Churn": [0, 1] * 50,
        }
    )

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col="Churn",
        test_size=0.2,
        random_state=42,
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert y_train.mean() == y_test.mean() == 0.5
