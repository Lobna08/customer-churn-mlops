"""Tests unitaires du modèle et de son évaluation."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.train import evaluate_model


def test_evaluate_model_performance():
    """Vérifie le calcul conforme des métriques et la supériorité au hasard."""
    X = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 8.0, 9.0],
            "x2": [0.5, 0.2, 0.9, 0.8],
        }
    )

    y = pd.Series([0, 0, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics

    assert metrics["accuracy"] >= 0.5
    assert metrics["roc_auc"] >= 0.5
