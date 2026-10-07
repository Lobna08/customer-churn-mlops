"""Tests unitaires du module d'ingestion des donnees."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_raw_data


def test_load_raw_data_valid(tmp_path: Path):
    """Verifie le chargement reussi d'un CSV valide via un fichier temporaire."""
    dummy_csv = tmp_path / "valid.csv"
    dummy_csv.write_text("col_a,col_b\n10,20\n30,40", encoding="utf-8")

    df = load_raw_data(dummy_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col_a", "col_b"]


def test_load_raw_data_file_not_found():
    """Verifie qu'un chemin inexistant leve formellement FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("chemin_totalement_inexistant_vers_donnees.csv")


def test_load_raw_data_empty_file(tmp_path: Path):
    """Verifie qu'un fichier vide declenche une ValueError explicite."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises((ValueError, pd.errors.EmptyDataError)):
        load_raw_data(empty_csv)
