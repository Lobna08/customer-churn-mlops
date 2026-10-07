 # Customer Churn MLOps

Industrialisation d'un modèle de prédiction du churn pour un opérateur télécom.
Ce dépôt transforme un prototype de notebook en projet Python structuré : code
modulaire, configuration externalisée, tests et contrôles de qualité
automatisés.

Le modèle en lui-même (une régression logistique) reste volontairement simple.
L'accent est mis sur les fondations logicielles, pas sur la performance
prédictive.

## Prérequis

- Python 3.11 (le projet cible `>=3.10` et `<3.13`)
- Poetry pour la gestion des dépendances

## Mise en route

```bash
git clone https://github.com/Lobna08/customer-churn-mlops.git
cd customer-churn-mlops
poetry install
```

Le jeu de données n'est pas versionné : il est ignoré par Git. Placez le fichier
`customer_churn.csv` dans `data/raw/` avant de lancer l'entraînement.

## Entraîner le modèle

```bash
poetry run python src/models/train.py
```

Sortie attendue :

~~~text
ACCURACY  : 0.8050
F1_SCORE  : 0.5412
ROC_AUC   : 0.8099
~~~



Le modèle est sérialisé dans `models/churn_logistic_regression.pkl`. Les chemins
et les hyperparamètres se règlent dans `configs/config.yaml` ; on peut aussi
pointer explicitement un autre fichier :

```bash
poetry run python src/models/train.py --config configs/config.yaml
```

## Tests

```bash
poetry run pytest tests/ -v
```

## Qualité de code

Le formatage (black, isort) et le linting (flake8) sont exécutés
automatiquement à chaque commit via pre-commit. Après un clone, il faut
installer le hook une fois :

```bash
poetry run pre-commit install
```

Pour tout vérifier manuellement :

```bash
poetry run pre-commit run --all-files
```

## Structure

~~~text
customer-churn-mlops/
├── configs/            # Configuration (chemins, hyperparamètres)
├── data/
│   ├── raw/            # Données brutes (non versionnées)
│   └── processed/
├── models/             # Modèles sérialisés (non versionnés)
├── src/
│   ├── data/           # Ingestion
│   ├── features/       # Nettoyage et feature engineering
│   └── models/         # Entraînement et évaluation
└── tests/              # Tests unitaires
~~~


## Limites connues

- La normalisation (min/max) et l'imputation sont calculées sur l'ensemble des
  données avant le découpage train/test. C'est une fuite de données : les
  métriques sont donc légèrement optimistes. Une version propre apprendrait ces
  paramètres sur le jeu d'entraînement uniquement (via un `Pipeline`
  scikit-learn).
- L'écart entre l'accuracy (0.80) et le F1 (0.54) traduit un déséquilibre des
  classes : le modèle détecte imparfaitement les clients qui résilient, qui
  sont pourtant la cible métier.
