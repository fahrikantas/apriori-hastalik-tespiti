import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports resolve when run as a script
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.decision_tree import train_decision_tree
from src.naive_bayes import train_naive_bayes
from src.random_forest import train_random_forest
from src.logistic_regression import train_logistic_regression
from src.svm import train_svm
from src.xgboost_model import train_xgboost
from src.lightgbm_model import train_lightgbm

trainers = [
    train_decision_tree,
    train_naive_bayes,
    train_random_forest,
    train_logistic_regression,
    train_svm,
    train_xgboost,
    train_lightgbm,
]

for trainer in trainers:
    try:
        res = trainer('Training.csv')
        acc = getattr(res, 'accuracy', None)
        print(f"{trainer.__name__}: accuracy={acc}")
    except Exception as e:
        print(f"{trainer.__name__} failed: {e}")
