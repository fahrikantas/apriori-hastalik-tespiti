import joblib
from pathlib import Path
models = ['decision_tree.pkl','naive_bayes.pkl','random_forest.pkl','logistic_regression.pkl','svm.pkl','xgboost.pkl','lightgbm.pkl']
for m in models:
    p = Path('models')/m
    if not p.exists():
        print(m, 'MISSING')
        continue
    b = joblib.load(p)
    acc = b.get('accuracy')
    meta = b.get('metadata') or {}
    print(m, 'accuracy=', acc, 'metadata_data_fp=', meta.get('data_fingerprint'))
