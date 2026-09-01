from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.preprocess import preprocess_training_data
from src.split import split_train_test
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

pre = preprocess_training_data('Training.csv')
frame = pre.frame
features = frame[pre.symptom_columns]
target = frame['prognosis']

le = LabelEncoder()
y = le.fit_transform(target)

X_train, X_test, y_train, y_test = split_train_test(features, y, test_size=0.2)
print('train size', len(X_train), 'test size', len(X_test))
print('unique test labels', set(y_test))
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print('accuracy', accuracy_score(y_test, preds))
print('y_test[:20]', y_test[:20])
print('preds[:20]', preds[:20])
print('classes', le.classes_)
print('unique y_train', sorted(set(y_train)))
print('unique preds', sorted(set(preds)))
