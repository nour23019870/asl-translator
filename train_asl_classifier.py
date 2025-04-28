# TRAIN ASL LETTER CLASSIFIER FROM LANDMARK DATA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Step 1: Load your dataset
df = pd.read_csv('dataset.csv')  # Make sure dataset.csv is in the same directory

# Step 2: Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 4: Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 6: Save the model
joblib.dump(clf, 'asl_classifier.joblib')
