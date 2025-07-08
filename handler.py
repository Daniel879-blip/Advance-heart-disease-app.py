
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def load_data():
    """Load and preprocess the heart disease dataset"""
    df = pd.read_csv("heart.csv")

    # Check if target column has at least two classes
    if 'target' not in df.columns:
        raise ValueError("❌ 'target' column not found in dataset.")

    if len(set(df['target'])) < 2:
        raise ValueError("❌ Dataset must contain at least two classes in target column.")

    X = df.drop('target', axis=1)
    y = df['target']

    return X, y, df

def train_model(X, y, classifier_name='KNN', params={}):
    """Train the selected classifier"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if classifier_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=params.get('K', 5))
    elif classifier_name == 'Logistic Regression':
        model = LogisticRegression(C=params.get('C', 1.0), max_iter=1000)
    elif classifier_name == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=params.get('max_depth', None))
    else:
        raise ValueError("❌ Invalid classifier selected.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    return model, metrics
