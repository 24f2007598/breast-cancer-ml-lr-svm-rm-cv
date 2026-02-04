from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "SVM": SVC(kernel="rbf"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        )
    }
    return models
