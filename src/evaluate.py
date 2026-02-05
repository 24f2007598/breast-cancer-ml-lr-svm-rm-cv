from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()

    return acc, cm, cv
