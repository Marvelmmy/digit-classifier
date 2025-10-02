from sklearn.metrics import accuracy_score, classification_report
import pickle

def evaluate_model():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuuracy, report