from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

def run_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return evaluate_model(y_test, y_pred, y_prob)

def evaluate_model(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
}

    if y_prob is not None:
        metrics["AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    else:
        metrics["AUC"] = "NA"

    return metrics, confusion_matrix(y_true, y_pred)
