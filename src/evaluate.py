# --- שלב 1: יבוא ספריות ---
# כאן נמצאות פונקציות חישוב מדדים וקרוס-וואלידציה
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_validate


def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics for a multiclass problem."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
    }


def perform_cross_validation(model, X, y, cv):
    """Perform 5-fold cross validation and return metrics summary and fold scores."""
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
    }
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    return {
        'accuracy': cv_results['test_accuracy'],
        'precision': cv_results['test_precision'],
        'recall': cv_results['test_recall'],
        'f1': cv_results['test_f1'],
    }
