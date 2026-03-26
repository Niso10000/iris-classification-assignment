from sklearn.datasets import load_iris
import pandas as pd
import os

# Load iris data directly from sklearn (no internet needed)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({
    0: 'setosa', 1: 'versicolor', 2: 'virginica'
})

# Save to CSV
os.makedirs('data', exist_ok=True)
df.to_csv('data/iris.csv', index=False)
print("iris.csv created successfully with", len(df), "rows")

# --- שלב 1: יבוא ספריות ומודולים ---
# פה אנחנו מייבאים ספריות סטנדרטיות וקומפוננטים של sklearn
import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Ensure src is in path for local imports
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluate import compute_classification_metrics, perform_cross_validation
from visualizations import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance,
    plot_correlation_heatmap,
    plot_decision_boundary,
    plot_cross_val_boxplot,
    plot_learning_curve,
)


# --- שלב 2: טעינת הנתונים וכתיבתם ל־CSV ---
def load_and_save_iris(csv_path):
    """Load iris dataset and save to CSV in consistent format."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if df.shape[0] == 150:
            # Normalize column names if needed
            if 'sepal_length' not in df.columns and 'sepal length (cm)' in df.columns:
                df = df.rename(columns={
                    'sepal length (cm)': 'sepal_length',
                    'sepal width (cm)': 'sepal_width',
                    'petal length (cm)': 'petal_length',
                    'petal width (cm)': 'petal_width',
                    'target': 'target',
                })
            if 'species' not in df.columns and 'species_name' in df.columns:
                df['species'] = df['species_name']
            if 'species' not in df.columns and 'target' in df.columns:
                df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            if 'sepal_length' in df.columns and 'species' in df.columns:
                df = df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]]
                df.to_csv(csv_path, index=False)
                return df
    # if we reach here, re-create from sklearn to guarantee format
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "target",
    })
    df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    df = df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]]
    df.to_csv(csv_path, index=False)
    return df


# --- שלב 3: עיבוד מוקדם של נתונים ---
def prepare_data(df):
    """Prepare train/test splits, scale features, encode labels."""
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y = df["species"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder


# --- שלב 4: יצירת מודלים ---
def get_models():
    """Return dictionary of initialized model estimators."""
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=120, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500, solver='lbfgs', random_state=42),
    }
    return models


# --- שלב 5: אימון, חיזוי והערכה ---
def train_evaluate_and_save(X_train, X_test, y_train, y_test, df):
    """Train all models, evaluate them and save figures."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    models = get_models()
    results = []
    cross_val_per_model = {}

    for name, model in models.items():
        print(f"\n--- Training model: {name} ---")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = compute_classification_metrics(y_test, y_pred)
        metrics['model'] = name

        print(f"Model: {name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision']:.4f}")
        print(f"Recall (macro): {metrics['recall']:.4f}")
        print(f"F1 (macro): {metrics['f1']:.4f}")

        cv_results = perform_cross_validation(model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        cross_val_per_model[name] = cv_results

        cm = compute_confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm=cm, classes=['setosa', 'versicolor', 'virginica'], model_name=name, path=os.path.join(output_dir, f"confusion_matrix_{name}.png"))

        results.append(metrics)

    all_results_df = pd.DataFrame(results)
    plot_model_comparison(all_results_df, path=os.path.join(output_dir, "model_comparison.png"))

    # Feature importance from RandomForest and Logistic Regression
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    best_model = models['RandomForest']
    plot_feature_importance(best_model, feature_names, path=os.path.join(output_dir, "feature_importance.png"))

    plot_correlation_heatmap(df[["sepal_length", "sepal_width", "petal_length", "petal_width"]], path=os.path.join(output_dir, "correlation_heatmap.png"))

    # decision boundary using best model and top2 features
    X_full = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y_full = LabelEncoder().fit_transform(df["species"].values)
    plot_decision_boundary(best_model, X_full, y_full, feature_names, path=os.path.join(output_dir, "decision_boundary.png"))

    plot_cross_val_boxplot(cross_val_per_model, path=os.path.join(output_dir, "cross_validation.png"))

    plot_learning_curve(best_model, X_train, y_train, path=os.path.join(output_dir, "loss_curve.png"))

    all_results_df.to_csv(os.path.join(output_dir, "model_metrics_summary.csv"), index=False)
    print("\nTraining and evaluation complete. Output files are in outputs/ ")

    return all_results_df, cross_val_per_model


def compute_confusion_matrix(y_true, y_pred):
    """Helper to compute confusion matrix directly in classifier."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)


def main():
    """Main entry point for the script."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, 'data', 'iris.csv')

    df = load_and_save_iris(csv_path)
    X_train, X_test, y_train, y_test, encoder = prepare_data(df)

    results_df, cv_results = train_evaluate_and_save(X_train, X_test, y_train, y_test, df)

    print('\nFinal results:')
    print(results_df)


if __name__ == '__main__':
    main()
