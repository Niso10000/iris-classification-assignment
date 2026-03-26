# --- שלב 1: יבוא ספריות ---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import learning_curve


def set_plot_style():
    """Apply consistent plotting style for all charts."""
    sns.set(style='whitegrid')
    plt.rc('font', size=11)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=13)
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)


# --- שלב 2: מטריצת בלבול ---
def plot_confusion_matrix(cm, classes, model_name, path):
    """Plot and save a confusion matrix heatmap."""
    set_plot_style()
    plt.figure(figsize=(10, 7), dpi=150)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- שלב 3: השוואת מודלים ---
def plot_model_comparison(metrics_df, path):
    """Plot bar chart for accuracy/precision/recall/f1 for all models."""
    set_plot_style()
    metrics_long = pd.melt(metrics_df, id_vars=['model'], value_vars=['accuracy', 'precision', 'recall', 'f1'],
                           var_name='metric', value_name='value')

    plt.figure(figsize=(10, 7), dpi=150)
    sns.barplot(data=metrics_long, x='metric', y='value', hue='model', palette='Set2')
    plt.title('Model Comparison: Accuracy, Precision, Recall, F1')
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- שלב 4: חשיבות תכונות ---
def plot_feature_importance(model, feature_names, path):
    """Plot feature importance sorted descending."""
    set_plot_style()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        raise ValueError('Model does not provide feature importance')

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 7), dpi=150)
    sns.barplot(data=importance_df, x='importance', y='feature', palette='husl')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- שלב 5: קורלציה בין תכונות ---
def plot_correlation_heatmap(df_features, path):
    """Plot correlation matrix heatmap for features."""
    set_plot_style()
    corr = df_features.corr()

    plt.figure(figsize=(10, 7), dpi=150)
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- שלב 6: גבול החלטה ---
def plot_decision_boundary(model, X, y, feature_names, path):
    """Plot 2D decision boundary for top 2 features."""
    set_plot_style()

    top_features = [2, 3]
    X_two = X[:, top_features]

    clf = clone(model)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_two_scaled = scaler.fit_transform(X_two)
    clf.fit(X_two_scaled, y)

    x_min = X_two_scaled[:, 0].min() - 0.5
    x_max = X_two_scaled[:, 0].max() + 0.5
    y_min = X_two_scaled[:, 1].min() - 0.5
    y_max = X_two_scaled[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7), dpi=150)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')
    sns.scatterplot(x=X_two_scaled[:, 0], y=X_two_scaled[:, 1], hue=y, palette='Set1', edgecolor='k')

    plt.title('Decision Boundary (petal length + petal width)')
    plt.xlabel(feature_names[top_features[0]])
    plt.ylabel(feature_names[top_features[1]])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- שלב 7: ואלידצית קרוס -- box plot ---
def plot_cross_val_boxplot(cross_val_results, path):
    """Plot box plot of cross validation accuracy for each model."""
    set_plot_style()
    rows = []

    for model_name, model_scores in cross_val_results.items():
        for value in model_scores.get('accuracy', []):
            rows.append({'model': model_name, 'accuracy': value})

    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 7), dpi=150)
    sns.boxplot(data=df, x='model', y='accuracy', palette='Set2')
    plt.title('Cross Validation Accuracy Box Plot')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- שלב 8: עקומת למידה ---
def plot_learning_curve(estimator, X, y, path):
    """Plot learning curve for the selected estimator."""
    set_plot_style()

    cv = 5
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring='accuracy',
        shuffle=True,
        random_state=42,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 7), dpi=150)
    plt.plot(train_sizes, train_mean, 'o-', color='b', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='b')
    plt.plot(train_sizes, test_mean, 'o-', color='r', label='Validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='r')

    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
