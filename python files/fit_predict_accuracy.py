from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def fit_predict_accuracy(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=["Control", "ProbableAD"])
    cm_display.plot()
    print(f"CNF Matrix: \n {cnf_matrix}")
    accuracy = accuracy_score(y_pred, y_test)
    print(f"Accuracy: {accuracy}")
    f1 = f1_score(y_pred, y_test, pos_label='Control')
    print(f"F1 Score Control: {f1}")
    f1 = f1_score(y_pred, y_test, pos_label='ProbableAD')
    print(f"F1 Score ProbableAD: {f1}")

    import numpy as np
    from sklearn.metrics import roc_curve, auc

    # Example: True labels (ground truth) and predicted probabilities

    y_test_binary = np.where(y_test == "ProbableAD", 1, 0)
    y_pred_binary = np.where(y_pred == "ProbableAD", 1, 0)
    print(y_pred_binary)
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
    roc_auc = auc(fpr, tpr)


    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random guess)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

