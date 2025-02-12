from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def fit_predict_accuracy(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=["Control", "ProbableAD"])
    cm_display.plot()
    plt.show()
    print(f"CNF Matrix: \n {cnf_matrix}")
    accuracy = accuracy_score(y_pred, y_test)
    print(f"Accuracy: {accuracy}")
    f1 = f1_score(y_pred, y_test, pos_label='Control')
    print(f"F1 Score Control: {f1}")
    f1 = f1_score(y_pred, y_test, pos_label='ProbableAD')
    print(f"F1 Score ProbableAD: {f1}")