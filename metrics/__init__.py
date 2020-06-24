from sklearn.metrics import classification_report, confusion_matrix

def print_cm(y_true, y_pred):
    pred_false, pred_true = confusion_matrix(y_true, y_pred)
    tn, fn = pred_false
    fp, tp = pred_true
    print(
        f'TN\tFN\tFP\tTP\n{tn}\t{fn}\t{fp}\t{tp}'
    )


def print_cr(y_true, y_pred):
    print(classification_report(y_true, y_pred))
