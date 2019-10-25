import numpy as np
import pandas as pd
import seaborn as sns
# import copy
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



def plot_roc_curve( y_predict_proba, y_truth):
    y_score = np.array(y_predict_proba)
    if len(y_truth.shape) == 1:
        dummies = pd.get_dummies(y_truth)
        y_dummies = dummies.values
    else:
        y_dummies = y_truth

    y_classes = dummies.columns

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i, class_name in enumerate(y_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_dummies.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    for i, class_name in enumerate(y_classes):
        plt.plot(fpr[i], tpr[i],
                 lw=lw, label='%s (area = %0.2f)' % (class_name, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # threshold for positive class
    ax2 = plt.gca().twinx()
    ax2.plot(fpr[1], thresholds[1], markeredgecolor='r', linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold')
    ax2.set_ylim([thresholds[1][-1], thresholds[1][0]])
    ax2.set_xlim([fpr[1][0], fpr[1][-1]])

    # plt.show()
    return plt.gcf()


def plot_precision_recall_curve(  y_predict_proba, y_truth):
    y_score = np.array(y_predict_proba)
    if len(y_truth.shape) == 1:
        dummies = pd.get_dummies(y_truth)
        y_dummies = dummies.values
    else:
        y_dummies = y_truth

    y_classes = dummies.columns
    for i, class_name in enumerate(y_classes):
        precision, recall, thresholds = precision_recall_curve(y_dummies[:, i], y_score[:, i])

        plt.step(recall, precision,
                 label=class_name,
                 lw=2,
                 where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")

    # ax2 = plt.gca().twinx()
    # ax2.plot(recall[1:], thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold')

    # plt.show()
    return plt.gcf()


def plot_confidence_performance(y_predict, y_predict_proba, y_truth, num_bins=20):
    predicted_probabilities = np.max(y_predict_proba, axis=1)
    is_correct = (y_truth == y_predict)
    ax = sns.regplot(x=predicted_probabilities, y=is_correct, x_bins=num_bins)
    plt.xlabel('Model Confidence')
    plt.ylabel('Average accuracy')
    # plt.show()
    return plt.gcf()



def plot_confusion_matrix(y_true, y_pred, classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if classes is not None:
        # Only use the labels that appear in the data
        classes = [classes[x] for x in unique_labels(y_true, y_pred)]
    else:
        classes = unique_labels(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
