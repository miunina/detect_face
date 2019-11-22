import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_roc_curve(fpr, tpr,name="train", path = "temp/"):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(path+"roc_curve.png")


def plot_confusion_matrix(confusion_matrix, n,m, figsize = (10,7), fontsize=14,name="train", path = "temp/"):
    df_cm = pd.DataFrame(
        confusion_matrix, index=n, columns=m,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path+"confusion_matrix.png")


















