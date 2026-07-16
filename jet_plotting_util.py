from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


def plot_feature_distributions(df, title, output_dir="processed_data"):
    """
    Plot feature distributions for QCD and TT jets and save the plots.

    Args:
        df (DataFrame): DataFrame containing features and labels
        title (str): Title for the plot
        output_dir (str): Directory to save the plots
    """
    features = [col for col in df.columns if col != "label"]
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data=df, x=feature, hue="label", bins=50, alpha=0.5)
        plt.title(feature)

    plt.suptitle(title)
    plt.tight_layout()

    # Save the plot
    plot_path = Path(output_dir) / f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.close()


def plot_jet_image(image, title="Jet Image"):
    """
    Plot a single jet image
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="viridis")
    plt.colorbar(label="Energy (GeV)")
    plt.title(title)
    plt.xlabel("η")
    plt.ylabel("φ")
    plt.show()


def plot_training_history(history, metrics=["loss", "accuracy"]):
    """
    Plot training history for neural networks
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    if len(metrics) == 1:
        axes = [axes]

    try:
        # this is the syntax for keras
        hist = history.history
    except Exception:
        hist = history
    for ax, metric in zip(axes, metrics):
        ax.plot(hist[metric], label=f"Training {metric}")
        ax.plot(hist[f"val_{metric}"], label=f"Validation {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=["QCD", "TT"]):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# plot roc curve
def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.6f)" % roc_auc)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
