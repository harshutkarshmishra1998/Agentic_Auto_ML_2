import numpy as np


def compute_classification(cm):
    cm = np.array(cm)
    tp = np.diag(cm)

    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    accuracy = tp.sum() / cm.sum()

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


def derive_metrics(training_result):
    if "classification" in training_result:
        block = training_result["classification"]
        cm = block.get("confusion_matrix")
        if cm is not None:
            return compute_classification(cm)

    return {}