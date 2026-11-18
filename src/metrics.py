"""Métricas binarias y selección de umbral (siempre sobre VALIDACIÓN, nunca sobre prueba)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def binary_metrics(y, s, thr):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(s) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")      # recall de la clase maligna
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    if np.isnan(prec) or np.isnan(sens):
        f1 = float("nan")
    else:
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    both = len(np.unique(y)) == 2
    return {
        "umbral": float(thr),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "sensibilidad_maligna": sens,
        "especificidad": spec,
        "precision": prec,
        "f1": f1,
        "accuracy": (tp + tn) / len(y),
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
        "n": int(len(y)),
    }


def select_threshold(y, s, strategy="youden", target_sens=0.90):
    """
    youden: maximiza sensibilidad + especificidad - 1
    sens:   el umbral MÁS ALTO que alcanza al menos `target_sens` de sensibilidad
            (prioriza no dejar pasar malignas, a costa de más falsos positivos)
    """
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.clip(thr, 0.0, 1.0)  # roc_curve pone +inf en la primera posición
    if strategy == "youden":
        i = int(np.argmax(tpr - fpr))
    elif strategy == "sens":
        ok = np.where(tpr >= target_sens)[0]
        i = int(ok[0]) if len(ok) else len(thr) - 1
    else:
        raise ValueError(strategy)
    return float(thr[i])


def threshold_table(y, s, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    return [binary_metrics(y, s, t) for t in thresholds]
