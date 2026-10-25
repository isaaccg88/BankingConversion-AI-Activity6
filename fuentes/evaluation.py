"""Métricas y validación reproducible para clasificación binaria."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    average_precision_score, confusion_matrix, f1_score, fbeta_score,
    precision_score, recall_score, roc_auc_score)

def metrics(y, proba, threshold=0.5):
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y,pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "precision": float(precision_score(y,pred,zero_division=0)),
        "recall": float(recall_score(y,pred,zero_division=0)),
        "specificity": float(tn / (tn+fp)),
        "f1": float(f1_score(y,pred,zero_division=0)),
        "f2": float(fbeta_score(y,pred,beta=2,zero_division=0)),
        "roc_auc": float(roc_auc_score(y,proba)),
        "average_precision": float(average_precision_score(y,proba)),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
    }

def choose_threshold(y, proba, min_precision=0.20):
    results = [metrics(y,proba,float(t)) for t in np.round(np.arange(.05,.851,.01),2)]
    eligible = [x for x in results if x["precision"] >= min_precision]
    pool = eligible if eligible else results
    chosen = max(pool, key=lambda x:(x["f2"],x["recall"],x["precision"]))
    return chosen,results,bool(eligible)
