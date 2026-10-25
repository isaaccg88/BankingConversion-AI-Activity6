"""Actividad 6: ejecutar evaluación, contraste estadístico y simulación A/B.

python -m fuentes.train
Genera resultados reales. No afirma causalidad ni prueba A/B en producción.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, f1_score, precision_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datos import CATEGORICAL, FEATURES, NUMERIC, OUT, ROOT, SEED, load_data, make_splits
from .evaluation import choose_threshold, metrics

FIG = OUT/"figuras"
FIG.mkdir(parents=True,exist_ok=True)
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,
                  default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+"\n",
                  encoding="utf-8")

def pipe(name):
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),
                          ("scale",StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL),
    ])
    if name=="LogisticRegression":
        model=LogisticRegression(max_iter=600,solver="lbfgs",random_state=SEED)
        grid={"model__C":[0.1,1.0],"model__class_weight":[None,"balanced"]}
    elif name=="RandomForest":
        model=RandomForestClassifier(n_estimators=150,random_state=SEED,
                        n_jobs=1,min_samples_leaf=2)
        grid={"model__max_depth":[10,None],"model__min_samples_leaf":[2,8]}
    else:
        raise ValueError(name)
    return Pipeline([("pre",pre),("model",model)]),grid

def bootstrap_paired(y, pa, pb, ta, tb, n=1000):
    rng=np.random.default_rng(SEED)
    nobs=len(y)
    observed=f1_score(y,pb>=tb,zero_division=0)-f1_score(y,pa>=ta,zero_division=0)
    deltas=[]
    for _ in range(n):
        idx=rng.integers(nobs,size=nobs)
        deltas.append(f1_score(y[idx],pb[idx]>=tb,zero_division=0)-
                      f1_score(y[idx],pa[idx]>=ta,zero_division=0))
    ci=np.quantile(deltas,[.025,.975])
    return {"comparacion":"RandomForest - LogisticRegression",
            "delta_f1":float(observed),
            "ic95_inferior":float(ci[0]),
            "ic95_superior":float(ci[1]),
            "repeticiones":n,"pareado":True}

def mcnemar(y, pa, pb, ta, tb):
    ea=(pa>=ta)!=y
    eb=(pb>=tb)!=y
    b=int((~ea & eb).sum())   # A correcto, B equivocado
    c=int((ea & ~eb).sum())   # A equivocado, B correcto
    return {"a_correcto_b_error":b,"a_error_b_correcto":c,
            "discordantes":b+c,
            "p_exacto_bilateral":float(binomtest(min(b,c),b+c,p=.5).pvalue)
                                    if b+c else 1.0}

def offline_ab(y, pscore, budget=.2):
    """Simulación: grupos disjuntos, selección aleatoria vs ranking modelo.

    Cada resultado histórico es OBSERVACIONAL; p-value exploratorio de las
    tasas de respuestas históricas no mide incremento causal del contacto.
    """
    rng=np.random.default_rng(SEED+6)
    order=rng.permutation(len(y))
    half=len(y)//2
    a,b=order[:half],order[half:]
    ka=max(1,int(len(a)*budget))
    kb=max(1,int(len(b)*budget))
    idx_a=rng.choice(a,size=ka,replace=False)
    idx_b=b[np.argsort(-pscore[b],kind="stable")[:kb]]
    ca=int(y[idx_a].sum())
    cb=int(y[idx_b].sum())
    ra,rb=ca/ka,cb/kb
    # IC Wald y prueba z bilaterales descriptivas, no inferencia causal.
    se=math.sqrt(ra*(1-ra)/ka+rb*(1-rb)/kb)
    lower=rb-ra-1.96*se; upper=rb-ra+1.96*se
    pooled=(ca+cb)/(ka+kb)
    sep=math.sqrt(pooled*(1-pooled)*(1/ka+1/kb))
    p=float(2*norm.sf(abs((rb-ra)/sep))) if sep else 1.
    return ({
        "tipo":"simulacion_AB_offline_no_causal",
        "grupo_A":"seleccion_aleatoria","grupo_B":"ranking_RandomForest",
        "n_grupo_A":len(a),"n_grupo_B":len(b),
        "contactados_A":ka,"contactados_B":kb,
        "conversiones_historicas_A":ca,"conversiones_historicas_B":cb,
        "tasa_A":float(ra),"tasa_B":float(rb),
        "diferencia_tasas":float(rb-ra),"ic95_inf":float(lower),
        "ic95_sup":float(upper),"p_z_exploratorio":p,
        "semilla":SEED+6,"presupuesto_relativo":budget,
        "advertencia":"NO experimento real; observaciones históricas, "
            "selección condicionada por score y ausencia de contrafactual "
            "sin contacto. No estimar uplift causal."
    },[int(x) for x in idx_a],[int(x) for x in idx_b])

def save_plots(y,probs,thresholds,cv,audit,ab):
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for n,p in probs.items():
        RocCurveDisplay.from_predictions(y,p,ax=axs[0],name=n)
        PrecisionRecallDisplay.from_predictions(y,p,ax=axs[1],name=n)
    axs[0].set_title("Figura 1. ROC por modelo (test)")
    axs[1].set_title("Figura 2. Precision-Recall (test)")
    fig.tight_layout()
    fig.savefig(FIG/"01_02_curvas_roc_pr.png",dpi=170)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(n,p) in zip(axes,probs.items()):
        ConfusionMatrixDisplay.from_predictions(y,p>=thresholds[n],
                   ax=ax,display_labels=["No","Sí"],cmap="Blues",colorbar=False)
        ax.set_title(f"{n} · umbral {thresholds[n]:.2f}")
    fig.tight_layout()
    fig.savefig(FIG/"03_matrices_confusion.png",dpi=170)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4))
    for n in ["LogisticRegression","RandomForest"]:
        t=cv.loc[cv["modelo"]==n,"mean_test_score"].to_numpy()
        ax.scatter(np.arange(len(t)),t,label=n,s=60)
    ax.set_ylabel("Average Precision (CV)")
    ax.set_xlabel("Configuración evaluada")
    ax.set_title("Figura 4. Búsqueda hiperparamétrica 5-fold")
    ax.legend()
    fig.tight_layout();fig.savefig(FIG/"04_cv.png",dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4))
    labels=["Política A: aleatoria","Política B: ranking RF"]
    vals=[ab["tasa_A"],ab["tasa_B"]]
    ax.bar(labels,vals)
    ax.set_ylim(0,max(vals)*1.2 if max(vals) else 1)
    ax.set_ylabel("Conversión HISTÓRICA en seleccionados")
    ax.set_title("Figura 5. Simulación A/B offline (NO efecto causal)")
    fig.tight_layout();fig.savefig(FIG/"05_ab_offline.png",dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4))
    names=list(audit.keys());vals=[audit[n]["f1"] for n in names]
    ax.bar(names,vals);ax.set_ylim(0,1)
    ax.set_ylabel("F1 en test")
    ax.set_title("Figura 6. F1 en test a umbral validado")
    fig.tight_layout();fig.savefig(FIG/"06_f1_test.png",dpi=170);plt.close(fig)

    for n,p in probs.items():
        t=np.linspace(.05,.85,81)
        from sklearn.metrics import fbeta_score
        # Únicamente una ilustración sobre test: no seleccionar umbral aquí.
        # No exportar curva de ajuste basada en test.
    return True

def run():
    start=time.perf_counter()
    d=load_data()
    parts=make_splits(d)
    X=d[FEATURES]
    y=(d.y=="yes").astype(int).to_numpy()
    xtr,xval,xtest=(X.iloc[parts[z]] for z in ("train","validation","test"))
    ytr,yval,ytest=(y[parts[z]] for z in ("train","validation","test"))
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    gsframes=[];results=[];scores={};thresholds={}
    validation={}
    trained={}
    for name in ["LogisticRegression","RandomForest"]:
        model,grid=pipe(name)
        gs=GridSearchCV(model,grid,scoring="average_precision",cv=cv,
             n_jobs=2,refit=True,return_train_score=True,error_score="raise")
        tic=time.perf_counter()
        gs.fit(xtr,ytr)
        fit_seconds=time.perf_counter()-tic
        cvf=pd.DataFrame(gs.cv_results_)
        cvf["modelo"]=name
        gsframes.append(cvf)
        trained[name]=gs.best_estimator_
        pval=gs.predict_proba(xval)[:,1]
        selected,curve,eligible=choose_threshold(yval,pval)
        validation[name]=selected
        thresholds[name]=selected["threshold"]
        pd.DataFrame(curve).assign(modelo=name).to_csv(
            OUT/f"umbrales_{name}.csv",index=False)
        test_start=time.perf_counter()
        p=gs.predict_proba(xtest)[:,1]
        infer_ms=(time.perf_counter()-test_start)*1000/len(ytest)
        scores[name]=p
        te=metrics(ytest,p,thresholds[name])
        record={
            "Modelo":name,"cv_average_precision":float(gs.best_score_),
            "validation_f2":selected["f2"],
            "validation_precision":selected["precision"],
            "validation_recall":selected["recall"],
            "validation_threshold":thresholds[name],
            "precision_constraint_feasible":eligible,
            "fit_seconds":fit_seconds,"test_ms_per_row":infer_ms,
            "best_params_json":json.dumps(gs.best_params_),
            **{"test_"+k:v for k,v in te.items()},
        }
        results.append(record)
        print(name,"AP(CV)",round(gs.best_score_,4),
              "val F2",round(selected["f2"],4),"test F1",round(te["f1"],4))
    # Selección previa a test: score CV y F2 validación; test sólo para reporte
    rank=sorted(results,key=lambda r:(r["validation_f2"],
                        r["cv_average_precision"]),reverse=True)
    chosen=rank[0]["Modelo"]
    # A/B usa RF como política B preespecificada sin importar qué modelo
    # resulte ganador, para evitar reoptimización en test.
    base=DummyClassifier(strategy="prior")
    base.fit(xtr,ytr)
    dummy_p=base.predict_proba(xtest)[:,1]
    dummy=metrics(ytest,dummy_p,0.5)
    analysis={n:metrics(ytest,scores[n],thresholds[n]) for n in scores}
    paired=bootstrap_paired(ytest,scores["LogisticRegression"],
          scores["RandomForest"],thresholds["LogisticRegression"],
          thresholds["RandomForest"])
    mc=mcnemar(ytest,scores["LogisticRegression"],scores["RandomForest"],
          thresholds["LogisticRegression"],thresholds["RandomForest"])
    ab,ia,ib=offline_ab(ytest,scores["RandomForest"])
    cvdf=pd.concat(gsframes,ignore_index=True)
    summary=pd.DataFrame(results)
    summary.to_csv(OUT/"comparacion_modelos.csv",index=False)
    cvdf.to_csv(OUT/"gridsearch_cv.csv",index=False)
    pred=pd.DataFrame({"indice_original":parts["test"],"y":ytest})
    for name in scores:
        pred[f"{name}_prob"]=scores[name]
        pred[f"{name}_pred"]=(scores[name]>=thresholds[name]).astype(int)
    pred.to_csv(OUT/"predicciones_test.csv",index=False)
    pd.DataFrame([paired]).to_csv(OUT/"bootstrap_pareado.csv",index=False)
    dump("mcnemar.json",mc)
    dump("simulacion_ab.json",ab)
    dump("baseline_dummy.json",dummy)
    dump("seleccion_congelada.json",{
        "modelo_seleccionado_en_validacion":chosen,
        "umbral":thresholds[chosen],
        "criterio":"F2 en validación, desempate AP CV",
        "umbrales_por_modelo":thresholds,
        "semilla":SEED,"test_no_usado_para_seleccion":True
    })
    save_plots(ytest,scores,thresholds,cvdf,analysis,ab)
    dump("resumen.json",{"filas":len(d),"positivos":int(y.sum()),
         "splits":{k:len(v) for k,v in parts.items()},
         "seleccion":chosen,"validacion":validation,
         "bootstrap":paired,"mcnemar":mc,"ab":ab,"baseline":dummy,
         "duracion_segundos":time.perf_counter()-start,
         "limitaciones":"Observacional, distribución histórica, A/B no causal"})
    print("RESULTADOS REALES",summary[["Modelo","test_f1","test_recall",
                          "test_roc_auc","test_average_precision"]].round(4))
    print("A/B offline:",ab["tasa_A"],ab["tasa_B"],
          "no se interpreta causalmente.")
    return summary

if __name__=="__main__":
    run()
