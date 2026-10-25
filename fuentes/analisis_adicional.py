"""Auditoría incremental. No altera el experimento original.

- Baseline all-positive, precision@20% y lift a presupuesto fijo.
- Tercer clasificador HistGradientBoosting (selección SOLO en validación).
- Desglose descriptivo por edad.
- Test ya conocido en Actividad 6 => resultados de tercer modelo exploratorios.
"""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .datos import load_data, make_splits, NUMERIC, CATEGORICAL, FEATURES, OUT, SEED
from .evaluation import metrics

FIG = OUT / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
NAME="HistGradientBoosting"

def budget_metrics(y, p, share=0.20):
    """Top k por probabilidad. Lift referenciado a prevalencia del mismo test."""
    y=np.asarray(y, dtype=int)
    p=np.asarray(p, dtype=float)
    k=max(1,int(np.floor(share*len(y))))
    selected=np.argsort(-p,kind="stable")[:k]
    hits=int(y[selected].sum())
    prevalence=float(y.mean())
    rate=hits/k
    return {
        "seleccionados":k,"positivos_seleccionados":hits,
        "precision_at_20":float(rate),
        "recall_at_20":float(hits/y.sum()),
        "lift_at_20":float(rate/prevalence),
        "prevalencia_test":prevalence,
        "advertencia":"Respuesta historica entre top-scored; no lift causal."
    }

def select_threshold_by_validation(y, p, objective="f1", min_prec=0.20):
    rows=[]
    for threshold in np.round(np.arange(.05,.851,.01),2):
        v=metrics(y,p,float(threshold))
        rows.append(v)
    feasible=[r for r in rows if r["precision"] >= min_prec]
    pool=feasible or rows
    champion=max(pool,key=lambda r:(r[objective],r["recall"],r["precision"]))
    return champion,rows,bool(feasible)

def main():
    df=load_data()
    spl=make_splits(df)
    X=df[FEATURES]
    y=(df.y=="yes").astype(int).to_numpy()
    x_train,x_val,x_test=(X.iloc[spl[z]] for z in ("train","validation","test"))
    y_train,y_val,y_test=(y[spl[z]] for z in ("train","validation","test"))
    original=pd.read_csv(OUT/"comparacion_modelos.csv")
    existing=pd.read_csv(OUT/"predicciones_test.csv")
    assert len(existing)==len(y_test) and np.array_equal(existing.y.to_numpy(),y_test)
    assert set(original.Modelo)=={"LogisticRegression","RandomForest"}
    assert existing["indice_original"].tolist()==spl["test"].tolist()

    # Fits only on TRAIN; all imputations / OHE / scalers fit inside 5-fold CV.
    preprocess=ColumnTransformer([
        ("num",Pipeline([("fill",SimpleImputer(strategy="median")),
                         ("scale",StandardScaler())]),NUMERIC),
        ("cat",Pipeline([("fill",SimpleImputer(strategy="most_frequent")),
                         ("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),CATEGORICAL)
    ],sparse_threshold=0.0)
    pipe=Pipeline([
        ("pre",preprocess),
        ("model",HistGradientBoostingClassifier(random_state=SEED,
                                 early_stopping=False,max_iter=110))
    ])
    grid={
        "model__max_leaf_nodes":[15,31],
        "model__l2_regularization":[0.0,1.0]
    }
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    search=GridSearchCV(pipe,grid,cv=cv,scoring="average_precision",
                        n_jobs=2,refit=True,return_train_score=False,
                        error_score="raise")
    t0=perf_counter()
    search.fit(x_train,y_train)
    elapsed=perf_counter()-t0
    pv=search.predict_proba(x_val)[:,1]
    selected_f1,grid_f1,eligible_f1=select_threshold_by_validation(y_val,pv,"f1")
    selected_f2,grid_f2,eligible_f2=select_threshold_by_validation(y_val,pv,"f2")
    # Use validation-F2 threshold to compare consistently to original models.
    threshold=selected_f2["threshold"]
    t0=perf_counter()
    ptest=search.predict_proba(x_test)[:,1]
    latency=(perf_counter()-t0)*1000/len(x_test)
    mtest=metrics(y_test,ptest,threshold)
    m_f1=metrics(y_test,ptest,selected_f1["threshold"])
    new_row={
        "Modelo":NAME,
        "cv_average_precision":float(search.best_score_),
        "validation_f2":selected_f2["f2"],
        "validation_precision":selected_f2["precision"],
        "validation_recall":selected_f2["recall"],
        "validation_threshold":threshold,
        "precision_constraint_feasible":eligible_f2,
        "fit_seconds":elapsed,"test_ms_per_row":latency,
        "best_params_json":json.dumps(search.best_params_),
        **{"test_"+k:v for k,v in mtest.items()}
    }
    variants={}
    variants["LogisticRegression"]=existing["LogisticRegression_prob"].to_numpy()
    variants["RandomForest"]=existing["RandomForest_prob"].to_numpy()
    variants[NAME]=ptest

    budget=[{"Modelo":n,**budget_metrics(y_test,p)} for n,p in variants.items()]
    allpositive=metrics(y_test,np.ones(len(y_test)),threshold=.5)
    baseline={
        "tipo":"always_positive",
        "descripcion":"Etiqueta todos como conversión; referencia útil de F1 con desbalance.",
        **allpositive
    }

    # Cohort audit: descriptive only. Do not select a model/threshold with test.
    age=df.iloc[spl["test"]]["age"].to_numpy()
    age_bins=np.where(age<=30,"hasta_30",np.where(age<=60,"31_a_60","mayor_60"))
    groups=[]
    for model,prob in variants.items():
        if model==NAME: th=threshold
        else: th=float(original.set_index("Modelo").loc[model,"validation_threshold"])
        for group in ["hasta_30","31_a_60","mayor_60"]:
            idx=np.where(age_bins==group)[0]
            correct=y_test[idx]
            prediction=(prob[idx]>=th).astype(int)
            positives=int(correct.sum())
            groups.append({
                "Modelo":model,"grupo_edad":group,
                "n":len(idx),"positivos":positives,
                "precision":float(precision_score(correct,prediction,zero_division=0)),
                "recall":float(recall_score(correct,prediction,zero_division=0)),
                "f1":float(f1_score(correct,prediction,zero_division=0))
            })

    path=OUT/"analisis_adicional"
    path.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([new_row]).to_csv(path/"tercer_modelo_metrics.csv",index=False)
    pd.DataFrame(search.cv_results_).to_csv(path/"tercer_modelo_gridsearch.csv",index=False)
    pd.DataFrame(grid_f1).to_csv(path/"umbrales_f1_validacion.csv",index=False)
    pd.DataFrame(grid_f2).to_csv(path/"umbrales_f2_validacion.csv",index=False)
    pd.DataFrame(budget).to_csv(path/"metricas_presupuesto20.csv",index=False)
    pd.DataFrame(groups).to_csv(path/"evaluacion_grupos_edad.csv",index=False)
    pd.DataFrame({"indice_original":spl["test"],"y":y_test,
        "prob_HistGradientBoosting":ptest,
        "pred_HistGradientBoosting":(ptest>=threshold).astype(int)
    }).to_csv(path/"predicciones_tercer_modelo.csv",index=False)
    (path/"baseline_always_positive.json").write_text(
        json.dumps(baseline,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    report={
        "naturaleza":"Análisis EXPLORATORIO POSTERIOR a primera revisión del test",
        "test_previamente_inspeccionado":True,
        "prohibido_afirmar_confirmacion_independiente":True,
        "nuevo_modelo":NAME,
        "hiperparametros_mejores_cv":search.best_params_,
        "average_precision_cv":float(search.best_score_),
        "f1_max_validacion":{"umbral":selected_f1["threshold"],
                              "f1":selected_f1["f1"],
                              "precision":selected_f1["precision"],
                              "recall":selected_f1["recall"],
                              "test_f1_exploratorio":m_f1["f1"]},
        "f2_max_validacion":{"umbral":threshold,"f2":selected_f2["f2"]},
        "test_f2_threshold_exploratorio":mtest,
        "presupuesto_contacto_porcentaje":20,
        "baseline_todos_positivos_f1":allpositive["f1"],
        "metricas_presupuesto":budget,
        "limitaciones":[
            "Test no es nuevo ni ciego: ya fue evaluado y examinado en entrega principal.",
            "No comparar causalmente cohortes de edad ni estrategias de llamada.",
            "Los resultados son exploratorios; la seleccion principal se congeló antes de esta extensión.",
            "Recall de grupos pequeños puede ser inestable; no hacer decisiones crediticias."
        ]
    }
    (path/"resumen_adicional.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    money=[5,10,20]
    scenario=[]
    for model,d in zip(variants,budget):
        for benefit in money:
            scenario.append({
                "Modelo":model,"beneficio_hipotetico_por_respuesta":benefit,
                "costo_hipotetico_por_contacto":1,
                "contactos":d["seleccionados"],
                "respuestas_historicas":d["positivos_seleccionados"],
                "utilidad_hipotetica":benefit*d["positivos_seleccionados"]-d["seleccionados"],
                "advertencia":"No son pesos ni ganancias reales; sin causalidad."
            })
    pd.DataFrame(scenario).to_csv(path/"sensibilidad_economica_hipotetica.csv",index=False)

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    dfb=pd.DataFrame(budget)
    axes[0].bar(dfb.Modelo,dfb.precision_at_20)
    axes[0].axhline(float(y_test.mean()),linestyle="--",label="Prevalencia test")
    axes[0].set_title("Precisión histórica con 20% de presupuesto")
    axes[0].set_ylabel("Precision@20%")
    axes[0].legend()
    axes[1].bar(dfb.Modelo,dfb.lift_at_20)
    axes[1].axhline(1,linestyle="--")
    axes[1].set_title("Lift en top-20% vs tasa general")
    axes[1].set_ylabel("Lift relativo; NO causal")
    for ax in axes: ax.tick_params(axis="x",rotation=12)
    fig.tight_layout()
    fig.savefig(FIG/"07_lift_presupuesto20.png",dpi=160)
    plt.close(fig)
    print("ANÁLISIS COMPLEMENTARIO COMPLETO")
    print("Todas las cifras sobre test son EXPLORATORIAS; no reemplazan modelo original.")
    print("HGB val F2",selected_f2["f2"],"test F1",mtest["f1"])
    print(pd.DataFrame(budget).to_string(index=False))
    return report

if __name__=="__main__":
    main()
