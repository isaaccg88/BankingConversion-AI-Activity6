"""Bank Marketing UCI 222: adquisición verificable y particiones congeladas.

Se utiliza bank-full.csv (45,211 filas); solo variables plausibles antes del
contacto. La ausencia de IDs de cliente impide verificar dependencia inter-cliente.
"""
from __future__ import annotations
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datos" / "bank-full.csv"
SPLITS = ROOT / "datos" / "splits.csv"
OUT = ROOT / "resultados"
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
SEED = 42
NUMERIC = ["age", "balance", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default",
               "housing", "loan", "poutcome"]
FEATURES = NUMERIC + CATEGORICAL
EXCLUDED = ["duration", "contact", "day", "month", "campaign"]


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_data() -> pd.DataFrame:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        with urllib.request.urlopen(UCI_URL, timeout=90) as response:
            outer = response.read()
        with zipfile.ZipFile(io.BytesIO(outer)) as z:
            with zipfile.ZipFile(io.BytesIO(z.read("bank.zip"))) as inner:
                raw = inner.read("bank-full.csv")
        RAW.write_bytes(raw)
    df = pd.read_csv(RAW, sep=";", keep_default_na=False)
    if df.shape != (45211, 17):
        raise ValueError(f"El dataset no coincide con bank-full.csv UCI: {df.shape}")
    if df["y"].isin(["yes", "no"]).sum() != len(df):
        raise ValueError("Etiqueta y inválida")
    if df.isna().any().any() or df[FEATURES].isna().any().any():
        raise ValueError("Valores nulos no previstos")
    if len(FEATURES) != 11 or "duration" in FEATURES:
        raise ValueError("Esquema predictivo incorrecto")
    meta = {
        "dataset": "UCI Bank Marketing ID 222; bank-full.csv (versión histórica)",
        "doi": "10.24432/C5K306", "licencia": "CC BY 4.0",
        "source_url": UCI_URL, "filas": len(df),
        "columnas_originales": df.columns.tolist(),
        "features": FEATURES, "excluidas": EXCLUDED,
        "positivos": int((df.y == "yes").sum()),
        "negativos": int((df.y == "no").sum()),
        "sha256": digest(RAW.read_bytes()),
        "supuestos": [
            "Sólo se analizan asociaciones históricas, no efecto causal del contacto.",
            "Al carecer de ID cliente, no se garantiza independencia por persona.",
            "Se excluyen duration, datos del contacto y campaña actual para un escenario antes de llamada.",
            "pdays=-1 es codificación histórica de no contactado previamente, no faltante.",
            "unknown es una categoría original, no un nulo artificial."
        ]
    }
    (OUT/"fuente_y_calidad.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"
    )
    return df


def make_splits(df: pd.DataFrame):
    if SPLITS.exists():
        split = pd.read_csv(SPLITS)
        assert len(split) == len(df), "Split desactualizado"
        assert split["indice_original"].is_unique
        assert np.array_equal(split.sort_values("indice_original")["y"].to_numpy(),
                              (df["y"] == "yes").astype(int).to_numpy())
    else:
        all_idx = np.arange(len(df))
        lab = (df["y"] == "yes").astype(int).to_numpy()
        main, test = train_test_split(
            all_idx, test_size=0.2, stratify=lab, random_state=SEED
        )
        train, val = train_test_split(
            main, test_size=0.25, stratify=lab[main], random_state=SEED
        )
        sets = np.full(len(df), "", dtype=object)
        sets[train], sets[val], sets[test] = "train", "validation", "test"
        split = pd.DataFrame({"indice_original": all_idx,
                              "conjunto": sets, "y": lab})
        SPLITS.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(SPLITS, index=False)
    ids = {part: split.loc[split.conjunto == part,
                           "indice_original"].to_numpy(dtype=int)
           for part in ("train", "validation", "test")}
    assert set.union(*[set(v) for v in ids.values()]) == set(range(len(df)))
    assert not (set(ids["train"]) & set(ids["validation"]))
    assert not (set(ids["train"]) & set(ids["test"]))
    assert not (set(ids["validation"]) & set(ids["test"]))
    return ids


if __name__ == "__main__":
    d = load_data()
    ids = make_splits(d)
    print("Dataset", d.shape, "conversiones:", (d.y=="yes").sum())
    print("Particiones:", {k: len(v) for k,v in ids.items()})
