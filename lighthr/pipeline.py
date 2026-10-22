"""The preprocessing and final model from 01_lighthr_audit.ipynb, as functions.

``prepare`` rebuilds the classification frame exactly as 01_lighthr_audit.ipynb
does, with the Russell Group flag computed one of two ways:

* ``russell_group="exact"``: exact upper-case match (as in 01_lighthr_audit.ipynb)
* ``russell_group="normalised"``: ``lighthr.universities.is_russell_group``
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lighthr.universities import RUSSELL_GROUP, is_russell_group

RENAME = {
    "Refree 1": "Ref1", "Refree 2": "Ref2", "Refree 3": "Ref3",
    "Age at time of application": "Age", "Year of Application": "Year",
    "A-levels scores": "ALevels", "Current Annual Income": "Income",
    "Current Employment": "Employment", "Current Employment Sector": "EmpSector",
    "References Submitted": "RefsSubmitted", "University Name": "University",
    "Was Hired by a Company within 6 months": "Hired",
}
ENCODED = ["Gender", "Education", "Employment", "EmpSector", "Address"]
ADABOOST_PARAMS = dict(n_estimators=200, learning_rate=1.0, random_state=42)  # Part B winner


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=RENAME)
    df["Employment"] = df["Employment"].fillna("None")
    df["EmpSector"] = df["EmpSector"].fillna("None")
    for col in ["Ref1", "Ref2", "Ref3"]:
        df[col] = df[col].fillna("None:None")
    return df


def _referee_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i in (1, 2, 3):
        parts = df[f"Ref{i}"].str.split(":", expand=True)
        ok = parts.notna().all(axis=1) & (df[f"Ref{i}"] != "None:None")
        df[f"Ref{i}_Emp"] = np.where(ok, parts[0], "None")
        df[f"Ref{i}_Sec"] = np.where(ok, parts[1], "None")
    emp = [f"Ref{i}_Emp" for i in (1, 2, 3)]
    sec = [f"Ref{i}_Sec" for i in (1, 2, 3)]
    df["Ref_Count"] = (df[emp] != "None").sum(axis=1)
    match = sum((df[s] == df["EmpSector"]).astype(int) for s in sec)
    df["Ref_Sector_Match"] = np.where(df["EmpSector"] == "None", 0, match)
    df["Ref_Sector_Diversity"] = df[sec].apply(lambda r: len({v for v in r if v != "None"}), axis=1)
    df["Ref_Salaried_Count"] = (df[emp] == "Salaried").sum(axis=1)
    return df


def prepare(path, russell_group: str = "normalised"):
    """Return (raw frame with engineered columns, encoded classification frame)."""
    df = _referee_features(load_raw(path))
    if russell_group == "exact":
        df["Russell_Group"] = df["University"].str.upper().isin(RUSSELL_GROUP).astype(int)
    elif russell_group == "normalised":
        df["Russell_Group"] = is_russell_group(df["University"]).astype(int)
    else:
        raise ValueError(russell_group)

    drop = ["Ref1", "Ref2", "Ref3", "Ref1_Emp", "Ref2_Emp", "Ref3_Emp",
            "Ref1_Sec", "Ref2_Sec", "Ref3_Sec", "University"]
    cls = df.drop(columns=drop)
    for col in ENCODED:
        cls[col] = LabelEncoder().fit_transform(cls[col].astype(str))
    return df, cls


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split(cls: pd.DataFrame) -> Split:
    """The same stratified 80/20 split as Part B (random_state=42)."""
    X, y = cls.drop(columns=["Hired"]), cls["Hired"]
    return Split(*train_test_split(X, y, test_size=0.2, random_state=42, stratify=y))


def fit_adaboost(X, y, sample_weight=None) -> AdaBoostClassifier:
    return AdaBoostClassifier(**ADABOOST_PARAMS).fit(X, y, sample_weight=sample_weight)
