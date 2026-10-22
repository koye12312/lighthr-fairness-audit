"""Matching messy university names to the Russell Group.

The LightHR data spells some universities several ways, for example
" University of Oxford" (leading space), "Universirt of Cambridge" (typo),
"University of Nottingham, The" and "London School of Economics & Political
Science". Exact matching counts those rows as non-Russell Group.
``is_russell_group`` normalises a name first and then compares it against one
canonical spelling per university.

Normalisation is deliberately narrow: case, whitespace, "&", a leading or
trailing "The", one known misspelling and two known alternative names. It
never does fuzzy or substring matching, because that would turn
"Oxford Brookes University" or "Sheffield Hallam University" into Russell
Group members.
"""
from __future__ import annotations

import re

import pandas as pd

RUSSELL_GROUP = frozenset({
    "UNIVERSITY OF BIRMINGHAM",
    "UNIVERSITY OF BRISTOL",
    "UNIVERSITY OF CAMBRIDGE",
    "CARDIFF UNIVERSITY",
    "DURHAM UNIVERSITY",
    "UNIVERSITY OF EDINBURGH",
    "UNIVERSITY OF EXETER",
    "UNIVERSITY OF GLASGOW",
    "IMPERIAL COLLEGE LONDON",
    "KING'S COLLEGE LONDON",
    "UNIVERSITY OF LEEDS",
    "UNIVERSITY OF LIVERPOOL",
    "LONDON SCHOOL OF ECONOMICS AND POLITICAL SCIENCE",
    "UNIVERSITY OF MANCHESTER",
    "NEWCASTLE UNIVERSITY",
    "UNIVERSITY OF NOTTINGHAM",
    "UNIVERSITY OF OXFORD",
    "QUEEN MARY UNIVERSITY OF LONDON",
    "QUEEN'S UNIVERSITY BELFAST",
    "UNIVERSITY OF SHEFFIELD",
    "UNIVERSITY OF SOUTHAMPTON",
    "UNIVERSITY COLLEGE LONDON",
    "UNIVERSITY OF WARWICK",
    "UNIVERSITY OF YORK",
})

# Misspellings seen in the data, fixed word by word.
_TYPOS = {"UNIVERSIRT": "UNIVERSITY"}

# Alternative official names that map to the canonical spelling above.
_ALIASES = {
    "UNIVERSITY OF DURHAM": "DURHAM UNIVERSITY",
    "QUEEN'S UNIVERSITY OF BELFAST": "QUEEN'S UNIVERSITY BELFAST",
}


def normalise(name: object) -> str:
    """Return a canonical upper-case spelling of a university name."""
    if not isinstance(name, str):
        return ""
    s = name.upper().replace("’", "'").replace("&", " AND ")
    s = re.sub(r"[^\w' ,]", " ", s)          # drop stray punctuation such as a final "."
    s = re.sub(r"\s+", " ", s).strip(" ,")
    s = re.sub(r",\s*THE$", "", s)           # "UNIVERSITY OF NOTTINGHAM, THE"
    s = re.sub(r"^THE ", "", s)              # "THE UNIVERSITY OF MANCHESTER"
    s = " ".join(_TYPOS.get(word, word) for word in s.split())
    return _ALIASES.get(s, s)


def is_russell_group(names: pd.Series) -> pd.Series:
    """Boolean Series: True where the (normalised) name is a Russell Group university."""
    return names.map(normalise).isin(RUSSELL_GROUP)


def spelling_report(names: pd.Series) -> pd.DataFrame:
    """Raw spellings that only match after normalisation, with row counts."""
    raw_upper = names.astype(str).str.upper()
    rescued = is_russell_group(names) & ~raw_upper.isin(RUSSELL_GROUP)
    out = (names[rescued].value_counts().rename_axis("raw name").reset_index(name="rows"))
    out["matches"] = out["raw name"].map(normalise)
    return out
