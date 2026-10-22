import pandas as pd
import pytest

from lighthr.universities import RUSSELL_GROUP, is_russell_group, normalise, spelling_report

# Every spelling variant found in the LightHR data, and what it should match.
MESSY_RUSSELL_GROUP = [
    " University of Oxford",
    "Universirt of Cambridge",
    "The University of Manchester",
    "University of Nottingham, The",
    "Queen's University of Belfast",
    "London School of Economics & Political Science",
    "University of Durham",
]

# Real universities whose names overlap a Russell Group name but are not members.
NOT_RUSSELL_GROUP = [
    "Oxford Brookes University",
    "Sheffield Hallam University",
    "Leeds Metropolitan University",
    "Leeds Trinity University",
    "Liverpool Hope University",
    "Liverpool John Moores University",
    "York St John University",
    "Southampton Solent University",
    "Manchester Metropolitan University",
    "Nottingham Trent University",
    "Birmingham City University",
    "University College Birmingham",
    "Glasgow Caledonian University",
    "Glasgow School of Art.",
    "Edinburgh Napier University",
    "Cardiff Metropolitan University",
    "Northumbria University Newcastle",
    "University of the West of England, Bristol",
    "The University of Buckingham",
    "Buckinghamshire New University",
    "Kingston University",
]


def test_all_24_canonical_names_match_themselves():
    assert len(RUSSELL_GROUP) == 24
    assert is_russell_group(pd.Series(sorted(RUSSELL_GROUP))).all()


@pytest.mark.parametrize("name", MESSY_RUSSELL_GROUP)
def test_messy_spellings_are_matched(name):
    assert is_russell_group(pd.Series([name])).iat[0], normalise(name)


@pytest.mark.parametrize("name", NOT_RUSSELL_GROUP)
def test_similar_names_are_not_matched(name):
    assert not is_russell_group(pd.Series([name])).iat[0], normalise(name)


def test_missing_values_are_not_russell_group():
    assert not is_russell_group(pd.Series([None, float("nan"), ""])).any()


def test_spelling_report_lists_only_rescued_rows():
    names = pd.Series(["University of Oxford", " University of Oxford", " University of Oxford",
                       "Oxford Brookes University"])
    report = spelling_report(names)
    assert report.to_dict("records") == [
        {"raw name": " University of Oxford", "rows": 2, "matches": "UNIVERSITY OF OXFORD"}
    ]
