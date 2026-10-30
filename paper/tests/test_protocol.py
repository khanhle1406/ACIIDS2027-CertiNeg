"""Check the actual data split and complete-group sampling protocol."""

from __future__ import annotations

from completion_data import KEYS, load_data, selected_rows


def test_completion_groups_are_complete_and_disjoint() -> None:
    df, tr, te = load_data()
    cal, test = selected_rows(df, tr, 16), selected_rows(df, te, 64)
    assert cal.parent.nunique() == 16
    assert test.parent.nunique() == 64
    assert set(cal.parent).isdisjoint(test.parent)
    assert set(cal.complaint_id).isdisjoint(test.complaint_id)
    for subset in (cal, test):
        expected = df.loc[df.parent.isin(subset.parent)].complaint_id
        assert set(subset.complaint_id) == set(expected)
        for keys in KEYS:
            assert subset.groupby(list(keys)).size().min() > 1
