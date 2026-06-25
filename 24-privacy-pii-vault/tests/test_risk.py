"""k-anonymity risk scorer: correct k, flagged rows, and l-diversity."""

from __future__ import annotations

from core.risk import build_equivalence_classes, k_anonymity


# A constructed table over quasi-identifiers {zip, age, sex}. By design:
#   - ("10001", 30, "M") appears 3 times  -> class size 3
#   - ("10001", 30, "F") appears 2 times  -> class size 2
#   - ("90210", 45, "F") appears 1 time   -> class size 1  (UNIQUE -> at risk)
# So the minimum class size k = 1, and the single unique row is flagged.
TABLE = [
    {"zip": "10001", "age": 30, "sex": "M", "disease": "flu"},      # 0
    {"zip": "10001", "age": 30, "sex": "M", "disease": "cold"},     # 1
    {"zip": "10001", "age": 30, "sex": "M", "disease": "flu"},      # 2
    {"zip": "10001", "age": 30, "sex": "F", "disease": "cancer"},   # 3
    {"zip": "10001", "age": 30, "sex": "F", "disease": "cancer"},   # 4
    {"zip": "90210", "age": 45, "sex": "F", "disease": "flu"},      # 5 unique
]
QIS = ["zip", "age", "sex"]


def test_equivalence_classes_partition_correctly():
    classes = build_equivalence_classes(TABLE, QIS)
    sizes = sorted(ec.size for ec in classes.values())
    assert sizes == [1, 2, 3]


def test_k_value_is_minimum_class_size():
    report = k_anonymity(TABLE, QIS, threshold=2)
    assert report.k == 1
    assert report.num_classes == 3
    assert report.is_k_anonymous is False


def test_flags_the_unique_row():
    report = k_anonymity(TABLE, QIS, threshold=2)
    # Row index 5 is the unique ("90210", 45, "F") record.
    assert report.flagged_rows == [5]


def test_higher_threshold_flags_more_rows():
    # With threshold 3, the F-class (size 2) and the unique row (size 1) fail.
    report = k_anonymity(TABLE, QIS, threshold=3)
    assert report.k == 1
    assert sorted(report.flagged_rows) == [3, 4, 5]
    assert report.is_k_anonymous is False


def test_k_anonymous_table_passes():
    # Drop the unique row and one F so every class has size >= 2... actually
    # build a clean 2-anonymous table directly.
    clean = [
        {"zip": "10001", "age": 30, "sex": "M"},
        {"zip": "10001", "age": 30, "sex": "M"},
        {"zip": "10001", "age": 30, "sex": "F"},
        {"zip": "10001", "age": 30, "sex": "F"},
    ]
    report = k_anonymity(clean, QIS, threshold=2)
    assert report.k == 2
    assert report.is_k_anonymous is True
    assert report.flagged_rows == []


def test_l_diversity_reports_min_distinct_sensitive_values():
    report = k_anonymity(TABLE, QIS, threshold=1, sensitive_attrs=["disease"])
    # Classes' distinct disease counts:
    #   M-class: {flu, cold}      -> 2
    #   F-class: {cancer}         -> 1  (homogeneous! leaks the value)
    #   unique:  {flu}            -> 1
    # l = min = 1.
    assert report.l_diversity["disease"] == 1


def test_empty_table_is_not_k_anonymous():
    report = k_anonymity([], QIS, threshold=2)
    assert report.k == 0
    assert report.is_k_anonymous is False
    assert report.flagged_rows == []
