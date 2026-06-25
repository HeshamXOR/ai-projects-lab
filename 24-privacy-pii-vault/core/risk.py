"""Re-identification risk scoring: k-anonymity and l-diversity from scratch.

WHY this module exists
----------------------
Removing direct identifiers (name, SSN) is not enough. *Quasi-identifiers*
(QIs) -- attributes like ZIP, birth date, sex -- can be combined to re-identify
individuals by linking to external data (Sweeney's classic result: 87% of the
US population is unique on {ZIP, birthdate, sex}). k-anonymity formalizes the
defense: a dataset is *k-anonymous* on a set of QIs if every record is
indistinguishable from at least ``k - 1`` others on those QIs. Equivalently,
group the rows into *equivalence classes* by their QI tuple; ``k`` is the size
of the smallest class. Rows in any class smaller than a threshold are
re-identifiable and must be suppressed or generalized.

We implement this directly: build the equivalence classes, take the minimum
size, and flag the small ones. We also compute a simple **l-diversity** note
per sensitive attribute -- a class can be k-anonymous yet leak a sensitive
value if everyone in it shares it (e.g. all "HIV positive"); l-diversity
requires at least ``l`` distinct sensitive values per class.

Everything here is pure Python over plain row dicts -- no pandas, no external
stats library -- so the algorithm is fully visible and unit-tested.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

Row = Dict[str, object]


@dataclass
class EquivalenceClass:
    """A group of rows sharing the same quasi-identifier tuple."""

    key: Tuple[Hashable, ...]
    row_indices: List[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Number of rows in this equivalence class."""
        return len(self.row_indices)


@dataclass
class RiskReport:
    """Result of a k-anonymity (and optional l-diversity) analysis.

    Attributes
    ----------
    k:
        The minimum equivalence-class size over the whole table (the
        k-anonymity value). 0 for an empty table.
    threshold:
        The k below which rows are considered re-identifiable.
    is_k_anonymous:
        True iff ``k >= threshold``.
    num_classes:
        Number of distinct quasi-identifier combinations.
    flagged_rows:
        Indices of rows that fall in an equivalence class smaller than
        ``threshold`` (the at-risk records).
    class_sizes:
        Mapping of QI tuple (as a string) -> class size, for reporting.
    l_diversity:
        Optional mapping sensitive_attr -> minimum distinct values across
        classes (the l value). Present only when sensitive attrs were given.
    smallest_classes:
        Up to a few of the smallest classes, for actionable output.
    """

    k: int
    threshold: int
    is_k_anonymous: bool
    num_classes: int
    flagged_rows: List[int]
    class_sizes: Dict[str, int]
    l_diversity: Dict[str, int] = field(default_factory=dict)
    smallest_classes: List[Tuple[str, int]] = field(default_factory=list)


def _qi_key(row: Row, quasi_identifiers: Sequence[str]) -> Tuple[Hashable, ...]:
    """Build the hashable equivalence-class key for ``row``.

    Missing QI columns are represented by ``None`` so rows that omit a field
    still group consistently rather than raising.
    """
    return tuple(row.get(qi) for qi in quasi_identifiers)


def build_equivalence_classes(
    rows: Sequence[Row], quasi_identifiers: Sequence[str]
) -> Dict[Tuple[Hashable, ...], EquivalenceClass]:
    """Partition ``rows`` into equivalence classes by their QI tuple."""
    classes: Dict[Tuple[Hashable, ...], EquivalenceClass] = {}
    for index, row in enumerate(rows):
        key = _qi_key(row, quasi_identifiers)
        ec = classes.get(key)
        if ec is None:
            ec = EquivalenceClass(key=key)
            classes[key] = ec
        ec.row_indices.append(index)
    return classes


def _l_diversity(
    rows: Sequence[Row],
    classes: Dict[Tuple[Hashable, ...], EquivalenceClass],
    sensitive_attrs: Sequence[str],
) -> Dict[str, int]:
    """Compute the minimum count of distinct sensitive values per class.

    For each sensitive attribute we look at every equivalence class, count the
    distinct values it contains, and take the minimum across classes -- that
    minimum is the dataset's l-diversity for that attribute. Low l means at
    least one class is homogeneous on a sensitive field and leaks it even when
    k-anonymous.
    """
    result: Dict[str, int] = {}
    for attr in sensitive_attrs:
        min_distinct: Optional[int] = None
        for ec in classes.values():
            distinct = {rows[i].get(attr) for i in ec.row_indices}
            count = len(distinct)
            min_distinct = count if min_distinct is None else min(min_distinct, count)
        result[attr] = min_distinct or 0
    return result


def k_anonymity(
    rows: Sequence[Row],
    quasi_identifiers: Sequence[str],
    threshold: int = 2,
    sensitive_attrs: Optional[Sequence[str]] = None,
) -> RiskReport:
    """Compute the k-anonymity of ``rows`` over ``quasi_identifiers``.

    Parameters
    ----------
    rows:
        The table as a sequence of dict records.
    quasi_identifiers:
        Column names treated as quasi-identifiers.
    threshold:
        Minimum acceptable equivalence-class size. Rows in smaller classes are
        flagged as re-identifiable; the table is k-anonymous iff ``k >=
        threshold``.
    sensitive_attrs:
        Optional sensitive columns to additionally report l-diversity for.

    Returns
    -------
    RiskReport
        Full analysis including k, flagged rows, and class sizes.
    """
    if threshold < 1:
        raise ValueError("threshold must be >= 1")
    if not quasi_identifiers:
        raise ValueError("at least one quasi-identifier column is required")

    if not rows:
        return RiskReport(
            k=0,
            threshold=threshold,
            is_k_anonymous=False,
            num_classes=0,
            flagged_rows=[],
            class_sizes={},
        )

    classes = build_equivalence_classes(rows, quasi_identifiers)
    sizes = [ec.size for ec in classes.values()]
    k = min(sizes)

    flagged: List[int] = []
    class_sizes: Dict[str, int] = {}
    for key, ec in classes.items():
        label = " | ".join(str(part) for part in key)
        class_sizes[label] = ec.size
        if ec.size < threshold:
            flagged.extend(ec.row_indices)
    flagged.sort()

    smallest = sorted(class_sizes.items(), key=lambda kv: kv[1])[:5]

    l_div: Dict[str, int] = {}
    if sensitive_attrs:
        l_div = _l_diversity(rows, classes, sensitive_attrs)

    return RiskReport(
        k=k,
        threshold=threshold,
        is_k_anonymous=k >= threshold,
        num_classes=len(classes),
        flagged_rows=flagged,
        class_sizes=class_sizes,
        l_diversity=l_div,
        smallest_classes=smallest,
    )
