"""Tests proving the from-scratch connected-components labeler.

These assert *concrete* component counts and per-blob areas on synthetic
images, including an 8-connectivity diagonal case that 4-connectivity must
split.
"""

import numpy as np

from core.components import (
    describe_blobs,
    flood_fill_label,
    label_components,
)


def _areas(labels, count):
    """Return a sorted list of component pixel-areas (excluding background)."""
    return sorted(int(np.count_nonzero(labels == k)) for k in range(1, count + 1))


def test_two_separate_blobs_counts_and_areas():
    """Two disjoint rectangles => exactly 2 components with known areas."""
    img = np.zeros((10, 10), dtype=np.uint8)
    img[1:4, 1:4] = 1          # 3x3 = 9 px
    img[6:8, 6:9] = 1          # 2x3 = 6 px

    labels, count = label_components(img, connectivity=8)
    assert count == 2
    assert _areas(labels, count) == [6, 9]

    # Descriptors should report the same areas and correct bounding boxes.
    blobs = describe_blobs(labels, count)
    by_area = {b.area: b for b in blobs}
    assert set(by_area) == {6, 9}
    assert by_area[9].bbox == (1, 1, 3, 3)
    assert by_area[6].bbox == (6, 6, 7, 8)


def test_diagonal_touch_8_vs_4_connectivity():
    """A diagonal chain is ONE blob under 8-conn but THREE under 4-conn."""
    img = np.zeros((5, 5), dtype=np.uint8)
    img[0, 0] = 1
    img[1, 1] = 1
    img[2, 2] = 1

    labels8, count8 = label_components(img, connectivity=8)
    assert count8 == 1
    assert _areas(labels8, count8) == [3]

    labels4, count4 = label_components(img, connectivity=4)
    assert count4 == 3
    assert _areas(labels4, count4) == [1, 1, 1]


def test_flood_fill_matches_two_pass():
    """The independent flood-fill labeler must agree with the two-pass one."""
    rng = np.random.default_rng(0)
    img = (rng.random((40, 40)) > 0.6).astype(np.uint8)

    _, c_twopass = label_components(img, connectivity=8)
    _, c_flood = flood_fill_label(img, connectivity=8)
    assert c_twopass == c_flood

    # 4-connectivity should produce >= as many components as 8-connectivity.
    _, c4 = label_components(img, connectivity=4)
    assert c4 >= c_twopass


def test_u_shape_single_component():
    """A U-shaped region stays a single 8-connected component."""
    img = np.zeros((6, 6), dtype=np.uint8)
    img[1:5, 1] = 1
    img[1:5, 4] = 1
    img[4, 1:5] = 1
    labels, count = label_components(img, connectivity=8)
    assert count == 1


def test_empty_mask_zero_components():
    img = np.zeros((8, 8), dtype=np.uint8)
    labels, count = label_components(img)
    assert count == 0
    assert labels.sum() == 0
    assert describe_blobs(labels, count) == []
