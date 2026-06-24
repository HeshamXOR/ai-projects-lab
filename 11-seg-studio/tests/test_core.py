"""Proofs for the from-scratch classical-CV segmentation."""

import numpy as np

from core.kmeans import kmeans, segment_image
from core.region_grow import region_grow
from core.components import connected_components, count_objects


def test_kmeans_separates_clear_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, (100, 3))
    b = rng.normal(5, 0.1, (100, 3))
    X = np.vstack([a, b])
    labels, centers = kmeans(X, 2, seed=1)
    # the two true groups should land in two different clusters
    assert len(set(labels[:100])) == 1
    assert len(set(labels[100:])) == 1
    assert labels[0] != labels[100]


def test_segment_image_shapes():
    img = (np.random.default_rng(0).random((20, 20, 3)) * 255).astype(np.uint8)
    seg, label_map = segment_image(img, k=3, seed=0)
    assert seg.shape == img.shape
    assert label_map.shape == (20, 20)
    assert label_map.max() < 3


def test_region_grow_fills_uniform_block():
    # a solid red square on black: growing from inside it should select the square
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15] = [200, 0, 0]
    mask = region_grow(img, (10, 10), threshold=30)
    # the whole red block is selected, background is not
    assert mask[5:15, 5:15].all()
    assert not mask[0, 0]


def test_connected_components_counts_blobs():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True     # blob 1
    mask[6:9, 6:9] = True     # blob 2
    labels = connected_components(mask)
    assert labels.max() == 2
    assert count_objects(mask) == 2


def test_connected_components_diagonal_not_connected():
    # 4-connectivity: diagonal touch is NOT one component
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    assert connected_components(mask).max() == 2
