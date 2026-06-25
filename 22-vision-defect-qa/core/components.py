"""Connected-components labeling and per-blob shape descriptors, from scratch.

WHY THIS MODULE EXISTS
----------------------
Industrial defect inspection rarely cares about *pixels*; it cares about
*defects* -- discrete connected regions of "suspicious" pixels (scratches,
dents, pits, contamination). To go from a binary defect mask to a list of
defect *objects* we must group spatially-connected foreground pixels into
labeled components and then summarize each component's geometry.

We deliberately implement connected-components labeling ourselves (no
``scipy.ndimage.label``, no OpenCV ``connectedComponents``) because it is the
algorithmic heart of the pipeline and the assignment requires the real thing.
We use the classic *two-pass* algorithm backed by a *union-find* (disjoint-set)
structure, supporting both 4- and 8-connectivity. A from-scratch iterative
flood-fill labeler is also provided and used as a cross-check in the tests.

The two-pass algorithm is preferred for the main path because it is O(N
alpha(N)) (effectively linear) and cache-friendly, whereas flood fill can blow
the recursion stack on large blobs unless written iteratively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "BlobDescriptor",
    "label_components",
    "flood_fill_label",
    "describe_blobs",
    "UnionFind",
]


class UnionFind:
    """Disjoint-set forest with path compression and union by rank.

    Provisional labels created during the first labeling pass are merged when
    we discover that two of them actually belong to the same component. The
    near-constant-time ``find``/``union`` make the whole labeler effectively
    linear in the number of pixels.
    """

    def __init__(self, n: int = 0) -> None:
        self._parent: List[int] = list(range(n))
        self._rank: List[int] = [0] * n

    def make_set(self) -> int:
        """Allocate a fresh singleton set and return its id."""
        idx = len(self._parent)
        self._parent.append(idx)
        self._rank.append(0)
        return idx

    def find(self, x: int) -> int:
        """Return the canonical root of ``x`` with iterative path compression."""
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression: point every node on the path straight at the root.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        """Merge the sets containing ``a`` and ``b``; return the new root."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return ra


def _neighbor_offsets(connectivity: int) -> Tuple[Tuple[int, int], ...]:
    """Return the *causal* (already-visited) neighbor offsets for pass 1.

    During a raster scan only the pixels above and to the left have been
    assigned labels, so we only look at those. For 4-connectivity that is the
    north and west neighbors; for 8-connectivity we add the two upper
    diagonals (north-west and north-east).
    """
    if connectivity == 4:
        return ((-1, 0), (0, -1))
    if connectivity == 8:
        return ((-1, 0), (0, -1), (-1, -1), (-1, 1))
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")


def label_components(
    mask: np.ndarray, connectivity: int = 8
) -> Tuple[np.ndarray, int]:
    """Label connected foreground regions of a binary ``mask`` (two-pass).

    Parameters
    ----------
    mask:
        2D array; any non-zero pixel is treated as foreground.
    connectivity:
        4 (von Neumann) or 8 (Moore). 8 connects diagonal touches.

    Returns
    -------
    (labels, count):
        ``labels`` is an int32 array the same shape as ``mask`` where 0 is
        background and 1..count are component ids assigned in raster order of
        first appearance. ``count`` is the number of distinct components.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    fg = mask != 0
    h, w = fg.shape
    labels = np.zeros((h, w), dtype=np.int32)

    uf = UnionFind()
    # Provisional label 0 is a sentinel meaning "unassigned"; real provisional
    # labels start at 1 and each gets its own union-find set.
    prov_for_set: Dict[int, int] = {}  # set-root -> provisional label
    offsets = _neighbor_offsets(connectivity)

    next_prov = 1
    # Map provisional label -> union-find element id.
    prov_to_uf: Dict[int, int] = {}

    # ---- Pass 1: assign provisional labels, record equivalences -----------
    for y in range(h):
        for x in range(w):
            if not fg[y, x]:
                continue
            neighbor_labels: List[int] = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] != 0:
                    neighbor_labels.append(int(labels[ny, nx]))
            if not neighbor_labels:
                # New component seed.
                prov = next_prov
                next_prov += 1
                prov_to_uf[prov] = uf.make_set()
                labels[y, x] = prov
            else:
                smallest = min(neighbor_labels)
                labels[y, x] = smallest
                # Union all neighboring provisional labels together.
                base = prov_to_uf[smallest]
                for nl in neighbor_labels:
                    uf.union(base, prov_to_uf[nl])

    # ---- Build contiguous final ids from union-find roots ------------------
    root_to_final: Dict[int, int] = {}
    final_count = 0
    # Deterministic ordering: assign final ids by ascending provisional label
    # so that components are numbered in raster order of first appearance.
    for prov in range(1, next_prov):
        root = uf.find(prov_to_uf[prov])
        if root not in root_to_final:
            final_count += 1
            root_to_final[root] = final_count

    # ---- Pass 2: relabel pixels to their final component id ----------------
    if final_count:
        # Vectorized remap via a lookup table indexed by provisional label.
        lut = np.zeros(next_prov, dtype=np.int32)
        for prov in range(1, next_prov):
            root = uf.find(prov_to_uf[prov])
            lut[prov] = root_to_final[root]
        labels = lut[labels]

    return labels, final_count


def flood_fill_label(
    mask: np.ndarray, connectivity: int = 8
) -> Tuple[np.ndarray, int]:
    """Alternative labeler using iterative flood fill (BFS via an explicit stack).

    Implemented independently of :func:`label_components` so the test-suite can
    cross-check the two against each other. Uses an explicit stack to avoid
    Python recursion-depth limits on large blobs.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    fg = mask != 0
    h, w = fg.shape
    labels = np.zeros((h, w), dtype=np.int32)

    if connectivity == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    elif connectivity == 8:
        steps = (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        )
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")

    current = 0
    for sy in range(h):
        for sx in range(w):
            if not fg[sy, sx] or labels[sy, sx] != 0:
                continue
            current += 1
            stack = [(sy, sx)]
            labels[sy, sx] = current
            while stack:
                y, x = stack.pop()
                for dy, dx in steps:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and fg[ny, nx]
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels, current


@dataclass
class BlobDescriptor:
    """Geometric summary of a single connected component (a candidate defect).

    All coordinates are in (row=y, col=x) pixel space. The descriptors are the
    classical shape features used by rule-based and ML defect classifiers.
    """

    label: int
    area: int
    # Bounding box as (min_y, min_x, max_y, max_x) inclusive.
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]  # (y, x)
    perimeter: float
    aspect_ratio: float            # bbox width / height (>= ... ) guarded
    extent: float                  # area / bbox area (fill ratio)
    eccentricity: float            # 0 = circular, ->1 = elongated
    orientation: float = field(default=0.0)  # radians of major axis

    def as_dict(self) -> Dict[str, object]:
        """JSON-friendly representation for the API layer."""
        return {
            "label": int(self.label),
            "area": int(self.area),
            "bbox": [int(v) for v in self.bbox],
            "centroid": [float(c) for c in self.centroid],
            "perimeter": float(self.perimeter),
            "aspect_ratio": float(self.aspect_ratio),
            "extent": float(self.extent),
            "eccentricity": float(self.eccentricity),
            "orientation": float(self.orientation),
        }


def _estimate_perimeter(component_mask: np.ndarray) -> float:
    """Estimate the perimeter of a binary blob by counting boundary edges.

    For each foreground pixel we count its 4-neighbor sides that face either
    the image border or a background pixel; that edge-count is a robust,
    resolution-stable perimeter estimate (it is the length of the polygonal
    boundary of the pixel set). This avoids the systematic under-counting you
    get from simply counting boundary pixels.
    """
    h, w = component_mask.shape
    # Pad with background so border pixels expose their outward edges.
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = component_mask
    inner = padded[1:-1, 1:-1]
    edges = 0
    edges += np.count_nonzero(inner & ~padded[:-2, 1:-1])   # top side exposed
    edges += np.count_nonzero(inner & ~padded[2:, 1:-1])    # bottom side
    edges += np.count_nonzero(inner & ~padded[1:-1, :-2])   # left side
    edges += np.count_nonzero(inner & ~padded[1:-1, 2:])    # right side
    return float(edges)


def describe_blobs(
    labels: np.ndarray, count: int, min_area: int = 1
) -> List[BlobDescriptor]:
    """Compute :class:`BlobDescriptor` for every labeled component.

    Parameters
    ----------
    labels, count:
        Output of :func:`label_components` / :func:`flood_fill_label`.
    min_area:
        Components smaller than this (in pixels) are dropped as noise.

    The eccentricity and orientation are derived from the second central
    moments of the pixel coordinates -- i.e. we fit the covariance ellipse of
    the blob and read off its axis lengths. eccentricity = sqrt(1 - lambda_min
    / lambda_max). This is the standard image-moments definition.
    """
    descriptors: List[BlobDescriptor] = []
    for lab in range(1, count + 1):
        ys, xs = np.nonzero(labels == lab)
        area = int(ys.size)
        if area < min_area:
            continue
        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        bbox = (min_y, min_x, max_y, max_x)
        bb_h = max_y - min_y + 1
        bb_w = max_x - min_x + 1
        cy = float(ys.mean())
        cx = float(xs.mean())

        comp_mask = labels[min_y : max_y + 1, min_x : max_x + 1] == lab
        perimeter = _estimate_perimeter(comp_mask)

        aspect_ratio = float(bb_w) / float(bb_h) if bb_h else 0.0
        extent = float(area) / float(bb_h * bb_w) if bb_h * bb_w else 0.0

        # Second central moments (image-moment covariance ellipse).
        dy = ys.astype(np.float64) - cy
        dx = xs.astype(np.float64) - cx
        if area > 1:
            mu_yy = float(np.mean(dy * dy))
            mu_xx = float(np.mean(dx * dx))
            mu_xy = float(np.mean(dx * dy))
            cov = np.array([[mu_yy, mu_xy], [mu_xy, mu_xx]], dtype=np.float64)
            eigvals = np.linalg.eigvalsh(cov)
            lam_min = float(max(eigvals[0], 0.0))
            lam_max = float(max(eigvals[1], 1e-12))
            eccentricity = float(np.sqrt(max(0.0, 1.0 - lam_min / lam_max)))
            orientation = 0.5 * float(np.arctan2(2.0 * mu_xy, mu_yy - mu_xx))
        else:
            eccentricity = 0.0
            orientation = 0.0

        descriptors.append(
            BlobDescriptor(
                label=lab,
                area=area,
                bbox=bbox,
                centroid=(cy, cx),
                perimeter=perimeter,
                aspect_ratio=aspect_ratio,
                extent=extent,
                eccentricity=eccentricity,
                orientation=orientation,
            )
        )
    return descriptors
