# EXPLAINER — seg-studio: classical CV from scratch

## What I implemented from scratch

- **K-means** image segmentation (Lloyd's algorithm + k-means++ init) — `core/kmeans.py`
- **Region growing** (interactive flood fill by color similarity) — `core/region_grow.py`
- **Connected-components labeling** (4-connectivity, iterative) — `core/components.py`

No OpenCV/scikit-image for any of the segmentation logic.

## How each works

**K-means color segmentation.** Treat each pixel as a point in RGB (optionally RGB+xy) space and cluster into `k` groups:
1. Initialize centers with k-means++ (pick spread-out seeds so clustering converges better).
2. Assign each pixel to its nearest center.
3. Move each center to the mean of its members.
4. Repeat until assignments stop changing.
Recoloring each pixel by its center's color produces the segmentation. Adding scaled (x,y) to the feature vector biases clusters toward spatially-compact regions.

**Region growing.** Interactive segmentation: from a seed pixel (the user's click), do a BFS over the 4-neighbor grid, adding a neighbor if its color is within a threshold of the growing region's *running mean*. An explicit queue makes the flood order clear. This is how "magic wand" selection works.

**Connected components.** Given a binary mask, iterative flood fill assigns each connected blob a unique label (4-connectivity). Enables object counting. The test confirms diagonal-only touching pixels are *not* connected under 4-connectivity — a classic correctness check.

## Proof it works

`tests/test_core.py`:
- K-means cleanly separates two well-separated 3D clusters.
- Region growing selects a solid colored block from an interior seed and excludes the background.
- Connected components counts two separate blobs and respects 4-connectivity.

## Limitations

- Pure-Python pixel loops are readable but slow; the app downscales images for interactivity.
- These are classical (unsupervised, no learned features) methods. They segment by color/spatial coherence, not semantics — a learned model (SAM/U-Net) would segment by object identity. That's the honest trade-off: this proves the fundamentals.
