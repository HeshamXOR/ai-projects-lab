# 🧩 seg-studio — classical image segmentation from scratch

Two computer-vision segmentation algorithms implemented **by hand** (no OpenCV/scikit-image for the logic), wrapped in an interactive Gradio app.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What I implemented from scratch

- **K-means color segmentation** — Lloyd's algorithm with k-means++ init, in RGB(+xy) space — `core/kmeans.py`
- **Region growing** — click-to-segment via BFS flood fill on color similarity — `core/region_grow.py`
- **Connected-components labeling** — iterative flood fill for object counting — `core/components.py`

See [EXPLAINER.md](EXPLAINER.md); verify with `pytest`.

## Why it's here

This is the computer-vision pillar. It shows segmentation isn't magic from a library call — it's clustering, graph traversal, and connectivity, which I can implement and explain.

## Run it

```bash
pip install -r requirements.txt
python app.py        # http://localhost:7860 (+ public gradio.live link)
```

- **K-means tab**: pick `k`, segment an image into color regions.
- **Click-to-segment tab**: click a point, grow a region by color similarity.

CPU-only. Images are auto-downscaled so the from-scratch loops stay responsive.

## Verify

```bash
pytest -q   # k-means separates clusters; region growing fills uniform blocks;
            # connected components counts blobs (4-connectivity)
```

## Limitations

- Pure-Python pixel loops are clear but not fast; images are downscaled for interactivity. A production version would vectorize or use a compiled backend.
- Classical methods (no learned features); a pretrained SAM/U-Net comparison is a natural extension.
