"""seg-studio — classical image segmentation from scratch.

Two CV algorithms implemented by hand (no OpenCV/scikit-image for the logic):
  * K-means color segmentation — cluster pixels into k regions.
  * Region growing — click a point, flood-fill a region by color similarity.

Upload an image and try both. Everything that does the segmenting lives in
core/ and is unit-tested.
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from core.kmeans import segment_image
from core.region_grow import region_grow, overlay_mask
from core.components import count_objects


def _downscale(img: np.ndarray, max_side: int = 256) -> np.ndarray:
    """Nearest-neighbor downscale so the from-scratch loops stay snappy."""
    h, w = img.shape[:2]
    scale = max(h, w) / max_side
    if scale <= 1:
        return img
    nh, nw = int(h / scale), int(w / scale)
    ys = (np.arange(nh) * scale).astype(int).clip(0, h - 1)
    xs = (np.arange(nw) * scale).astype(int).clip(0, w - 1)
    return img[ys][:, xs]


def on_kmeans(image, k):
    if image is None:
        return None, "Upload an image first."
    img = _downscale(np.asarray(image)[:, :, :3])
    seg, labels = segment_image(img, k=int(k))
    return seg, f"Segmented into {int(k)} color regions with from-scratch k-means."


def on_click_segment(image, threshold, evt: gr.SelectData):
    if image is None:
        return None, "Upload an image and click on it."
    img = _downscale(np.asarray(image)[:, :, :3])
    # evt.index is (x, y) in displayed coords; map to the downscaled array
    disp = np.asarray(image)
    sx = int(evt.index[0] * img.shape[1] / disp.shape[1])
    sy = int(evt.index[1] * img.shape[0] / disp.shape[0])
    mask = region_grow(img, (sx, sy), threshold=float(threshold))
    out = overlay_mask(img, mask)
    pct = 100 * mask.mean()
    return out, f"Region grown from ({sx},{sy}) covers {pct:.1f}% of the image."


with gr.Blocks(title="seg-studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🧩 seg-studio — classical image segmentation from scratch\n"
        "Two computer-vision algorithms implemented by hand (no OpenCV for the "
        "logic): **k-means** color segmentation and **region growing** "
        "(click-to-segment). See `core/` and the tests."
    )
    with gr.Tab("K-means color segmentation"):
        with gr.Row():
            inp = gr.Image(type="numpy", label="Image")
            outp = gr.Image(type="numpy", label="Segmented")
        k = gr.Slider(2, 8, value=4, step=1, label="Number of regions (k)")
        btn = gr.Button("Segment", variant="primary")
        status = gr.Markdown()
        btn.click(on_kmeans, [inp, k], [outp, status])
    with gr.Tab("Click-to-segment (region growing)"):
        gr.Markdown("Upload an image, then **click a point** to grow a region from it.")
        with gr.Row():
            inp2 = gr.Image(type="numpy", label="Image (click to seed)")
            outp2 = gr.Image(type="numpy", label="Selected region")
        threshold = gr.Slider(5, 80, value=25, step=1, label="Color threshold")
        status2 = gr.Markdown()
        inp2.select(on_click_segment, [inp2, threshold], [outp2, status2])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
