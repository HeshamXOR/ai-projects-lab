"""AI Image Generator — Gradio preview.

Two tabs:
  * Stable Diffusion text-to-image (pretrained, GPU recommended).
  * Toy diffusion (from scratch): train a tiny NumPy denoiser on 2D data and
    watch the reverse-diffusion process turn noise into a shape — the diffusion
    math (core/) made visible.
"""

from __future__ import annotations

import io

import gradio as gr

import generate as g


def on_generate(prompt, negative, steps, guidance, seed):
    seed_val = int(seed) if seed is not None else -1
    image, status = g.generate(prompt, negative, steps, guidance, seed_val)
    return image, status


def on_toy(shape, steps, progress=gr.Progress()):
    """Train the from-scratch toy diffusion model and plot noise -> data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from core.ddim import Diffusion
    from core.toy import NoisePredictor, sample, make_spiral, make_two_moons

    data = make_spiral(800) if shape == "spiral" else make_two_moons(800)
    diff = Diffusion(T=100)
    model = NoisePredictor(hidden=128, seed=0)
    progress(0.1, desc="training the from-scratch denoiser…")
    model.train(data, diff, epochs=int(steps), batch=128, lr=2e-3)
    progress(0.8, desc="sampling via DDIM…")
    gen = sample(model, diff, n=600, steps=40)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
    ax1.scatter(data[:, 0], data[:, 1], s=6, c="#34d6df"); ax1.set_title("target data")
    ax2.scatter(gen[:, 0], gen[:, 1], s=6, c="#7c5cff"); ax2.set_title("generated (from noise)")
    for a in (ax1, ax2):
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=100); plt.close(fig); buf.seek(0)
    return Image.open(buf), "Trained a from-scratch noise-predictor and sampled with the DDIM loop in `core/`."


with gr.Blocks(title="AI Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎨 AI Image Generator + diffusion from scratch\n"
        "Generate images with Stable Diffusion — **and** see the diffusion "
        "process itself, implemented from scratch on 2D data."
    )
    with gr.Tab("Stable Diffusion"):
        gr.Markdown(g.device_label())
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    value="a cozy cabin in a snowy forest at dusk, warm lights, highly detailed, digital art",
                    lines=3,
                )
                negative = gr.Textbox(label="Negative prompt", value="blurry, low quality, distorted", lines=2)
                with gr.Row():
                    steps = gr.Slider(5, 50, value=25, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance")
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                image = gr.Image(label="Result", type="pil")
                status = gr.Markdown()
        btn.click(on_generate, inputs=[prompt, negative, steps, guidance, seed], outputs=[image, status])
        gr.Examples(
            examples=[
                "an astronaut riding a horse on the moon, photorealistic",
                "a watercolor painting of a Japanese garden in autumn",
                "a futuristic city skyline at night, neon, cyberpunk",
            ],
            inputs=prompt,
        )
    with gr.Tab("Diffusion from scratch (2D)"):
        gr.Markdown(
            "Train a tiny noise-predictor (NumPy, hand-written backprop) on a 2D "
            "shape, then sample new points with the **DDIM** loop. Watch Gaussian "
            "noise become structure — the same idea SD uses on image latents."
        )
        with gr.Row():
            shape = gr.Dropdown(["two moons", "spiral"], value="two moons", label="Target shape")
            toy_steps = gr.Slider(200, 2000, value=800, step=100, label="Training steps")
        toy_btn = gr.Button("Train & sample (from scratch)", variant="primary")
        toy_img = gr.Image(label="Target vs. generated")
        toy_status = gr.Markdown()
        toy_btn.click(on_toy, inputs=[shape, toy_steps], outputs=[toy_img, toy_status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
