"""AI Image Generator — Gradio preview.

Type a prompt, get an image. Stable Diffusion via diffusers. GPU recommended.
"""

from __future__ import annotations

import gradio as gr

import generate as g


def on_generate(prompt, negative, steps, guidance, seed):
    seed_val = int(seed) if seed is not None else -1
    image, status = g.generate(prompt, negative, steps, guidance, seed_val)
    return image, status


with gr.Blocks(title="AI Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎨 AI Image Generator\n"
        "Generate images from text prompts with Stable Diffusion. "
        "Tune steps, guidance, and a negative prompt to shape the result."
    )
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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
