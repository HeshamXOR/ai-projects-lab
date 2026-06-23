"""Image Q&A (multimodal) — Gradio preview.

Upload an image: get an automatic caption, then ask visual questions about it.
Uses BLIP. Runs on GPU if available, else CPU.
"""

from __future__ import annotations

import gradio as gr

import vision


def on_caption(image):
    return vision.caption(image)


def on_ask(image, question, history):
    history = history or []
    if image is None:
        history.append((question, "Upload an image first."))
        return history, ""
    ans = vision.answer(image, question)
    history.append((question, ans))
    return history, ""


with gr.Blocks(title="Image Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🖼️ Image Q&A (Multimodal)\n"
        "Upload an image to get an automatic caption, then ask questions about "
        "what's in it. Powered by the BLIP vision-language model."
    )
    gr.Markdown(vision.device_label())
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Image")
            cap_btn = gr.Button("Describe image", variant="secondary")
            caption_out = gr.Textbox(label="Caption", interactive=False)
        with gr.Column():
            chat = gr.Chatbot(label="Visual Q&A", height=360)
            q = gr.Textbox(label="Ask about the image", placeholder="e.g. What color is the car? How many people?")
            ask_btn = gr.Button("Ask", variant="primary")

    cap_btn.click(on_caption, inputs=img, outputs=caption_out)
    ask_btn.click(on_ask, inputs=[img, q, chat], outputs=[chat, q])
    q.submit(on_ask, inputs=[img, q, chat], outputs=[chat, q])

    gr.Examples(
        examples=["What is in this image?", "What colors do you see?", "How many objects are there?", "Where was this taken?"],
        inputs=q,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
