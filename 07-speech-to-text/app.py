"""Speech-to-Text + Summary — Gradio preview.

Upload or record audio, get a transcript and (for longer audio) a summary.
"""

from __future__ import annotations

import gradio as gr

import transcribe as t


def on_transcribe(audio, do_summary):
    if audio is None:
        return "Upload or record audio first.", ""
    res = t.transcribe(audio, do_summary=do_summary)
    summary = res.summary or "_(Summary shown for longer audio — this clip was short.)_"
    return res.transcript, summary


with gr.Blocks(title="Speech-to-Text + Summary", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎙️ Speech-to-Text + Summary\n"
        "Upload or record audio to get an accurate transcript (OpenAI Whisper), "
        "plus an automatic summary for longer recordings — great for meetings, "
        "lectures, and voice notes."
    )
    gr.Markdown(t.device_label())
    with gr.Row():
        audio = gr.Audio(type="filepath", label="Audio (upload or record)")
        do_summary = gr.Checkbox(value=True, label="Also summarize (for long audio)")
    btn = gr.Button("Transcribe", variant="primary")
    with gr.Row():
        transcript = gr.Textbox(label="Transcript", lines=10)
        summary = gr.Textbox(label="Summary", lines=10)

    btn.click(on_transcribe, inputs=[audio, do_summary], outputs=[transcript, summary])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
