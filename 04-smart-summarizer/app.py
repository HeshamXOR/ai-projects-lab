"""Smart Summarizer — Gradio preview.

Paste a long article, document, or meeting transcript; get a concise summary
plus extracted action items. GPU if available, else CPU.
"""

from __future__ import annotations

import gradio as gr

import summarizer

SAMPLE = (
    "In today's product sync the team reviewed the Q3 roadmap. Sarah reported that "
    "the new onboarding flow increased activation by 12 percent, though drop-off on "
    "the payment step remains high. The group agreed the payment UX is the top "
    "priority. Raj will run a usability study on the checkout by next Friday. "
    "Maria needs to coordinate with the design team on a simplified form. "
    "We also discussed infrastructure costs, which rose 8 percent last month due to "
    "increased inference traffic; the team should evaluate caching and a smaller "
    "default model. Action item: Tom will benchmark the cheaper model this week. "
    "Finally, the launch date was confirmed for the end of the quarter, pending the "
    "checkout fixes. Everyone agreed to reconvene next Tuesday to review progress."
)


def on_summarize(text):
    res = summarizer.summarize(text)
    items = "\n".join(f"- {it}" for it in res.action_items) or "_No clear action items detected._"
    md = (
        f"## Summary\n{res.summary}\n\n"
        f"_{res.compression}_\n\n"
        f"## ✅ Action items\n{items}"
    )
    return md


with gr.Blocks(title="Smart Summarizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 📝 Smart Summarizer\n"
        "Paste a long article, report, or meeting transcript and get a concise "
        "summary plus extracted action items. Handles long inputs via chunked "
        "(map-reduce) summarization."
    )
    gr.Markdown(summarizer.device_label())
    text = gr.Textbox(label="Text to summarize", lines=16, value=SAMPLE)
    btn = gr.Button("Summarize", variant="primary")
    out = gr.Markdown()

    btn.click(on_summarize, inputs=text, outputs=out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
