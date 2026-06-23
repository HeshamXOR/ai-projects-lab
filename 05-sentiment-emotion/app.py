"""Sentiment & Emotion Analyzer — Gradio preview.

Analyze a single piece of text, or upload a CSV to score a whole column.
"""

from __future__ import annotations

import csv
import io

import gradio as gr

import analyzer


def on_analyze(text):
    res = analyzer.analyze(text)
    if not res.emotions:
        return "Enter some text to analyze."
    top_emotion = next(iter(res.emotions))
    bars = "\n".join(
        f"- **{label}**: {prob:.0%}" for label, prob in list(res.emotions.items())[:6]
    )
    return (
        f"## Sentiment: **{res.sentiment}** ({res.sentiment_score:.0%} confidence)\n"
        f"## Top emotion: **{top_emotion}**\n\n"
        f"### Emotion breakdown\n{bars}"
    )


def on_batch(file):
    if file is None:
        return None, "Upload a CSV with a 'text' column."
    try:
        with open(file.name, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        return None, f"Could not read CSV: {e}"
    if not rows:
        return None, "CSV is empty."
    col = "text" if "text" in rows[0] else list(rows[0].keys())[0]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([col, "sentiment", "sentiment_score", "top_emotion"])
    for r in rows[:200]:  # cap for the demo
        a = analyzer.analyze(r.get(col, ""))
        top = next(iter(a.emotions), "—")
        writer.writerow([r.get(col, ""), a.sentiment, a.sentiment_score, top])
    # write to a temp file Gradio can serve
    path = "batch_results.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(out.getvalue())
    return path, f"Scored {min(len(rows), 200)} rows (column '{col}')."


with gr.Blocks(title="Sentiment & Emotion", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 😊 Sentiment & Emotion Analyzer\n"
        "Classify text as positive / negative / neutral **and** detect fine-grained "
        "emotions (joy, anger, sadness, fear, love, surprise). Single text or batch CSV."
    )
    gr.Markdown(analyzer.device_label())
    with gr.Tab("Single text"):
        text = gr.Textbox(
            label="Text",
            lines=4,
            value="I absolutely loved this product, it exceeded all my expectations!",
        )
        btn = gr.Button("Analyze", variant="primary")
        out = gr.Markdown()
        btn.click(on_analyze, inputs=text, outputs=out)
        gr.Examples(
            examples=[
                "This is the worst experience I've ever had.",
                "I'm a little nervous about the results tomorrow.",
                "What a pleasant surprise, thank you so much!",
            ],
            inputs=text,
        )
    with gr.Tab("Batch CSV"):
        gr.Markdown("Upload a CSV with a **`text`** column (or the first column is used).")
        csv_in = gr.File(label="CSV", file_types=[".csv"])
        batch_btn = gr.Button("Score CSV", variant="primary")
        csv_out = gr.File(label="Results CSV")
        batch_status = gr.Markdown()
        batch_btn.click(on_batch, inputs=csv_in, outputs=[csv_out, batch_status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
