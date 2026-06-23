"""Semantic Code Search — Gradio preview.

Point it at a folder of source code, then search functions/classes by natural
language. Defaults to indexing this project folder so the preview works
immediately.
"""

from __future__ import annotations

import os

import gradio as gr

from codesearch import CodeIndex

index = CodeIndex()
STATE = {"count": 0, "folder": ""}


def on_index(folder):
    folder = (folder or ".").strip()
    if not os.path.isdir(folder):
        return f"Not a folder: {folder}"
    n = index.index_folder(folder)
    STATE["count"] = n
    STATE["folder"] = folder
    if n == 0:
        return f"No code blocks found in {folder} (looked for .py/.js/.ts/.java/etc.)."
    return f"Indexed **{n}** functions/classes from `{folder}`. Search below."


def on_search(query):
    if STATE["count"] == 0:
        return "Index a folder first."
    hits = index.search(query)
    if not hits:
        return "No results."
    out = []
    for h in hits:
        lang = os.path.splitext(h.path)[1].lstrip(".") or "text"
        out.append(
            f"### `{h.name}`  —  {h.path}:{h.start_line}\n"
            f"```{lang}\n{h.code}\n```"
        )
    return "\n\n".join(out)


with gr.Blocks(title="Semantic Code Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🔎 Semantic Code Search\n"
        "Index a code folder and search functions/classes by **meaning**, not "
        "keywords. Ask things like *'where is the auth token validated?'* and get "
        "the relevant code back. Runs on CPU."
    )
    with gr.Row():
        folder = gr.Textbox(label="Folder to index", value=".", scale=3)
        index_btn = gr.Button("Index", variant="secondary", scale=1)
    status = gr.Markdown()
    query = gr.Textbox(label="Search query", placeholder="e.g. function that reads a PDF into chunks")
    search_btn = gr.Button("Search", variant="primary")
    results = gr.Markdown()

    index_btn.click(on_index, inputs=folder, outputs=status)
    search_btn.click(on_search, inputs=query, outputs=results)
    query.submit(on_search, inputs=query, outputs=results)

    # Index the current folder at startup so the preview is ready.
    demo.load(lambda: on_index("."), outputs=status)
    gr.Examples(
        examples=["read a file into chunks", "compute similarity between vectors", "launch the web app"],
        inputs=query,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
