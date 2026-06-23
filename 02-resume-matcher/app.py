"""Resume <-> Job Matcher — Gradio preview.

Paste a resume and a job description; get a match score, matched skills, and a
gap analysis. CPU-only, no API key.
"""

from __future__ import annotations

import gradio as gr

from matcher import Matcher

matcher = Matcher()


def _chips(items, empty_msg):
    if not items:
        return f"_{empty_msg}_"
    return " ".join(f"`{s}`" for s in items)


def on_match(resume, jd):
    res = matcher.match(resume, jd)
    report = (
        f"## Match score: **{res.score}/100**\n"
        f"{res.verdict}\n\n"
        f"### ✅ Matched skills\n{_chips(res.matched, 'none')}\n\n"
        f"### ⚠️ Missing (in the job, not your resume)\n{_chips(res.missing, 'none — great coverage!')}\n\n"
        f"### ➕ Extra skills you bring\n{_chips(res.extra, 'none detected')}"
    )
    return report


SAMPLE_RESUME = (
    "Software engineer with 4 years building web apps in Python and JavaScript. "
    "Experienced with React, REST APIs, Docker, and PostgreSQL. Led a small team "
    "and mentored juniors. Comfortable with Git and agile workflows."
)
SAMPLE_JD = (
    "We are hiring a backend engineer proficient in Python and SQL. The role "
    "involves building REST APIs, deploying with Docker and Kubernetes on AWS, "
    "and collaborating in an agile team. Machine learning experience is a plus."
)

with gr.Blocks(title="Resume ↔ Job Matcher", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎯 Resume ↔ Job Matcher\n"
        "Paste your resume and a job description to get a semantic match score "
        "and a concrete gap analysis — see exactly which required skills you're "
        "missing. Runs entirely on CPU."
    )
    with gr.Row():
        resume = gr.Textbox(label="Your resume", lines=14, value=SAMPLE_RESUME)
        jd = gr.Textbox(label="Job description", lines=14, value=SAMPLE_JD)
    btn = gr.Button("Analyze match", variant="primary")
    out = gr.Markdown()

    btn.click(on_match, inputs=[resume, jd], outputs=out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
