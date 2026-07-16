import gradio as gr
import openai
import os
import json
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("config.json") as f:
    config = json.load(f)

ASSISTANT_NAME = config["assistant_name"]
ASSISTANT_INSTRUCTIONS = config["assistant_instructions"]
VECTOR_STORE_ID = config["vector_store_id"]
MODEL = config["model"]

WELCOME_MESSAGE = "Ask me about the type of book you want to read. 📚"

STARTER_QUESTIONS = [
    "🌧️  Cozy mystery for a rainy day?",
    "🎨  Great graphic novels for adults?",
    "🚀  Best sci-fi of the last decade?",
    "✨  Fantasy series for young adults?",
]

FORMATS = ["All Formats", "Novel", "Graphic Novel", "Comic Book", "Short Stories", "Novella", "Picture Book"]
GENRES = ["All Genres", "Fantasy", "Science Fiction", "Mystery", "Romance", "Horror",
          "Historical Fiction", "Biography", "Self-Help", "Children's", "Thriller", "Literary Fiction"]
AGE_GROUPS = ["All Ages", "Children (0–8)", "Middle Grade (8–12)", "Young Adult (12–18)", "Adult (18+)"]


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def get_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please add it in the Secrets panel.")
    return openai.OpenAI(api_key=api_key)


def extract_citations(output) -> list[str]:
    seen, citations = set(), []
    for item in output:
        contents = getattr(item, "content", [])
        for block in contents:
            for ann in getattr(block, "annotations", []):
                if getattr(ann, "type", "") == "file_citation":
                    fname = getattr(ann, "filename", "Unknown source")
                    if fname not in seen:
                        seen.add(fname)
                        citations.append(fname)
    return citations


def build_instructions(reading_list: list[str]) -> str:
    base = ASSISTANT_INSTRUCTIONS
    if reading_list:
        titles = ", ".join(f'"{t}"' for t in reading_list)
        base += (
            f"\n\nThe user already has these books on their reading list: {titles}. "
            "Avoid re-recommending them unless specifically asked."
        )
    return base


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def respond(
    message: str,
    history: list,
    prev_response_id: Optional[str],
    reading_list: list[str],
    fmt_filter: str,
    genre_filter: str,
    age_filter: str,
):
    """Streaming chat generator. Yields (partial_text, updated_prev_id, reading_list)."""
    client = get_client()

    # Attach active filters to the user message
    active_filters = [
        v for k, v in [
            ("Format", fmt_filter), ("Genre", genre_filter), ("Age group", age_filter)
        ]
        if v and not v.startswith("All")
    ]
    user_input = message
    if active_filters:
        user_input = f"[Filters — {', '.join(active_filters)}]  {message}"

    kwargs = dict(
        model=MODEL,
        instructions=build_instructions(reading_list),
        input=user_input,
        tools=[{"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}],
        stream=True,
    )
    if prev_response_id:
        kwargs["previous_response_id"] = prev_response_id

    try:
        accumulated = ""
        final_response = None

        stream = client.responses.create(**kwargs)
        for event in stream:
            if event.type == "response.output_text.delta":
                accumulated += event.delta
                yield accumulated, prev_response_id, reading_list
            if event.type == "response.completed":
                final_response = event.response

        new_prev_id = final_response.id if final_response else prev_response_id

        # Append expandable citations
        citations = extract_citations(final_response.output) if final_response else []
        if citations:
            cite_lines = "\n".join(f"- {c}" for c in citations)
            accumulated += f"\n\n<details><summary>📚 Sources ({len(citations)})</summary>\n\n{cite_lines}\n\n</details>"

        yield accumulated, new_prev_id, reading_list

    except Exception as exc:
        yield f"⚠️  {exc}", prev_response_id, reading_list


# ---------------------------------------------------------------------------
# Reading list helpers
# ---------------------------------------------------------------------------

def add_to_list(title: str, reading_list: list[str]) -> tuple[list[str], str, str]:
    title = title.strip()
    if not title:
        return reading_list, render_list(reading_list), "Please enter a book title."
    if title in reading_list:
        return reading_list, render_list(reading_list), f'"{title}" is already on your list.'
    reading_list = reading_list + [title]
    return reading_list, render_list(reading_list), f'✅ Added "{title}"'


def remove_from_list(title: str, reading_list: list[str]) -> tuple[list[str], str, str]:
    updated = [t for t in reading_list if t != title]
    return updated, render_list(updated), f'Removed "{title}"'


def render_list(reading_list: list[str]) -> str:
    if not reading_list:
        return "_Your reading list is empty._"
    items = "".join(f"<li>{t}</li>" for t in reading_list)
    return f"<ol>{items}</ol>"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* ── Cozy colour palette ── */
:root {
    --cream:   #fdf6ee;
    --tan:     #e8d9c5;
    --brown:   #7c5c3e;
    --forest:  #4a6741;
    --accent:  #c97c3a;
    --soft-shadow: 0 2px 12px rgba(124,92,62,0.12);
}

body, .gradio-container { background: var(--cream) !important; font-family: 'Georgia', serif; }

/* Header */
#app-header {
    text-align: center;
    padding: 28px 16px 8px;
    border-bottom: 2px solid var(--tan);
    margin-bottom: 16px;
}
#app-header h1 { color: var(--brown); font-size: 2rem; margin: 0; letter-spacing: .02em; }
#app-header p  { color: var(--forest); font-size: 1rem; margin: 4px 0 0; }

/* Chatbot */
#chatbot { border: 1.5px solid var(--tan) !important; border-radius: 12px !important;
           background: #fff !important; box-shadow: var(--soft-shadow); }
#chatbot .message.user   { background: var(--tan)   !important; color: #3a2a1a !important; }
#chatbot .message.bot    { background: #fff          !important; color: #2a2010 !important; }

/* Input row */
#msg-box textarea { border-radius: 10px !important; border: 1.5px solid var(--tan) !important;
                    background: #fffdf8 !important; }
#send-btn { background: var(--accent) !important; color: #fff !important;
            border-radius: 10px !important; font-weight: bold; }
#send-btn:hover { background: var(--brown) !important; }

/* Starters */
.starter-btn { background: var(--cream) !important; border: 1.5px solid var(--tan) !important;
               color: var(--brown) !important; border-radius: 20px !important;
               font-size: .88rem !important; padding: 6px 14px !important; }
.starter-btn:hover { background: var(--tan) !important; }

/* Filter panel */
#filter-panel { background: #fffdf8; border: 1.5px solid var(--tan);
                border-radius: 12px; padding: 12px 16px; margin-bottom: 12px;
                box-shadow: var(--soft-shadow); }
#filter-panel label { color: var(--brown) !important; font-weight: bold; }

/* Reading list */
#reading-list-panel { background: #fffdf8; border: 1.5px solid var(--tan);
                      border-radius: 12px; padding: 12px 16px;
                      box-shadow: var(--soft-shadow); }
#reading-list-panel h3 { color: var(--brown); margin: 0 0 8px; font-size: 1rem; }
#reading-list-display ol { padding-left: 18px; margin: 0; }
#reading-list-display li { color: var(--forest); padding: 3px 0; font-size: .9rem; }

/* Accordion */
details summary { cursor: pointer; color: var(--forest); font-size: .85rem; }

/* Clear btn */
#clear-btn { background: transparent !important; border: 1px solid var(--tan) !important;
             color: var(--brown) !important; border-radius: 8px !important; }
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Book Suggester") as demo:

        # ── State ──
        prev_id    = gr.State(None)
        read_list  = gr.State([])

        # ── Header ──
        gr.HTML(f"""
        <div id="app-header">
          <h1>📖 {ASSISTANT_NAME}</h1>
          <p>{WELCOME_MESSAGE}</p>
        </div>
        """)

        with gr.Row():
            # ── Left sidebar ──
            with gr.Column(scale=1, min_width=240):

                # Filters
                with gr.Group(elem_id="filter-panel"):
                    gr.Markdown("### 🔍 Filters")
                    fmt_dd   = gr.Dropdown(FORMATS,    value="All Formats", label="Format",    interactive=True)
                    genre_dd = gr.Dropdown(GENRES,     value="All Genres",  label="Genre",     interactive=True)
                    age_dd   = gr.Dropdown(AGE_GROUPS, value="All Ages",    label="Age group", interactive=True)

                gr.HTML("<br>")

                # Reading list
                with gr.Group(elem_id="reading-list-panel"):
                    gr.HTML("<h3>🗂️ My Reading List</h3>")
                    list_display = gr.HTML(
                        value="_Your reading list is empty._",
                        elem_id="reading-list-display",
                    )
                    with gr.Row():
                        list_input  = gr.Textbox(placeholder="Add a book title…", show_label=False, scale=3)
                        add_btn     = gr.Button("＋", scale=1, size="sm")
                    list_status = gr.Markdown("")

                    # Remove dropdown (populated from state)
                    remove_dd  = gr.Dropdown(choices=[], label="Remove a book", interactive=True, visible=False)
                    remove_btn = gr.Button("Remove", size="sm", visible=False)

            # ── Main chat area ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=480,
                    show_label=False,
                    render_markdown=True,
                )

                # Starter question buttons
                with gr.Row():
                    starter_btns = [
                        gr.Button(q, elem_classes=["starter-btn"], size="sm")
                        for q in STARTER_QUESTIONS
                    ]

                with gr.Row():
                    msg_box  = gr.Textbox(
                        placeholder="Ask about a book, genre, or mood…",
                        show_label=False,
                        scale=5,
                        elem_id="msg-box",
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")
                    clear_btn = gr.Button("Clear", scale=1, elem_id="clear-btn")

        # ── Wiring ──

        def user_submit(message, history):
            """Append user turn immediately."""
            return "", history + [{"role": "user", "content": [{"type": "text", "text": message}]}]

        def bot_stream(history, pid, rl, fmt, genre, age):
            message = history[-1]["content"][0]["text"]
            base_history = history[:-1]
            try:
                for text, new_pid, new_rl in respond(message, base_history, pid, rl, fmt, genre, age):
                    yield base_history + [
                        {"role": "user", "content": [{"type": "text", "text": message}]},
                        {"role": "assistant", "content": [{"type": "text", "text": text}]},
                    ], new_pid, new_rl
            except Exception as exc:
                import traceback
                print("BOT_STREAM ERROR:", exc, flush=True)
                traceback.print_exc()
                err_text = f"⚠️  {type(exc).__name__}: {exc}"
                yield base_history + [
                    {"role": "user", "content": [{"type": "text", "text": message}]},
                    {"role": "assistant", "content": [{"type": "text", "text": err_text}]},
                ], pid, rl

        # Send on button click
        (
            msg_box
            .submit(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue=False)
            .then(bot_stream, [chatbot, prev_id, read_list, fmt_dd, genre_dd, age_dd],
                  [chatbot, prev_id, read_list])
        )
        (
            send_btn
            .click(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue=False)
            .then(bot_stream, [chatbot, prev_id, read_list, fmt_dd, genre_dd, age_dd],
                  [chatbot, prev_id, read_list])
        )

        # Clear
        def clear_chat():
            return [], None

        clear_btn.click(clear_chat, outputs=[chatbot, prev_id])

        # Starter questions
        def set_msg(q):
            return q.split("  ", 1)[-1]  # strip emoji prefix

        for btn in starter_btns:
            (
                btn
                .click(set_msg, inputs=[btn], outputs=[msg_box])
                .then(user_submit, [msg_box, chatbot], [msg_box, chatbot], queue=False)
                .then(bot_stream, [chatbot, prev_id, read_list, fmt_dd, genre_dd, age_dd],
                      [chatbot, prev_id, read_list])
            )

        # Reading list — add
        def _add(title, rl):
            new_rl, html, status = add_to_list(title, rl)
            choices = new_rl if new_rl else []
            return new_rl, html, status, "", gr.update(choices=choices, visible=bool(choices)), gr.update(visible=bool(choices))

        add_btn.click(
            _add,
            inputs=[list_input, read_list],
            outputs=[read_list, list_display, list_status, list_input, remove_dd, remove_btn],
        )
        list_input.submit(
            _add,
            inputs=[list_input, read_list],
            outputs=[read_list, list_display, list_status, list_input, remove_dd, remove_btn],
        )

        # Reading list — remove
        def _remove(title, rl):
            new_rl, html, status = remove_from_list(title, rl)
            choices = new_rl if new_rl else []
            return new_rl, html, status, gr.update(choices=choices, value=None, visible=bool(choices)), gr.update(visible=bool(choices))

        remove_btn.click(
            _remove,
            inputs=[remove_dd, read_list],
            outputs=[read_list, list_display, list_status, remove_dd, remove_btn],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=port, css=CSS)
