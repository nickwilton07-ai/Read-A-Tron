# Book Suggester

A cozy, AI-powered book recommendation chat app built with Python and Gradio.

## Stack
- **Python** with **Gradio** (UI framework)
- **OpenAI Responses API** (`client.responses.create()`) with `file_search` tool
- Pre-loaded vector store of books (ID stored in `config.json`)

## How to Run
```
python app.py
```
Binds to `0.0.0.0` and the port from the `PORT` env var (default 7860).

## Environment Variables / Secrets
| Name | Required | Description |
|------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key — add via Replit Secrets |

## Features
- 💬 Streaming chat with the OpenAI Responses API
- 📚 Book search via file_search against the pre-built vector store
- 🗂️ Private reading list (per session)
- 🔍 Format, genre, and age-group filters
- 📖 Expandable source citations
- ✨ Suggested starter questions

## Config (`config.json`)
- `assistant_name` — display name shown in the UI
- `assistant_instructions` — system prompt sent to the model
- `model` — OpenAI model to use
- `vector_store_id` — pre-built vector store (do not rebuild)

## User Preferences
- Keep the cozy, minimal, friendly visual style
- Do not rebuild or replace the vector store
