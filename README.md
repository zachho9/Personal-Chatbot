# A Personal Chatbot

This project is about creating a personal AI chatbot for my daily use. It allows me to switch between major LLM providers using APIs. It comes with both a CLI and a web UI (powered by Chainlit).

## Features

- **Multi-provider support**: OpenAI, Google, Anthropic
- **Dual interface**: CLI (`main.py`) and web UI (`webui.py`) — both fully functional and independent
- **In-conversational memory**: conversation history is saved using LangGraph's checkpointer, so context is maintained within each conversation
- **Cross-conversational memory**: long-term memory using LangGraph's InMemoryStore to remember user preferences and facts across different conversations, persisted to a JSON file on disk
- **Thread management**: start new conversations or continue existing ones
- **Streaming responses**: the web UI streams LLM responses token by token
- **Chat history persistence**: the web UI stores conversation history in a local SQLite database with a sidebar for browsing and resuming past conversations

## Tech Stack

- Python 3.11+
- LangGraph for agent orchestration and state management
- LangChain for LLM provider integrations
- Chainlit for the web UI
- SQLite for persistent conversation storage
- uv for package management

## Setup

1. Clone the repo
2. Install dependencies:
   ```
   uv sync
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your-key-here
   GOOGLE_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   CHAINLIT_AUTH_SECRET=your-random-secret-here
   ```
   You can generate the auth secret with:
   ```
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
4. Initialize the web UI database (only needed once):
   ```
   uv run python init_db.py
   ```

## Usage

### CLI
```
uv run main.py
```

You will be prompted to choose an LLM provider, then choose whether to start a new conversation or continue an existing one. Type `exit` to return to the main menu, or choose quit to exit the program.

### Web UI
```
uv run chainlit run webui.py -w
```

A browser tab will automatically open at `http://localhost:8000`. Log in with the default credentials (`user` / `user`), select an LLM provider from the dropdown, and start chatting. Past conversations are available in the sidebar on the left.