# A Personal Chatbot

This project is about creating a personal AI chatbot for my daily use. It allows me to switch between major LLM providers using APIs.

## Features

- Multi-provider support: OpenAI (gpt-4o-mini), Google (gemini-2.5-flash), Anthropic (claude-haiku-4-5)
- In-conversational memory: conversation history is saved to a local SQLite database using LangGraph's checkpointer, so conversations persist across program restarts
- Cross-conversational memory (planned): long-term memory using LangGraph's Store interface to remember user preferences and facts across different conversations
- Thread management: start new conversations or continue existing ones by thread ID

## Tech Stack

- Python 3.11+
- LangGraph for agent orchestration and state management
- LangChain for LLM provider integrations
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
   ```

## Usage

```
uv run main.py
```

You will be prompted to choose an LLM provider, then choose whether to start a new conversation or continue an existing one. Type `exit` to quit.
