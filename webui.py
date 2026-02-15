import os
import uuid
import json
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer


import sqlite3 as _sqlite3
import json as _json

# Register adapters so SQLite can handle Python lists (used by Chainlit for tags)
_sqlite3.register_adapter(list, lambda l: _json.dumps(l))
_sqlite3.register_converter("TEXT", lambda b: b.decode("utf-8"))


load_dotenv()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Simple local auth - customize as you like
    if (username, password) == ("user", "user"):
        return cl.User(
            identifier="user",
            metadata={"role": "user"}
        )
    return None


# ======================================================
# State definition (same as main.py)
# ======================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ======================================================
# LLM factory
# ======================================================
def get_llm(provider: str):
    if provider == "OpenAI":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_retries=2)
    elif provider == "Google":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_retries=2)
    elif provider == "Anthropic":
        return ChatAnthropic(model="claude-haiku-4-5", temperature=0.0, max_retries=2)


# ======================================================
# Graph node functions
# ======================================================
def make_process_node(llm):
    async def process(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> AgentState:
        user_id = config["configurable"].get("user_id", "default")
        memories = store.search(("memories", user_id))

        messages = state["messages"]
        if memories:
            memory_text = "\n".join(item.value.get("memory", "") for item in memories)
            system_msg = SystemMessage(content=f"You have the following memories about this user:\n{memory_text}\nUse these to personalize your responses.")
            messages = [system_msg] + messages

        response = await llm.ainvoke(messages)
        return {"messages": [AIMessage(content=response.content)]}
    return process


def make_write_memory_node(llm):
    async def write_memory(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> AgentState:
        user_id = config["configurable"].get("user_id", "default")
        extract_response = await llm.ainvoke(
            state["messages"] + [HumanMessage(content="Extract any new facts or preferences about the user from this conversation. If there are new facts, respond with just the facts, one per line. If there are no new facts, respond with exactly 'NONE'.")]
        )
        response_text = extract_response.content.strip()
        if response_text != "NONE":
            memory_id = str(uuid.uuid4())
            store.put(("memories", user_id), memory_id, {"memory": response_text})
        return {"messages": []}
    return write_memory


# ======================================================
# Helper functions (same as main.py)
# ======================================================
def load_summaries():
    if os.path.exists("./database/thread_summaries.json"):
        with open("./database/thread_summaries.json", "r") as f:
            return json.load(f)
    return {}

def save_summary(thread_id, summary):
    summaries = load_summaries()
    summaries[thread_id] = summary
    with open("./database/thread_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

def load_memories_from_disk():
    if os.path.exists("./database/user_memories.json"):
        with open("./database/user_memories.json", "r") as f:
            return json.load(f)
    return {}

def save_memories_to_disk(store, user_id):
    memories = store.search(("memories", user_id))
    data = {}
    for item in memories:
        data[item.key] = item.value
    with open("./database/user_memories.json", "w") as f:
        json.dump(data, f, indent=2)

def restore_memories_to_store(store, user_id):
    data = load_memories_from_disk()
    for key, value in data.items():
        store.put(("memories", user_id), key, value)


# ======================================================
# Build graph + agent for a given LLM
# ======================================================
def build_agent(llm, checkpointer, memory_store):
    graph = StateGraph(AgentState)
    graph.add_node("process", make_process_node(llm))
    graph.add_node("write_memory", make_write_memory_node(llm))
    graph.add_edge(START, "process")
    graph.add_edge("process", "write_memory")
    graph.add_edge("write_memory", END)
    return graph.compile(checkpointer=checkpointer, store=memory_store)


# ======================================================
# Shared infrastructure (initialized once)
# ======================================================
os.makedirs("./database", exist_ok=True)
checkpointer = MemorySaver()
user_id = "default"
memory_store = InMemoryStore()
restore_memories_to_store(memory_store, user_id)


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///database/webui_history.db")


# ======================================================
# Chainlit: Chat Profiles (LLM provider selection)
# ======================================================
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(name="OpenAI", markdown_description="Use **GPT-4o-mini**"),
        cl.ChatProfile(name="Google", markdown_description="Use **Gemini 2.5 Flash**"),
        cl.ChatProfile(name="Anthropic", markdown_description="Use **Claude Haiku 4.5**"),
    ]


# ======================================================
# Chainlit: Chat start
# ======================================================
@cl.on_chat_start
async def on_chat_start():
    profile = cl.user_session.get("chat_profile")
    llm = get_llm(profile)
    agent = build_agent(llm, checkpointer, memory_store)

    # Store in session
    cl.user_session.set("agent", agent)
    cl.user_session.set("llm", llm)
    cl.user_session.set("thread_id", str(uuid.uuid4()))

    # Save the profile to thread metadata for resume
    cl.context.session.thread_metadata = {"chat_profile": profile}
    
    await cl.Message(content=f"Chat started with **{profile}**. How can I help you?").send()


# ======================================================
# Chainlit: Handle messages with streaming
# ======================================================
@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    llm = cl.user_session.get("llm")
    thread_id = cl.user_session.get("thread_id")

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    # Stream the response token by token
    msg = cl.Message(content="")

    # Use astream_events to get token-level streaming
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message.content)]},
        config=config,
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            # Only stream tokens from the "process" node (not write_memory)
            if event.get("metadata", {}).get("langgraph_node") == "process":
                token = event["data"]["chunk"].content
                if token:
                    await msg.stream_token(token)

    await msg.send()

    # Save memories to disk after each message
    save_memories_to_disk(memory_store, user_id)

# chat resume handler
@cl.on_chat_resume
async def on_chat_resume(thread):
    profile = thread.get("metadata", {}).get("chat_profile", "OpenAI")
    llm = get_llm(profile)
    agent = build_agent(llm, checkpointer, memory_store)

    thread_id = thread.get("id", str(uuid.uuid4()))

    cl.user_session.set("agent", agent)
    cl.user_session.set("llm", llm)
    cl.user_session.set("thread_id", thread_id)