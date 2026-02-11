import os
import uuid
import json
import sqlite3
from pprint import pprint
from typing import TypedDict, List, Union, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import SystemMessage
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

 
provider = input("Enter LLM provider:\n(1) OpenAI\n(2) Google\n(3) Anthropic\n").strip().lower()
if provider == "1":
    llm = ChatOpenAI(model="gpt-4o-mini", 
                     temperature=0.0, 
                     max_tokens=None,
                     timeout=None,
                     max_retries=2)
elif provider == "2":
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 temperature=0.0,
                                 max_tokens=None,
                                 timeout=None,
                                 max_retries=2)
elif provider == "3":
    llm = ChatAnthropic(model="claude-haiku-4-5",
                        temperature=0.0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2)


# Graph nodes
# ======================================================
def process(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> AgentState:
    """This node will solve the request you input"""
    user_id = config["configurable"].get("user_id", "default")
    memories = store.search(("memories", user_id))
    
    messages = state["messages"]
    if memories:
        memory_text = "\n".join(item.value.get("memory", "") for item in memories)
        system_msg = SystemMessage(content=f"You have the following memories about this user:\n{memory_text}\nUse these to personalize your responses.")
        messages = [system_msg] + messages

    response = llm.invoke(messages)

    print(f"\nAI: {response.content}")

    return {"messages": [AIMessage(content=response.content)]}


def write_memory(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> AgentState:
    """Extract and save user facts from the conversation"""
    user_id = config["configurable"].get("user_id", "default")
    
    extract_response = llm.invoke(
        state["messages"] + [HumanMessage(content="Extract any new facts or preferences about the user from this conversation. If there are new facts, respond with just the facts, one per line. If there are no new facts, respond with exactly 'NONE'.")]
    )
    
    response_text = extract_response.content.strip()
    if response_text != "NONE":
        memory_id = str(uuid.uuid4())
        store.put(("memories", user_id), memory_id, {"memory": response_text})
    
    return {"messages": []}


# Helper functions: short-term memory (thread summaries)
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

        
# Helper functions: long-term memory (cross-thread user facts)
# ======================================================
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


# Graph setup
# ======================================================
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_node("write_memory", write_memory)
graph.add_edge(START, "process")
graph.add_edge("process", "write_memory")
graph.add_edge("write_memory", END)


# Database and memory initialization
# ======================================================
os.makedirs("./database", exist_ok=True)
db_connection = sqlite3.connect("./database/chatbot_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(db_connection)

user_id = "default"
memory_store = InMemoryStore()
restore_memories_to_store(memory_store, user_id)

agent = graph.compile(checkpointer=checkpointer, store=memory_store)


# Main conversation loop
# ======================================================
while True:
    choice = input("Choose from following:\n(1) New conversation\n(2) Continue an existing one\n(3) Quit\n").strip().lower()
    
    if choice == "3":
        break
    elif choice == "2":
        threads = []
        for checkpoint in checkpointer.list(None):
            tid = checkpoint.config["configurable"]["thread_id"]
            if tid not in threads:
                threads.append(tid)
        
        if not threads:
            print("No existing conversations found. Starting a new one.\n")
            thread_id = str(uuid.uuid4())
            print(f"New conversation started. Your thread ID is: {thread_id}\n")
        else:
            summaries = load_summaries()
            print("\nExisting conversations:")
            for i, tid in enumerate(threads, 1):
                label = summaries.get(tid, tid)
                print(f"  {i}. {label}")
            pick = int(input("\nEnter the number of the conversation to continue: "))
            thread_id = threads[pick - 1]
    else:
        thread_id = str(uuid.uuid4())
        print(f"New conversation started. Your thread ID is: {thread_id}\n")

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    user_input = input("Enter your message: ")
    while user_input != "exit":
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        user_input = input("Enter your message: ")
    
    save_memories_to_disk(memory_store, user_id)
    summaries = load_summaries()
    state = agent.get_state(config)
    if state.values.get("messages") and thread_id not in summaries:
        summary_response = llm.invoke(
            state.values["messages"] + [HumanMessage(content="Summarize this entire conversation in no more than 10 words using the same major language as the conversation. Respond with plain text only.")],
        )
        save_summary(thread_id, summary_response.content)

    print("\nReturning to main menu...\n")