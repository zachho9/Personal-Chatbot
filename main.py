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
    

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    print(f"\nAI: {response.content}")

    return {"messages": [AIMessage(content=response.content)]}


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


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 

db_connection = sqlite3.connect("./database/chatbot_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(db_connection)
agent = graph.compile(checkpointer=checkpointer)



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

    config = {"configurable": {"thread_id": thread_id}}

    user_input = input("Enter your message: ")
    while user_input != "exit":
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        user_input = input("Enter your message: ")
    
    summaries = load_summaries()
    state = agent.get_state(config)
    if state.values.get("messages") and thread_id not in summaries:
        summary_response = llm.invoke(
            state.values["messages"] + [HumanMessage(content="Summarize this entire conversation in no more than 10 words using the same major language as the conversation.")]
        )
        save_summary(thread_id, summary_response.content)

    print("\nReturning to main menu...\n")