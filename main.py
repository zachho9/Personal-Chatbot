import os
import uuid
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

 
provider = input("Enter LLM provider (openai/google/anthropic): ").strip().lower()
if provider == "openai":
    llm = ChatOpenAI(model="gpt-4o-mini", 
                     temperature=0.0, 
                     max_tokens=None,
                     timeout=None,
                     max_retries=2)
elif provider == "google":
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 temperature=0.0,
                                 max_tokens=None,
                                 timeout=None,
                                 max_retries=2)
elif provider == "anthropic":
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


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 

db_connection = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(db_connection)
agent = graph.compile(checkpointer=checkpointer)



choice = input("Start a (n)ew conversation or (c)ontinue an existing one? ").strip().lower()
if choice == "c":
    thread_id = input("Enter the thread ID to continue: ").strip()
else:
    thread_id = str(uuid.uuid4())
    print(f"New conversation started. Your thread ID is: {thread_id}")
    print("(Save this ID if you want to continue this conversation later)\n")

config = {"configurable": {"thread_id": thread_id}}

user_input = input("Enter your message: ")
while user_input != "exit":
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    user_input = input("Enter your message: ")