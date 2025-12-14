import os
from pprint import pprint
from typing import TypedDict, List, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

 
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

    state["messages"].append(AIMessage(content=response.content)) 
    print(f"\nAI: {response.content}")
    print("\nCURRENT STATE: ", state["messages"])

    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()


conversation_history = []

user_input = input("Enter your message: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter your message: ")