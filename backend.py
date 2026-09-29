import os
import certifi

from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

def get_databse_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add database_url to .env"
        )
    
    if "sslmode=" not in database_url:
        saperator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{saperator}sslmode=require"
    return database_url

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

# initialize llm
llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = GROQ_API_KEY
)

# define the state
class TravelState(TypedDict):
    message : Annotated[list[AnyMessage], operator.add]
    user_query : str
    flight_results : str
    hotel_results : str
    itinerary : str
    llm_calls : str

# Agent- 01
def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results" : flight_data,
        "message" : [
            AIMessage(content = "Flight results Fetched")
        ],
        "llm_calls" : state.get("llm_calls",0)+1
    }

# Agent-02 Hotel agent
def hotel_agent(state:TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query=query)

    return{
        "hotel_results" : hotel_results,
        "message" :[
            AIMessage ("Hotels information fetched")
        ],
        "llm_calls" : state.get('llm_calls',0)+1
    }

# Agent=03 itinerary agent
def itinerary_agent(state:TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query : {state['user_query']}

Flight results : {state['flight_results']}

Hotel results : {state['hotel_results']}

Make the itinerary practical, budget-aware and easy to follow.
"""    
    response = llm.invoke([
        SystemMessage(content= "You are a expert travel planner."),
        HumanMessage(content = prompt)
    ])

    return{
        "itinerary" : response.content,
        "message" : [response],
        "llm_calls" : state.get('llm_calls',0)+1
    }

# Agent-04 get all the response and combine the final response

def final_response_agent(state:TravelState):
    final_response_agent_prompt = f"""
Generate a final detailed and structured travel response for the user.

User Request:
User Query : {state['user_query']}

Flight results : {state['flight_results']}

Hotel results : {state['hotel_results']}

itinerary results : {state['hotel_results']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""
    response = llm.invoke([
        SystemMessage(content = "You are a professional AI travel booking agent"),
        AIMessage(content = final_response_agent_prompt)
    ])

    return {
        "message" : [response],
        "llm_calls" : state.get('llm_calls',0)+1
    }