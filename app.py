import os
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
import streamlit as st

llm = OllamaLLM(model="gemma3", temperature=0)

# define State Structure
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

#Define Node function

def categorize(state: State) -> State:
    """Categorize the customer query into Technical, Billing, or General."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. Respond with only the category name. No explanation. Query: {query}"
    )
    chain = prompt | llm
    category = chain.invoke({"query": state["query"]})
    return {"category": category}

def analyze_sentiments(state: State) -> State:
    """Analyze the sentiment of the customer query Positive, Neutral, or Negative."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following cutomer query. "
        "Respont with either 'Positive', 'Neutral' or 'Negative'."
        "No explanation, just the sentiment.\n\nQuery: {query}"
    )
    chain = prompt | llm
    sentiment = chain.invoke({"query": state["query"]})
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    """Provide a technical support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]})
    return {"response": response}

def handle_billing(state: State) -> State:
    """Provide a billing support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]})
    return {"response": response}

def handle_general(state: State) -> State:
    """Provide a general support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]})
    return {"response": response}

def escalate(state: State) -> State:
    """Escalate the query to a human agent due to negative sentiment."""
    return {"response": "This query has been escalated to a human agent due to its negative sentiment."}

def route_query(state: State) -> str:
    """Route the query based on its sentiment and category."""
    if "sentiment" in state and state["sentiment"].strip().lower() == "negative":
        return "escalate"
    elif "category" in state:
        category = state["category"].strip().lower()
        if category == "technical":
            return "handle_technical"
        elif category == "billing":
            return "handle_billing"
    return "handle_general"
    


# create and cofigure_graph

workflow = StateGraph(State)

# Add nodes

workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiments", analyze_sentiments)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

# add edges

workflow.add_edge("categorize", "analyze_sentiments")
workflow.add_conditional_edges(
    "analyze_sentiments",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)

workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

# set entry point
workflow.set_entry_point("categorize")

# compile the graph
app = workflow.compile()


st.set_page_config(page_title="Customer Support Bot", page_icon="🤖")

st.title("🤖 AI Customer Support Assistant")
query = st.text_area("Enter your query here:")

if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            result = app.invoke({"query": query})
        st.success("Query Processed ✅")

        st.markdown("### Results")
        st.markdown(f"**Category:** {result['category']}")
        st.markdown(f"**Sentiment:** {result['sentiment']}")
        st.markdown(f"**Response:** {result['response']}")


