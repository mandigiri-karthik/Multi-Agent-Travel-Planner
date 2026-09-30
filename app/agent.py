from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from nodes.trip_requirements import ask_follow_up, extract_trip, finalize
from utils.state import TravelState

builder = StateGraph(TravelState)
builder.add_node(
    "extract_trip",
    extract_trip,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0),
)
builder.add_node("ask_follow_up", ask_follow_up)
builder.add_node("finalize", finalize)

builder.add_edge(START, "extract_trip")   # routing after this is done via Command(goto=...)
builder.add_edge("ask_follow_up", END)
builder.add_edge("finalize", END)

# If you run through langgraph.json (LangGraph Server / Studio), compile without
# a checkpointer, because the platform provides one.
graph = builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # fresh conversation per run

    print("Trip planner ready. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print(f"\nAgent: {result['messages'][-1].content}\n")

        if result.get("trip_request"):
            print("(Trip request complete. Type a new trip to start over.)\n")