from backend import run_travel_agents

user_input = input("Enter travel request: ")

res = run_travel_agents(
    user_input=user_input,
    thread_id="user_test"
)

print("\n FINAL ANSWER ")
print(res["answer"])

print("\n FLIGHT RESULTS ")
print(res["flight_results"])

print("\nHOTEL RESULTS ")
print(res["hotel_results"])

print("\nITINERARY ")
print(res["itinerary"])

print("\n LLM CALLS ")
print(res["llm_calls"])

print("\n THREAD ID ")
print(res["thread_id"])